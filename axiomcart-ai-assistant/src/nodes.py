
"""
Nodes used in Graph.
"""


from typing import Annotated, Literal

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from src.state import AxiomCartState, ClassificationResult, WorkerInput
from src.config import llm, get_logger
from langgraph.types import Send, Command, interrupt
from src.tools import escalate_to_human, get_similar_products, get_order_status
from src.data import SUPPORT_POLICIES

import operator

logger = get_logger("nodes")

PRODUCT_PROMPT = """\
You are the Product Discovery Agent for AxiomCart.

ROLE: Help customers find and learn about products. You also handle
general conversation (greetings, thanks, chitchat).

TOOLS:
  search_product_catalog – semantic search over our product database

GUIDELINES:
- For greetings or general chat, respond warmly without calling tools.
- For product questions, always search the catalog first.
- Highlight key features and prices.
- If a product is out of stock, suggest alternatives.
- If the search returns products the customer has already seen or that
  don't match what they asked for (wrong brand, wrong category, etc.),
  be honest and say we don't currently carry what they're looking for.
  Do NOT present irrelevant products as if they match the request.
- Keep responses concise and helpful.
"""

SUPPORT_PROMPT = f"""\
You are the Sales Support Agent for AxiomCart.

ROLE: Handle order enquiries and escalate issues to human agents.

TOOLS:
  get_order_status   – look up an order by order ID or customer email
  escalate_to_human  – create a ticket for human support (sends email notification)

POLICIES:
{SUPPORT_POLICIES}

GUIDELINES:
- If the customer has NOT provided an order ID or email, you MUST ask
  for it before calling any tools. Say something like: "Could you
  please provide your order ID (e.g. ORD101) or registered email
  address so I can look up your order?"
- Be empathetic and professional.
- Only call escalate_to_human when the customer explicitly asks for
  a human agent OR the issue cannot be resolved.
- After retrieving information, respond directly to the customer.
"""


def build_context(messages: list[AnyMessage]) -> str:

    parts = []
    if not messages:
        return ""
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistance: {m.content}")
    if not parts:
        return ""
    return f"CONVERSATION SO FAR: \n {"\n".join(parts)} \n\n"


class AgentState():
    messages: Annotated[list[AnyMessage], operator.add]


# Build subgraph for Product Catalog
product_tools = [get_similar_products]
product_tools_with_names = {tool.name: tool for tool in product_tools}

support_tools = [get_order_status, escalate_to_human]
support_tools_with_names = {tool.name: tool for tool in support_tools}

product_llm = llm.bind_tools(product_tools)
support_llm = llm.bind_tools(support_tools)

def product_model(state: AgentState):
    """Call the product LLM (with tools bound)."""
    result = product_llm.invoke(state['messages'])
    logger.info(f"[product:model] tool calls {bool(result.tool_calls)}")

    return {"messages": [result]}

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return END


def product_tools(state: AgentState) -> dict:
    last_message = state['messages'][-1]
    results = []
    for tc in last_message.tool_calls:
        name, args = tc['name'], tc['args']
        logger.info("[product:tools] %s(%s)", name, args)
        result = product_tools_with_names.get(name).invoke(args) if name in product_tools_with_names else f"Unknown tool {name}"
        results.append(ToolMessage(content=str(result), tool_call_id=tc['id']))
    return {"messages": results}

pb = StateGraph(AgentState)
pb.add_node("model", product_model)
pb.add_node("tools", product_tools)
pb.set_entry_point("model")
pb.add_conditional_edges("model", should_continue)
pb.add_edge("tools", "model")
product_subgraph = pb.compile()



def product_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    user_query = state['user_query']
    task_description = state['task_description']
    context = build_context(state['messages'])
    result = product_subgraph.invoke({"messages": [
        SystemMessage(content=PRODUCT_PROMPT),
        HumanMessage(content=f"{context} \nTask: {task_description}\n Customer Query: {user_query}")
    ]})

    answer = result['messages'][-1].content

    return Command(
        update={"agent_results": [{"source": "product_discovery", "response": answer}]},
        goto="synthesizer"
    )

# Build subgraph for Support Agent

def support_model(state: AgentState):
    messages = state['messages']
    response = support_llm.invoke(messages)
    logger.info(f"[support:model] tool calls {bool(response.tool_calls)}")
    
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in messages)
        if not any_tools_called:
            logger.info("[support:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info(f"[support:model] HITL: User replied {user_reply}")
            return {"messages": [response, HumanMessage(content=str(user_reply))]}
    return {"messages": [response]}


def support_tools(state: AgentState) -> dict:
    last_message = state['messages'][-1]
    results= []
    for tc in last_message.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[support:tools] %s(%s)", name, args)
        out = support_tools_with_names[name].invoke(args) if name in support_tools_with_names else f"Unknown tool: {name}"
        
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}

def support_should_continue(state: AgentState) -> str:
    """Route after support model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

sb = StateGraph(AgentState)
sb.add_node("model", support_model)
sb.add_node("tools", support_tools)
sb.set_entry_point("model")
sb.add_conditional_edges("model", support_should_continue)
sb.add_edge("tools", "model")

support_subgraph = sb.compile()


# Support Agent

def support_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:

    """Run the sales-support agent via its model ⇄ tools subgraph.

    HITL is handled through conversation persistence: if the agent
    needs info (e.g. order ID), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_description = state.get('task_description',user_query)
    logger.info("Support Agent  task=%r", task_description)

    context = build_context(state.get("messages", []))

    result = support_subgraph.invoke({"messages": [
        SystemMessage(content=SUPPORT_PROMPT),
        HumanMessage(content=f"{context}\n Task: {task_description} \n Customer Query:  {user_query}")
    ]})

    answer = result["messages"][-1].content
    return Command(
        update={"agent_results": [{"source": "sales_support", "response": answer}]},
        goto="synthesizer"

    )

def orchistrator_node(state: AxiomCartState) -> Command[Literal["product_agent", "support_agent", "synthesizer"]]:
    """
    The orchistrator node will decide which node the request should go to.
    """

    user_query = state.get("user_query")
    messages = state.get('messages')
    prompt = (
         f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{user_query}"\n\n'
        'AGENTS:\n'
        '  product_agent – product searches, recommendations, catalog questions,\n'
        '                  AND general conversation (greetings, thanks, chitchat)\n'
        '  support_agent   – order status, complaints, escalation to human support\n\n'
        'RULES:\n'
        '1. Greetings, chitchat, general questions (hi, hello, thanks, how are you)\n'
        '   → product_agent only\n'
        '2. Product-only queries  → product_agent only\n'
        '3. Order/support queries → support_agent only\n'
        '4. Mixed queries         → BOTH agents, requires_synthesis = true\n'
        '\nIMPORTANT: Only route to support_agent when the query clearly involves\n'
        'an order, complaint, or support issue. When in doubt, use product_agent.\n'
    )

    structured_llm = llm.with_structured_output(ClassificationResult)

    classification = structured_llm.invoke(prompt)
    targets = []

    for task in classification.tasks:
        targets.append(Send(task.agent, {
            "messages": messages,
            "user_query": state.get('user_query', []),
            "task_description": task.task_description
        }))
    

    return Command(
        update={
            "requires_synthesis": classification.requires_synthesis,
            "tasks": classification.tasks,
            "agent_results": [],
            "user_query": user_query
        },
        goto=targets
    )


def synthesizer_node(state: AxiomCartState) -> dict:
    """Merge results from one or more agents into a single user-facing reply."""
    results = state.get("agent_results", [])
    user_query = state.get('user_query')

    if not results:
        logger.warning("Synthesizer received no agent results")
        return {"final_answer": "Sorry, I couldn't process that request. Please try again."}
    
    if len(results) == 1:
        logger.info("Synthesizer  single-agent pass-through")
        return {"final_answer": results[0]["response"]}
    
    logger.info("Synthesizer  merging %d agent responses", len(results))

    parts = "\n\n".join(
        f"[{r['source'].upper()}] : \n {r['response']}" for r in results
    )
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'AxiomCart Assistant'."
    )

    merged = llm.invoke(prompt)
    return {"final_answer": merged.content}



