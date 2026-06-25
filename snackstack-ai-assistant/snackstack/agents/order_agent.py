

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from snackstack.tools.order_tools import get_order_status
from snackstack.config import llm
from snackstack.logger import get_logger
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from snackstack.agents.prompts import order_prompt

from snackstack.agents.utils import build_context

from snackstack.state import WorkerInput

logger = get_logger("order_agent")

# ----------------------------------------------------
# ORDER SUBGRAPH
# ----------------------------------------------------

llm_with_order_tools = llm.bind_tools([get_order_status])

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    tools_call_count: int


def order_model(state: AgentState):
    """Invoke the order-enabled language model on the current conversation state. """
    response = llm_with_order_tools.invoke(state['messages'])

    # If no tools calls and no tools have been called yet,
    # the agent is asking for missing into - interrupt

    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[order:model] HITL: Interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[order:model] Got the response from user %r", user_reply)
            return {"messages" : [response, HumanMessage(content=str(user_reply))]}
        
    return {"messages": [response]}

tool_node = ToolNode([get_order_status])

def increment_tool_count(state: AgentState):
    """Update the total tool call count from the latest assistant response. """
    last_message = state['messages'][-1]
    last_message_tool_calls = len(getattr(last_message, "tool_calls", []))
    state_tool_calls = state["tools_call_count"]
    logger.info("Tool calls count: %d", last_message_tool_calls + state_tool_calls)
    return {"tools_call_count": last_message_tool_calls + state_tool_calls}

def order_should_continue(state: AgentState) -> str:
    """Route after support model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def tools_guardrail(state: AgentState):
    """Guard against excessive tool usage and route the graph execution. """
    if state.get('tools_call_count') >= 5:
        logger.info("[Guardrail] Maximum tool call limit reached, Forcing exit")
        return "force_end"
    return order_should_continue(state)


sb = StateGraph(AgentState)
sb.add_node("model", order_model)
sb.add_node("tools", tool_node)
sb.add_node("counter", increment_tool_count)


sb.add_edge(START, "model")
sb.add_conditional_edges("model", tools_guardrail, {
    "tools": "counter",
    END: END,
    "force_end": END,
    "model": "model"
})

sb.add_edge("counter", "tools")
sb.add_edge("tools", "model")

order_subgraph = sb.compile()

def order_agent(state: WorkerInput) -> Command[Literal['synthesizer']]:
    """Invoke the order subgraph for a user query and return a synthesizer command.
    """
    user_query = state["user_query"]
    context = build_context(state['messages'])
    task_desc = state['task_description']

    response = order_subgraph.invoke({
        "messages": [
            SystemMessage(content=order_prompt),
            HumanMessage(content=f"{context} Task: {task_desc} Customer Query: {user_query}")
        ],
        "tools_call_count": 0})

    answer = response['messages'][-1].content
    return Command(
        update= {"order_response": [answer]},
        goto="synthesizer"
    )


# import uuid
# thread_id = uuid.uuid4().hex 

# config = {"configurable": {"thread_id": thread_id}}
# memory = MemorySaver()

# order_subgraph2 = sb.compile(checkpointer=memory)

# TEMP_PROMPT = """What is the status of my order. You have the tool get_order_status to get order status"""
# result = order_subgraph2.invoke({"messages" : [TEMP_PROMPT], "tools_call_count": 0}, config)
# print(result['__interrupt__'])
# while "__interrupt__" in result :
#     question = result["__interrupt__"][0].value
#     user_resp = input(f"{question} :")
#     result = order_subgraph2.invoke(Command(resume=user_resp), config)

# logger.info(result['messages'][-1].content)


# TEMP_PROMPT = """What is the status of my order. You have the tool get_order_status to get order status"""
# result = order_agent({"messages" : [], "user_query": TEMP_PROMPT, "task_description": TEMP_PROMPT})


