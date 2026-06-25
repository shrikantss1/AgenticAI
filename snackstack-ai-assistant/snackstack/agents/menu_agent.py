

"""Menu agent subgraph for building and executing menu-related queries.

This module defines the state and behavior for the menu-focused agent in
SnackStack. It binds the menu search tool to the language model, manages the
conversation and tool call state, enforces guardrails on tool usage, and
produces a command result that routes the response to the synthesizer stage.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.types import Command
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from snackstack.tools.menu_tools import search_menu_catalog
from snackstack.config import llm
from snackstack.logger import get_logger
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from snackstack.agents.prompts import menu_prompt

from snackstack.agents.utils import build_context

from snackstack.state import WorkerInput

logger = get_logger("menu_agent")

# ----------------------------------------------------
# MENU SUBGRAPH
# ----------------------------------------------------

llm_with_menu_tools = llm.bind_tools([search_menu_catalog])

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    tools_call_count: int


def menu_model(state: AgentState):
    """Invoke the menu-enabled language model on the current conversation state. """
    response = llm_with_menu_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode([search_menu_catalog])

def increment_tool_count(state: AgentState):
    """Update the total tool call count from the latest assistant response. """
    last_message = state['messages'][-1]
    last_message_tool_calls = len(getattr(last_message, "tool_calls", []))
    state_tool_calls = state["tools_call_count"]
    logger.info("Tool calls count: %d", last_message_tool_calls + state_tool_calls)
    return {"tools_call_count": last_message_tool_calls + state_tool_calls}


def tools_guardrail(state: AgentState):
    """Guard against excessive tool usage and route the graph execution. """
    if state.get('tools_call_count') >= 5:
        logger.info("[Guardrail] Maximum tool call limit reached, Forcing exit")
        return "force_end"
    return tools_condition(state)


sb = StateGraph(AgentState)
sb.add_node("model", menu_model)
sb.add_node("tools", tool_node)
sb.add_node("counter", increment_tool_count)


sb.add_edge(START, "model")
sb.add_conditional_edges("model", tools_guardrail, {
    "tools": "counter",
    END: END,
    "force_end": END
})

sb.add_edge("counter", "tools")
sb.add_edge("tools", "model")

menu_subgraph = sb.compile()

def menu_agent(state: WorkerInput) -> Command[Literal['synthesizer']]:
    """Invoke the menu subgraph for a user query and return a synthesizer command.

    Args:
        state: WorkerInput containing user query, task description, and previous messages.

    Returns:
        A Command directing the workflow to the synthesizer with the menu response.
    """
    user_query = state["user_query"]
    context = build_context(state['messages'])
    task_desc = state['task_description']

    response = menu_subgraph.invoke({
        "messages": [
            SystemMessage(content=menu_prompt),
            HumanMessage(content=f"{context} Task: {task_desc} Customer Query: {user_query}")
        ],
        "tools_call_count": 0})

    answer = response['messages'][-1].content
    return Command(
        update= {"menu_response": [answer]},
        goto="synthesizer"
    )




# TEMP_PROMPT = """What are the menu options for Veg. You have the tool search_menu_catalog to explore the menu options"""
# result = menu_subgraph.invoke({"messages" : [TEMP_PROMPT], "tools_call_count": 0})

# logger.info(result['messages'][-1].content)

# TEMP_PROMPT = """What are the menu options for Veg. You have the tool search_menu_catalog to explore the menu options"""
# result = menu_agent({"messages" : [], "user_query": TEMP_PROMPT, "task_description": TEMP_PROMPT})

# logger.info(result.update['menu_response'])
