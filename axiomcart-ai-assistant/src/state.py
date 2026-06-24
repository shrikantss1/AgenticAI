"""
This class contains all the states used in langgraph workflow.
"""
from typing import Annotated, List, Literal, TypedDict
from langchain.messages import AnyMessage
from pydantic import BaseModel, Field
import operator

def agent_results_reducer(current: list[dict], update: list[dict]) -> list[dict]:
    """Like operator.add, but an empty list signals a reset."""
    if not update:
        return []
    return current + update

class AgentTask(BaseModel):
    """ A single task assigned to a specialist agent. """
    agent: Literal['product_agent', 'support_agent'] = Field(
        description="Which agent handles this task"
    )
    task_description: str = Field(
        description="What agent should do"
    )

class ClassificationResult(BaseModel):
    """ Orchistrator's Routing Decision """
    tasks: List[AgentTask] = Field(
        description="Tasks to dispatch"
    )
    requires_synthesis: bool = Field(
        description="True when multiple agents must have their results"
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )


class AxiomCartState(TypedDict):

    """Top-level state that flows through the entire graph."""

    # Converation
    user_query:str
    messages: Annotated[list[AnyMessage], operator.add]

    # Routing
    tasks: list[AgentTask]
    requires_synthesis: bool

    # Collected results from agents
    agent_results: Annotated[list[AnyMessage], agent_results_reducer]

    # Final response returned to user
    final_answer: str


class WorkerInput(TypedDict):
    """ Payload delivered to an agent worker node via Send() """
    
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    task_description: str
    

