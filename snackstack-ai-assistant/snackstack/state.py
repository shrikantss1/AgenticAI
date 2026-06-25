



from typing import Annotated, Literal, TypedDict
from langchain.messages import AnyMessage
import operator

from pydantic import BaseModel, Field

def agent_results_reducer(current: list[dict], update: list[dict]) -> list[dict]:
    """Like operator.add, but an empty list signals a reset."""
    if not update:
        return []
    return current + update

class AgentTask(BaseModel):
    """A single task assigned to a specialist agent."""
    agent: Literal['menu_agent', 'order_agent'] = Field(description="Which agent handles this task")
    task_description: str = Field(description="The desciption of the task which agent needs to handle")


class SnackState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    route: list[AgentTask]
    menu_response: Annotated[list[AnyMessage], agent_results_reducer]
    order_response: Annotated[list[AnyMessage], agent_results_reducer]
    final_answer: str
    requires_synthesis: bool


class WorkerInput(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    task_description: str
