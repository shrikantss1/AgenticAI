


from typing import List, Literal

from snackstack.state import SnackState
from snackstack.agents.prompts import orchistration_prompt_fn
from snackstack.config import llm
from snackstack.logger import get_logger

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field
from langgraph.types import Command, Send

import operator

from snackstack.state import AgentTask


logger = get_logger("orchistrator")

class ClassificationResult(BaseModel):
    """Orchestrator's routing decision."""
    tasks: List[AgentTask] = Field(description="The list of tasks classified for the user query")
    reasoning: str = Field(description="Reasoning for the classification")
    requires_synthesis: bool = Field(
        description="True when multiple agents must have their results merged"
    )


orchistrator_llm = llm.with_structured_output(ClassificationResult)


def orchistrator_node(state: SnackState) -> Command[Literal['menu_agent', 'order_agent', 'synthesizer']] :
    user_query = state.get('user_query', "")
    prompt = orchistration_prompt_fn(user_query)
    classification = orchistrator_llm.invoke(prompt)

    targets = []
    for task in classification.tasks:
        logger.info("[orchistrator:node] task: %r", task)
        targets.append(Send( task.agent, {
                "messages": state.get('messages'),
                "user_query": user_query,
                "task_description": task.task_description
            }
        ))
    if not targets:
        targets = [Send("synthesizer", {})]

    return Command(
        update= {
            "route": classification.tasks,
            "requires_synthesis": classification.requires_synthesis,
            "menu_response": [],
            "order_response": []
        },
        goto=targets
    )
    
    

# orchistrator_node({"user_query": "What are the dishes available for veg and check my previous order"})
