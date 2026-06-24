from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    generated_text: str


def review_node(state: ReviewState):
    # Ask a reviewer to edit the generated content
    updated = interrupt(
        {
            "instruction": "Review and edit this content",
            "content": state["generated_text"],
        }
    )
    return {"generated_text": updated}


builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "review-42"}}
initial = graph.stream_events(
    {"generated_text": "Initial draft"}, config=config, version="v3"
)

print(f"Initial: {initial.interrupted}")
print(f"Output: {initial.output}")
_ = initial.output  # drive the stream to completion
print(initial.interrupts)  # -> (Interrupt(value={'instruction': ..., 'content': ...}),)

ans = input(initial.interrupts[0].value['instruction'])
# Resume with the edited text from the reviewer
# final_state = graph.stream_events(
#     Command(resume="Improved draft after review"),
#     config=config,
#     version="v3",
# )


final_state = graph.stream_events(
    Command(resume=ans),
    config=config,
    version="v3",
)

print(final_state.output["generated_text"])  # -> "Improved draft after review"


# result = graph.invoke(
#      {"generated_text": "Initial draft"}, config=config, version="v3"
# )

# while "__interrupt__" in result and result["__interrupt__"]:
#     question = result["__interrupt__"][0].value['instruction']
#     user_ans = input(f"Agent Asks: {question}")
#     result = graph.invoke(Command(resume=user_ans), config)


