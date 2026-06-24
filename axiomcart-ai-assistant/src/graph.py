"""
Build and compile the LangGraph StateGraph.

Graph topology:

  START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                        └─ support_agent ──↗

Each agent is internally a subgraph with a model ⇄ tools loop.
The MemorySaver checkpointer persists conversation history across
turns, enabling multi-turn HITL (agent asks a question on one turn,
user answers on the next).
"""


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from src.config import get_logger
from src.state import AxiomCartState
from src.nodes import orchistrator_node, product_agent, support_agent, synthesizer_node
logger = get_logger("graph")


def build_graph() -> StateGraph:

    builder = StateGraph(AxiomCartState)
    builder.add_node("orchestrator", orchistrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "orchestrator")

    builder.add_edge("synthesizer", END)

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory)

    logger.info("Graph compiled  (with MemorySaver for conversation persistence)")
    return graph


axiomcart_graph = build_graph()