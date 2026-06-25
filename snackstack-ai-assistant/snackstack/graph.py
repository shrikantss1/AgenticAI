


from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from snackstack.state import SnackState
from snackstack.agents.orchestrator import orchistrator_node
from snackstack.agents.menu_agent import menu_agent
from snackstack.agents.order_agent import order_agent
from snackstack.agents.synthesizer import synthesizer_node
from snackstack.logger import get_logger



logger = get_logger("graph")

def build_graph():


    builder = StateGraph(SnackState)
    builder.add_node("orchistrator", orchistrator_node)
    builder.add_node("menu_agent", menu_agent)
    builder.add_node("order_agent", order_agent)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "orchistrator")
    builder.add_edge("synthesizer", END)

    memory = MemorySaver()

    

    graph = builder.compile(checkpointer=memory)

    logger.info("Graph compiled with MemorySaver for conversation persitence")

    return graph


snackstack_graph = build_graph()

# TESTING
import uuid
thread_id = uuid.uuid4().hex
config = {"configurable": {"thread_id": thread_id}}

result = snackstack_graph.invoke({
    "user_query": "What is the status of my order"
}, config)

print(f"------- State: {snackstack_graph.get_state(config).interrupts}")
for task in snackstack_graph.get_state(config).tasks:
    if task.interrupts:
        print(f"------------- Interrupted Task----------------")
        for interrupt_info in task.interrupts:
            print(f"Interrupt Prompt: {interrupt_info.value}")
        

# logger.info(result['final_answer'])









