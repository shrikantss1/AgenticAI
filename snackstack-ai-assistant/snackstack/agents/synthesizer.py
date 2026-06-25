


"""Synthesizer agent node.

This module defines a node that combines specialist agent responses
into a single final answer for the customer.
"""

from snackstack.state import SnackState
from snackstack.config import llm


def synthesizer_node(state: SnackState) -> dict:
    """Generate the final response for the customer.

    Args:
        state (StackState): Current state with keys 'user_query',
            'menu_response', and 'order_response'.

    Returns:
        dict: A dictionary containing 'final_answer'.
    """
    user_query = state['user_query']
    menu_respose = state['menu_response']
    order_response = state['order_response']
    
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"MENU AGENT RESPONSES:\n{menu_respose}\n\n"
        f"ORDER AGENT RESPONSES:\n{order_response}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'SnackStack Assistant'."
    )

    merged = llm.invoke(prompt)
    return {"final_answer": merged.content}



# result = synthesizer_node({
#     "user_query": "Give me some veg options and where is my order ORD201",
#     "menu_response": "Veg Options are idli and dosa",
#     "order_response": "Order ORD201 is delayed"
# })
# print(result)


