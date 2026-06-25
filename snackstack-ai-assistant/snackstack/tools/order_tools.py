


from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from snackstack.tools.rag import menu_item_vector_store
from snackstack.logger import get_logger
from snackstack.config import llm
from snackstack.data import ORDER_DATABASE

logger = get_logger("order_tools")


class OrderSearchClassifier(BaseModel):
    search_criteria: Literal['orderid', 'trackingid', 'email', 'other'] = Field(
        description="This will classify the criteria to search order database."
        )
    search_key: str = Field(description="The search key to search the order database.")


ORDER_SEARCH_CLASSIFIER_PROMPT = """
Classify the identifier into one of the following search_criteria values: orderid, trackingid, email, other.
- If the identifier matches ORD-{id}, classify as orderid and normalize search_key to ORD-{id}.
- If the identifier matches SS{id}TRK, classify as trackingid and normalize search_key to SS{id}TRK. Remove special characters between the string.
- If the identifier is an email address, classify as email and normalize search_key to the email.
- Otherwise classify as other and keep search_key as the original identifier.

During normalization:
- orderid form is ORD-{id}
- trackingid form is SS{id}TRK

Return only a JSON object with keys search_criteria and search_key.
"""

order_search_classifier_llm = llm.with_structured_output(OrderSearchClassifier)

@tool
def get_order_status(identifier: str) -> dict:
    ''' The tool returns the order status of given identifier'''
    if not identifier:
        return "Invalid search identifier"
    
    result = order_search_classifier_llm.invoke([
        SystemMessage(content=ORDER_SEARCH_CLASSIFIER_PROMPT),
        HumanMessage(content=identifier)
        ])
    
    search_criteria = result.search_criteria
    search_key = result.search_key
    if search_criteria == 'orderid':
        order = ORDER_DATABASE.get(search_key)
        if not order:
            return {"error": f"No order found with id {search_key}"}
        return order
    elif search_criteria == 'trackingid':
        for order_id, details in ORDER_DATABASE.items():
            if details.get("tracking") == search_key:
                return details
        return {"error": f"No order found with id {search_key}"}
    
    elif search_criteria == 'email':
        for order_id, details in ORDER_DATABASE.items():
            if details.get("email") == search_key:
                return details
        return {"error": f"No order found with id {search_key}"}
    else:
        return {"error": f"No order found with id {search_key}"}



# logger.info(search_menu_catalog.invoke("Veg dishes"))

# logger.info(get_order_status.invoke("priya@example.com"))