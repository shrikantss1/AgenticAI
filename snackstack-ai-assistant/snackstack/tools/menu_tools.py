


from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from snackstack.tools.rag import menu_item_vector_store
from snackstack.logger import get_logger
from snackstack.config import llm
from snackstack.data import ORDER_DATABASE

logger = get_logger("manu_tools")


@tool
def search_menu_catalog(query: str) -> str:
    """ The tool searches for menu catalog for given query """
    if not query or not query.strip():
        logger.error("The query is empty, please provide valid query.")
        return "The query is empty, please provide valid query."
    

    docs = menu_item_vector_store.similarity_search(query)
    
    if not docs:
        logger.error("No valid menu items found for the query %s", query)
        return "No valid menu items found for this query, please rephrase."
    
    result = "Found following menu items \n\n"

    for i, doc in enumerate(docs, 1):
        result += f"\nMenu Item{i}: \n{doc.page_content}"
    
    return result




# logger.info(search_menu_catalog.invoke("Veg dishes"))

# logger.info(get_order_status.invoke("priya@example.com"))