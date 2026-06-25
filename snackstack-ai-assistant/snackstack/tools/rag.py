
"""Builds a Chroma vector store from snackstack menu catalog data.

This module constructs Document objects for each menu item with relevant metadata
and initializes a Chroma collection using the configured embeddings.
"""

from langchain_chroma import Chroma
from snackstack.data import MENU_CATALOG
from snackstack.config import embeddings
from langchain_core.documents import Document
from snackstack.logger import get_logger

logger = get_logger("rag")


def build_vector_store():

    documents = []

    for item in MENU_CATALOG:
        content = (
        f"id: {item['id']}\n"
        f"dish: {item['dish']}\n"
        f"cuisine: {item['cuisine']}\n"
        f"price: {item['price']}\n"
        f"rating: {item['rating']}\n"
        f"dietary: {item['dietary']}\n"
        f"description: {item['description']}\n"
        )

        documents.append(Document(
            page_content=content,
            metadata = {
                "cuisine": item['cuisine'],
                "price": item['price'],
                "dietary": item['dietary']
            }
        ))
    
    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = embeddings,
        collection_name="menu_catalog"
    )

    return vector_store


menu_item_vector_store = build_vector_store()




