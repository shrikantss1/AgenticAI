'''
RAG: Build a vector store form product catalog.
'''

import json

from src.data import PRODUCT_CATALOG
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import get_logger
from src.config import embeddings

# Read the product catalog


logger = get_logger("rag")


def _build_documents():
    documents = []

    for product in PRODUCT_CATALOG:
        content = (
            f"Product: {product['name']}\n"
            f"Brand: {product["brand"]}\n"
            f"Category: {product["category"]}\n"
            f"Rating: {product["rating"]}\n"
            f"Features: {",".join(product["features"])}\n"
            f"Description: {product["description"]}\n"
            f"In Stock: {product["in_stock"]}\n"
            f"Colors: {",".join(product["colors"])}\n"
        )

        document = Document(
            page_content=content,
            metadata = {
                "id": product["id"],
                "name": product["name"],
                "brand": product['brand'],
                "category": product['category'],
                "rating": product['rating'],
                "price": product['price']
            }
        )

        documents.append(document)
    return documents

def build_vector_store() -> Chroma:
    documents=_build_documents()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="axiomcart_products"
    )
    logger.info(f"Built the vector store for {len(documents)} products")
    return vector_store


product_vector_store = build_vector_store()





