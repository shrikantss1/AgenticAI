"""
All the tools for the AxionCart AI Assistant
"""

from src.data import ORDER_DATABASE, ESCALATION_QUEUE
from src.rag import product_vector_store
from langchain.tools import tool
from src.config import get_logger
import random
import time

logger = get_logger("tools")

def normalise_order_id(raw: str) -> str:
    """Accept 'ORD101', 'ORD-101', 'ord-101', or just '101' → 'ORD101'."""
    upper = raw.upper().strip()
    clean = upper.replace("ORD-", "").replace("ORD", "").strip()
    return f"ORD{clean}"

def lookup_order_by_email(email: str) -> dict | None:
    """Find the first order matching a customer email."""
    email_lower = email.lower().strip()

    for oid, order in ORDER_DATABASE.items():
        if order['customer_email'].lower() == email_lower:
            return {"order_id": oid, **order}
    return None

# ------------------------------------------------------------
# PRODUCT DISCOVERY TOOL
# ------------------------------------------------------------
@tool
def get_similar_products(query: str) -> str:
    """Search the AxiomCart product catalog using semantic search (RAG).

    Args:
        query: natural-language search, e.g. "wireless headphones under 5000"
    """
    docs = product_vector_store.similarity_search(query)
    if not docs:
        logger.error("No products are found matching the query")
    results = f"Found following products for the query:\n\n"
    for i, doc in enumerate(docs):
        results += f"Produt {i}: \n {doc.page_content}\n\n"
    return results


# ------------------------------------------------------------
# SUPPORT AGENT TOOL
# ------------------------------------------------------------

@tool
def get_order_status(identifier: str) -> str:
    
    """Look up the current status of a customer order.

    Args:
        identifier: an order ID (e.g. "ORD101") OR a customer email address
    """
    logger.info("get_order_status  identifier=%r", identifier)

    if "@" in identifier:
        match = lookup_order_by_email(identifier)
        if match:
            oid = match['order_id']
            order = {k: v for k, v in match.items() if k != "order_id"}
    else: 
        oid = normalise_order_id(identifier)
        order = ORDER_DATABASE.get(oid)
        if not order:
            return f"Order {oid} not found. Please verify the order ID"

    info = (
        f"Order {oid}:\n"
        f"  Customer : {order['customer_name']} ({order['customer_email']})\n"
        f"  Product  : {order['product']}\n"
        f"  Price    : ₹{order['price']:,}\n"
        f"  Status   : {order['status']}\n"
        f"  Ordered  : {order['order_date']}\n"
        f"  ETA      : {order['estimated_delivery']}"
    )
    if order.get("delay_reason"):
        info += f"\n  Delay    : {order['delay_reason']}"
    return info
    

@tool
def escalate_to_human(order_id: str, issue_summary: str, priority: str = "normal") -> str:

    """Escalate to a human support agent. Customer details are pulled
    from the order database. An email notification is sent if Resend
    is configured.

    Args:
        order_id:      the related order (e.g. "ORD101")
        issue_summary: brief description of the problem
        priority:      low | normal | high | urgent
    """

    order_id = normalise_order_id(order_id)
    logger.info(f"escalate_to_human order_id: {order_id}, priority: {priority}")
    order = ORDER_DATABASE.get(order_id)

    customer_name = order["customer_name"] if order else "Unkown"
    customer_email = order["customer_email"] if order else "Unknown"

    ticket_id = f"ESC-{random.randint(10000, 99999)}"

    ESCALATION_QUEUE.append(
        {"ticket_id": ticket_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name,
        "customer_email": customer_email,
        "order_id": order_id,
        "issue_summary": issue_summary,
        "priority": priority,
        "status": "open",}
    )

    response_times = {"urgent": "1 hour", "high": "4 hours", "normal": "24 hours", "low": "48 hours"}

    return (
        f"Escalation ticket created.\n"
        f"  Ticket   : {ticket_id}\n"
        f"  Priority : {priority.upper()}\n"
        f"  Customer : {customer_name} ({customer_email})\n"
        f"  ETA      : within {response_times.get(priority, '24 hours')}\n"
        f"A human agent will follow up shortly."
    )

