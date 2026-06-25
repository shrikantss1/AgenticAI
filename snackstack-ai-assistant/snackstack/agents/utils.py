
"""Utility helpers for agent conversation formatting.

This module provides convenience functions for converting prior
conversation messages into a single text block suitable for use
as agent context.
"""

from langchain.messages import AIMessage, AnyMessage, HumanMessage


# Build a plain-text representation of prior conversation turns for use
# as context when invoking the agent. Human messages are labeled as
# "Customer" and AI messages as "Assistant".
def build_context(messages: list[AnyMessage]) -> str:
    """Format prior conversation turns as text for agent context."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "CONVERSATION SO FAR:\n" + "\n".join(parts) + "\n\n"