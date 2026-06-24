import logging
import os
import sys

from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

# ── Logger ───────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("config")

OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAPI_API_KEY:
    logger.error("OPENAPI_KEY is missing. ")
    sys.exit(1)

openai_client = OpenAI(api_key=OPENAPI_API_KEY)
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.2
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

logger.info("OpenAI clients initialised (model: gpt-4o, embedding: text-embedding-3-small)")
