"""
Configuration module for initializing OpenAI clients.

Loads OPENAI_API_KEY from environment variables and initializes:
- ChatOpenAI LLM client (gpt-4o model)
- OpenAI API client
- OpenAI embeddings client (text-embedding-3-small model)
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from snackstack.logger import get_logger

load_dotenv()

logger = get_logger("config")
# --------------------------------------------
# LOAD OPENAPI_KEY
# --------------------------------------------
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
if not OPENAI_API_KEY:
    logger.info("OPENAI_API_KEY is not found")
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

logger.info("OpenAI clients initialised  (model: gpt-4o, embeddings: text-embedding-3-small)")


