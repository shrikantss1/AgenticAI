
import os
import json
from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

# Initialize the OpenAI client
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# LOAD DATA: Load the data you want to create embeddings for. 

print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} tickets.")



