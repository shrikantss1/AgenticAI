
import os
import json
from dotenv import load_dotenv
import numpy as np
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

print("\n" + "#" * 50 + "\n")
print("Sample ticket:")
print(json.dumps(tickets[0], indent=2))
print("\n" + "#" * 50 + "\n")

print("\n" + "#" * 50 + "\n")
print("Preparing data for embedding...")
text_tickets = [f"{ticket['title']}: {ticket['description']}" 
                for ticket in tickets]
print(f"Prepared {len(text_tickets)} text entries for embedding.")
print("\n" + "#" * 50 + "\n")   

print("\n" + "#" * 50 + "\n")
print("Creating embeddings...")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
embedding_dimensions = 1536  # For text-embedding-3-small
response = client.embeddings.create(
    input=text_tickets,
    model=embedding_model
)

print(f"Received response with {len(response.data)} embeddings.")
embeddings = [np.array(data.embedding) for data in response.data]
print(f"Created {len(embeddings)} embeddings with {embedding_dimensions} dimensions each.")
print("\n" + "#" * 50 + "\n")
print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])
print("  (These 1536 numbers encode the semantic meaning of the text)")