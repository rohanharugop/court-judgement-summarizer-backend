import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load .env variables
load_dotenv()

# Read env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not PINECONE_INDEX:
    raise ValueError("❌ Pinecone env variables not set")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Load judgment chunks
with open("processed-data/judgment_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

BATCH_SIZE = 50
vectors = []

for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk["text"]).tolist()

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "case_name": chunk["metadata"]["case_name"],
            "text": chunk["text"]
        }
    })

    if len(vectors) == BATCH_SIZE:
        index.upsert(vectors=vectors)
        vectors = []

if vectors:
    index.upsert(vectors=vectors)

print("✅ All vectors uploaded to Pinecone successfully")
