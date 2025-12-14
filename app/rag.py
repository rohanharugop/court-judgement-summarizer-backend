import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

# Load env vars
load_dotenv()

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Embedding model (small & fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Groq client
client = Groq(api_key=GROQ_API_KEY)


def retrieve_precedents(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(query).tolist()

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "case_name": match["metadata"]["case_name"],
            "excerpt": match["metadata"]["text"][:500]
        }
        for match in response["matches"]
    ]


def build_prompt(query, precedents):
    context = "\n\n".join(
        f"Case: {p['case_name']}\nExcerpt: {p['excerpt']}"
        for p in precedents
    )

    return f"""
You are LexBrief AI, an intelligent legal research assistant.

Step 1 — Analyze the user's intent and sentiment.
Adapt your tone accordingly.

Step 2 — Using the provided excerpts, explain the legal relevance
of each judgment to the user's query.

User Query:
"{query}"

Relevant judgment excerpts:
{context}

Rules:
- Do NOT give legal advice
- Do NOT hallucinate facts
- Ask for clarification if the query is vague
- Use professional legal language
"""


def generate_explanation(query, precedents):
    prompt = build_prompt(query, precedents)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        timeout=30
    )

    return response.choices[0].message.content
