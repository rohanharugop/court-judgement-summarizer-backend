import json
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")




# Load models ONCE
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

# Load vector store
index = faiss.read_index(str(BASE_DIR / "vector-store" / "faiss.index"))

with open(BASE_DIR / "vector-store" / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

with open(BASE_DIR / "processed-data" / "judgment_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]


import re





def retrieve_precedents(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "case_name": metadata[idx]["case_name"],
            "excerpt": texts[idx][:500]
        })

    return results


def build_prompt(query, precedents):
    context = "\n\n".join(
        f"Case: {p['case_name']}\nExcerpt: {p['excerpt']}"
        for p in precedents
    )

    return f"""
You are LexBrief AI, an intelligent legal research assistant.

Step 1 — Analyze the user's query:
- Determine the user's intent and sentiment (e.g., greeting, curiosity, confusion, academic interest).
- Do NOT explicitly mention sentiment labels in the final answer.

User Query:
"{query}"

Step 2 — Response strategy:
- If the query expresses confusion or uncertainty, respond in a calm, explanatory tone.
- If the query is purely informational, respond concisely and formally.
- If the query is a case name or precedent, explain its legal relevance.
- If the query is vague, politely guide the user toward a clearer legal question.

Step 3 — Legal reasoning:
Using the excerpts below, explain why each judgment is relevant to the query.
Focus on legal principles, constitutional interpretation, and precedent value.

Relevant court judgment excerpts:
{context}

Rules:
- Do NOT provide legal advice.
- Do NOT invent facts.
- Do NOT mention internal reasoning steps.
- Maintain a professional but approachable tone.
"""



def generate_explanation(query, precedents):
    prompt = build_prompt(query, precedents)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
