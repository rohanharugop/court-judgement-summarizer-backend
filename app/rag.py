import os
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq

load_dotenv()

# ENV
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Init clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
client = Groq(api_key=GROQ_API_KEY)

def retrieve_precedents(query: str, top_k: int = 5):
    response = index.query(
        text=query,
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
You are LexBrief AI, a legal research assistant.

Analyze the user's intent and sentiment.
Respond accordingly.

User Query:
"{query}"

Relevant Judgments:
{context}

Rules:
- Do NOT hallucinate
- Do NOT give legal advice
- Ask for clarification if vague
"""


def generate_explanation(query, precedents):
    prompt = build_prompt(query, precedents)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
