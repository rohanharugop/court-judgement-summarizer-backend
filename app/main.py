from fastapi import FastAPI
from app.models import QueryRequest, RAGResponse, Precedent
from app.rag import retrieve_precedents, generate_explanation

app = FastAPI(
    title="Legal Judgment RAG API",
    description="Court Judgment Summarizer + Precedent Finder",
    version="1.0"
)


@app.get("/")
def health_check():
    return {"status": "Legal RAG API is running"}



@app.post("/rag", response_model=RAGResponse)
def rag_query(request: QueryRequest):

    if not request.query.strip():
        return {
            "query": request.query,
            "precedents": [],
            "explanation": (
                "Please enter a legal question, case name, or judgment excerpt "
                "so I can assist you."
            )
        }

    precedents = retrieve_precedents(request.query, request.top_k)
    explanation = generate_explanation(request.query, precedents)

    return {
        "query": request.query,
        "precedents": precedents,
        "explanation": explanation
    }
