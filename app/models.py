from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class Precedent(BaseModel):
    case_name: str
    excerpt: str

class RAGResponse(BaseModel):
    query: str
    precedents: List[Precedent]
    explanation: str
