from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import ask_rag

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "RAG API is running"}

# RAG endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = ask_rag(request.query)
    return {
        "query": request.query,
        "answer": answer
    }