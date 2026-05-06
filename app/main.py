from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_chain import ask_rag

app = FastAPI(title="RAG Resume Assistant API")

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
    try:
        answer = ask_rag(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "query": request.query,
        "answer": answer
    }
