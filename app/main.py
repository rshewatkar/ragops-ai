from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title= "RAGOps AI API")

# Request body schema
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAGOps AI is running 🚀"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": request.question,
            "stream": False
        }
    )

    result = response.json()["response"]

    return {
        "question": request.question,
        "answer": result
    }