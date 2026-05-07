from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_chain import ask_rag
import mlflow
import time


app = FastAPI(
    title="RAGOPS AI",
    description = "Resume RAG Assistaant with MLflow Tracking",
    version="1.0"
    )

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
    
    start_time = time.time()
    
    #start mlfloe run
    with mlflow.start_run():
        
        #Lo api metadata
        mlflow.log_param("api_endpoint","/Ask")
        mlflow.log_param("user_query",request.query)
        
        # get RAG repsonse
        answer = ask_rag(request.query)
        
        #Track latency
        latency = round(time.time()  - start_time, 3)
        
        mlflow.log_metrci("response_time_sec", latency)
        
        return {
        "query": request.query,
        "answer": answer,
        "response_time_sec":latency
    }
