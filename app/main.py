# app/main.py
import os
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from pydantic import BaseModel

from app.rag_chain import create_vector_store, get_vector_store_count, ask_rag


from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# PROMETHEUS METRICS

REQUEST_COUNT = Counter(
    "ragops_request_count",
    "Total number of API requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "ragops_request_latency_seconds",
    "Latency of API requests in seconds",
    ["endpoint"]
)

ASK_REQUEST_COUNT = Counter(
    "ragops_ask_requests_total",
    "Total number of /ask requests"
)

RAG_SUCCESS_COUNT = Counter(
    "ragops_success_responses_total",
    "Total successful RAG responses"
)

RAG_FAILURE_COUNT = Counter(
    "ragops_failed_responses_total",
    "Total failed RAG responses"
)

RAG_NOT_FOUND_COUNT = Counter(
    "ragops_not_found_responses_total",
    "Total 'Not found' responses"
)

RAG_PIPELINE_LATENCY = Histogram(
    "ragops_pipeline_latency_seconds",
    "Total RAG pipeline execution time"
)

VECTOR_DB_DOCUMENT_COUNT = Gauge(
    "ragops_vector_db_documents",
    "Total documents stored in ChromaDB"
)

ERROR_COUNT = Counter(
    "ragops_error_count",
    "Total number of API errors",
    ["endpoint"]
)

#  Auto-ingest PDF on startup 
RESUME_PATH = os.getenv("RESUME_PATH", "data/Rahul_Shewatkar_Resume.pdf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the container starts."""
    if os.path.exists(RESUME_PATH):
        count = get_vector_store_count()
        VECTOR_DB_DOCUMENT_COUNT.set(count)
        if count > 0:
            print(f"[Startup] ChromaDB ready with {count} existing documents.")
        else:
            print(f"[Startup] Auto-ingesting resume from {RESUME_PATH}...")
            create_vector_store(RESUME_PATH)   # builds ChromaDB from PDF
            count = get_vector_store_count()
            VECTOR_DB_DOCUMENT_COUNT.set(count)
            print("[Startup] Resume ingested. ChromaDB ready.")
    else:
        print(f"[Startup] WARNING: No resume found at {RESUME_PATH}")
        print("[Startup] Resume PDF not found.")
        
    yield   # app runs here

app = FastAPI(title="RAGOPS-AI API", lifespan=lifespan)

# PROMETHEUS MIDDLEWARE

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(process_time)

    return response

# Request/Response models 
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str

# Root Endpoint 
@app.get("/")
def root():
    return {
        "message": "RAGOPS-AI API is running 🚀",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }
    
# Endpoints 
@app.get("/health")
def health():
    db_path = "/app/db"
    has_data = os.path.exists(db_path) and len(os.listdir(db_path)) > 0
    return {
        "status": "ok",
        "chromadb_has_data": has_data,
        "resume_path": RESUME_PATH,
        "resume_exists": os.path.exists(RESUME_PATH)
    }

# =========================================================
# ASK ENDPOINT
# =========================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    ASK_REQUEST_COUNT.inc()

    start_time = time.time()

    try:
        answer = ask_rag(request.query)

        latency = time.time() - start_time
        RAG_PIPELINE_LATENCY.observe(latency)

        if answer.strip().lower() == "not found":
            RAG_NOT_FOUND_COUNT.inc()
        else:
            RAG_SUCCESS_COUNT.inc()

        return AskResponse(answer=answer)

    except Exception as e:

        ERROR_COUNT.labels(
            endpoint="/ask"
        ).inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
       
# PROMETHEUS METRICS ENDPOINT

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )












