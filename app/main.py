# app/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag_chain import create_vector_store, get_vector_store_count, ask_rag

# ── Auto-ingest PDF on startup ───────────────────────────────────────────────
RESUME_PATH = os.getenv("RESUME_PATH", "data/Rahul_Shewatkar_Resume.pdf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the container starts."""
    if os.path.exists(RESUME_PATH):
        count = get_vector_store_count()
        if count > 0:
            print(f"[Startup] ChromaDB ready with {count} existing documents.")
        else:
            print(f"[Startup] Auto-ingesting resume from {RESUME_PATH}...")
            create_vector_store(RESUME_PATH)   # builds ChromaDB from PDF
            print("[Startup] Resume ingested. ChromaDB ready.")
    else:
        print(f"[Startup] WARNING: No resume found at {RESUME_PATH}")
        print("[Startup] Call POST /upload to ingest manually.")
    yield   # app runs here

app = FastAPI(title="RAGOPS-AI API", lifespan=lifespan)

# ── Request/Response models ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str

# ── Endpoints ────────────────────────────────────────────────────────────────
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

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer = ask_rag(request.query)
    return AskResponse(answer=answer)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Manually upload a PDF to ingest."""
    import shutil
    os.makedirs("/app/data", exist_ok=True)
    save_path = f"/app/data/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    create_vector_store(save_path)
    return {"status": "ingested", "file": file.filename}
