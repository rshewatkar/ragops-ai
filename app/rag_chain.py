import os
import re
import threading
import warnings

# =========================
# ENV + WARNINGS
# =========================

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Suppress GitPython warning — explicit path set in Dockerfile
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault(
    "GIT_PYTHON_GIT_EXECUTABLE", "/usr/bin/git"
)

warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from .*",
    module="transformers.*",
)

# =========================
# MLFLOW
# =========================

import mlflow

# Read from env so local dev can override without rebuilding Docker
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_EXPERIMENT  = os.getenv("MLFLOW_EXPERIMENT_NAME", "ragops-ai")

mlflow.set_tracking_uri(_MLFLOW_URI)

try:
    mlflow.set_experiment(_EXPERIMENT)
except Exception as e:
    # MLflow unreachable at import time — warn but don't crash
    print(f"[MLflow] Could not set experiment '{_EXPERIMENT}': {e}")

# =========================
# LANGCHAIN IMPORTS
# =========================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# OPENAI CLIENT (HuggingFace router)
# =========================

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

# Verify token present at startup — fail fast with a clear message
if not os.getenv("HF_TOKEN"):
    print(
        "[WARNING] HF_TOKEN env var not set. "
        "LLM calls will fail with 401 Unauthorized."
    )

# =========================
# CONSTANTS
# =========================

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL        = "deepseek-ai/DeepSeek-V4-Pro:novita"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
COLLECTION_NAME  = "ragops-resume"

# =========================
# GLOBAL SINGLETONS
# =========================

_VECTOR_STORE = None
_EMBEDDINGS   = None
_LOCK         = threading.RLock()

# =========================
# HELPERS — singleton loaders
# =========================

def get_db_path() -> str:
    return os.getenv(
        "CHROMA_DB_DIR",
        "/app/db" if os.path.exists("/app/db") else os.path.abspath("db"),
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance.
    Thread-safe double-checked locking.
    Called at startup (lifespan) so first /ask request
    never pays the 30–90s cold-load cost.
    """
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        with _LOCK:
            if _EMBEDDINGS is None:
                print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
                _EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL
                )
                print("[Embeddings] Model loaded.")

    return _EMBEDDINGS


def get_vector_store() -> Chroma:
    """
    Returns a cached Chroma vector store instance.
    Thread-safe double-checked locking.
    Called at startup (lifespan) so first /ask request
    never pays the ChromaDB connect cost.
    """
    global _VECTOR_STORE

    if _VECTOR_STORE is None:
        with _LOCK:
            if _VECTOR_STORE is None:
                print(f"[ChromaDB] Connecting to: {get_db_path()}")
                _VECTOR_STORE = Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=get_db_path(),
                    embedding_function=get_embeddings(),
                )
                count = _VECTOR_STORE._collection.count()
                print(f"[ChromaDB] Connected. {count} documents in collection.")

    return _VECTOR_STORE


def get_vector_store_count() -> int:
    """
    Returns the number of documents in the vector store.
    Used by lifespan to check if ChromaDB needs initialization.
    """
    try:
        db = get_vector_store()
        return db._collection.count()
    except Exception as e:
        print(f"[ChromaDB] Could not get document count: {e}")
        return 0


# =========================
# PDF CHUNKING
# =========================

def load_and_chunk_pdf(file_path: str) -> list:
    loader  = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"[PDF] Loaded {len(documents)} pages → {len(chunks)} chunks")
    return chunks


# =========================
# CREATE VECTOR STORE
# =========================

def create_vector_store(file_path: str) -> Chroma:
    """
    Ingests a PDF into ChromaDB.
    Call this once during setup — not on every request.
    """
    global _VECTOR_STORE

    chunks = load_and_chunk_pdf(file_path)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=get_db_path(),
        collection_name=COLLECTION_NAME,
    )

    # Update singleton so subsequent calls use the newly created store
    with _LOCK:
        _VECTOR_STORE = vector_db

    print("[ChromaDB] Vector store created and cached.")
    return vector_db


# =========================
# LLM
# =========================

def call_llm(prompt: str) -> str:
    """
    Calls the HuggingFace-routed LLM.
    Returns the answer string.
    Raises on failure so the caller can decide how to handle it.
    """
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI Resume Assistant.\n\n"
                    "Rules:\n"
                    "- Answer professionally\n"
                    "- Use bullet points when helpful\n"
                    "- Keep answers concise\n"
                    "- Answer ONLY from the provided context\n"
                    "- Do not hallucinate or invent information\n"
                    "- If the answer is not in the context, say exactly: Not found"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=400,
    )

    return completion.choices[0].message.content.strip()


# =========================
# UTILS
# =========================

def clean_output(answer: str) -> str:
    if not answer:
        return "Not found"
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()


def unique_lines(lines: list) -> list:
    seen, result = set(), []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            result.append(line)
    return result


def normalize_resume_text(text: str) -> str:
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fallback_answer(query: str, context: str) -> str:
    """
    Returns a concise extractive answer when the hosted LLM is unavailable.
    This keeps the API usable even if the external inference token is missing
    or lacks provider permissions.
    """
    q = query.lower()
    normalized_context = normalize_resume_text(context)
    lines = [
        normalize_resume_text(line)
        for line in unique_lines(re.split(r"[\n\r]+", context))
    ]

    if any(term in q for term in ("library", "libraries", "ml librar")):
        known_libraries = [
            ("Scikit-learn", r"\bscikit\s*-?\s*learn\b"),
            ("XGBoost", r"\bxgboost\b"),
            ("H2O AutoML", r"\bh2o\s+automl\b"),
            ("TensorFlow", r"\btensorflow\b"),
            ("Pandas", r"\bpandas\b"),
            ("NumPy", r"\bnumpy\b"),
            ("Matplotlib", r"\bmatplotlib\b"),
            ("Seaborn", r"\bseaborn\b"),
        ]
        found = [
            name for name, pattern in known_libraries
            if re.search(pattern, normalized_context, flags=re.IGNORECASE)
        ]
        return "ML libraries: " + ", ".join(found) if found else "Not found"

    if "skill" in q:
        skill_lines = [
            line for line in lines
            if re.match(
                r"^(languages|ml libraries|deployment|tools|familiar with):",
                line,
                flags=re.IGNORECASE,
            )
        ]
        if skill_lines:
            return "\n".join(f"- {line}" for line in skill_lines)

    keyword_map = {
        "skill": ("skill", "technical", "technology", "tool", "language",
                  "framework", "library", "python", "sql"),
        "experience": ("experience", "intern", "work", "role", "company"),
        "project": ("project", "built", "deployed", "model", "application"),
        "education": ("education", "degree", "university", "college"),
        "contact": ("email", "phone", "linkedin", "github", "contact"),
    }

    keywords = ()
    for intent, intent_keywords in keyword_map.items():
        if intent in q or any(word in q for word in intent_keywords):
            keywords = intent_keywords
            break

    if keywords:
        matches = [
            line for line in lines
            if any(keyword in line.lower() for keyword in keywords)
        ]
    else:
        matches = lines

    snippets = matches[:6] if matches else lines[:4]
    if not snippets:
        return "Not found"

    return "\n".join(f"- {snippet}" for snippet in snippets)


def relevance_score(query: str, context: str) -> float:
    stopwords = {"what", "is", "are", "his", "her", "the", "a", "an", "of", "to"}
    query   = re.sub(r"[^\w\s]", "", query.lower())
    context = re.sub(r"[^\w\s]", "", context.lower())
    query_words   = set(query.split()) - stopwords
    context_words = set(context.split())
    if not query_words:
        return 0.0
    overlap = query_words.intersection(context_words)
    return round(len(overlap) / len(query_words), 3)


# =========================
# QUERY ROUTING
# Returns (retrieval_query, k) tuned per intent
# =========================

def route_query(query: str) -> tuple[str, int]:
    q = query.lower()

    if any(w in q for w in (
        "skill", "technology", "tool", "language", "framework",
        "library", "libraries", "ml"
    )):
        return query + " skills ML Libraries technologies tools languages", 6

    if any(w in q for w in ("experience", "work", "job", "intern", "role", "company")):
        return query + " experience internship work history", 6

    if any(w in q for w in ("project", "built", "deployed", "model", "app")):
        return query + " projects machine learning deployed", 6

    if any(w in q for w in ("education", "degree", "university", "college", "study")):
        return query + " education degree university qualification", 4

    if any(w in q for w in ("contact", "email", "phone", "linkedin", "github")):
        return query + " contact email phone number", 3

    return query, 4


# =========================
# MLflow BACKGROUND LOGGER
# Runs in a daemon thread — never blocks /ask response
# =========================

def _log_to_mlflow(params: dict, metrics: dict):
    try:
        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
    except Exception as e:
        print(f"[MLflow] Logging skipped (non-fatal): {e}")


def log_mlflow_async(params: dict, metrics: dict):
    t = threading.Thread(target=_log_to_mlflow, args=(params, metrics))
    t.daemon = True
    t.start()


# =========================
# MAIN RAG PIPELINE
# =========================

def ask_rag(query: str, chat_history: list = None) -> str:
    """
    Full RAG pipeline:
      1. Route query → choose retrieval strategy
      2. Vector similarity search
      3. Build prompt with context + history
      4. Call LLM
      5. Log to MLflow async (non-blocking)
      6. Return clean answer

    Raises exceptions on hard failures (LLM down, DB missing).
    Callers (main.py) handle HTTP error responses.
    """

    # --- Query routing ---
    retrieval_query, k = route_query(query)

    # --- Vector search ---
    db   = get_vector_store()
    docs = db.similarity_search(retrieval_query, k=k)

    if not docs:
        # Collection empty or query totally off-topic
        log_mlflow_async(
            params={"query": query, "retrieval_k": k},
            metrics={"num_chunks": 0, "context_relevance": 0.0,
                     "answer_length": 0, "is_found": 0},
        )
        return "Not found"

    context = "\n\n".join(unique_lines([doc.page_content for doc in docs]))

    # --- Relevance score (local, fast) ---
    rel_score = relevance_score(query, context)

    # --- Conversation history (last 4 turns) ---
    history_text = ""
    if chat_history:
        for role, msg in chat_history[-4:]:
            history_text += f"{role}: {msg}\n"

    # --- Prompt ---
    prompt = f"""Conversation History:
{history_text}

Context (from resume):
{context}

Question:
{query}

Provide a professional, concise answer based only on the context above.
"""

    # --- LLM call with local extractive fallback ---
    try:
        raw_answer = call_llm(prompt)
        answer     = clean_output(raw_answer)
    except Exception as e:
        print(f"[LLM] Provider call failed; using extractive fallback: {e}")
        answer = fallback_answer(query, context)

    # --- MLflow async logging (never blocks response) ---
    log_mlflow_async(
        params={
            "query":           query,
            "retrieval_query": retrieval_query,
            "retrieval_k":     k,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model":       LLM_MODEL,
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
        },
        metrics={
            "num_chunks":        len(docs),
            "context_relevance": rel_score,
            "answer_length":     len(answer),
            "is_found":          0 if answer == "Not found" else 1,
        },
    )

    return answer
