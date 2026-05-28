import os
import re
import threading
import warnings

# ENV + WARNINGS

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

os.environ.setdefault(
    "GIT_PYTHON_GIT_EXECUTABLE",
    "/usr/bin/git"
)

warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from .*",
    module="transformers.*",
)

# MLFLOW

import mlflow

_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "ragops-ai"
)

mlflow.set_tracking_uri(_MLFLOW_URI)

try:
    experiment = mlflow.get_experiment_by_name(_EXPERIMENT)

    if experiment is None:
        mlflow.create_experiment(_EXPERIMENT)
        print(f"[MLflow] Created experiment: {_EXPERIMENT}")

    mlflow.set_experiment(_EXPERIMENT)

except Exception as e:
    print(f"[MLflow] Could not set experiment '{_EXPERIMENT}': {e}")


# LANGCHAIN IMPORTS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# OPENAI CLIENT

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

if not os.getenv("HF_TOKEN"):
    print(
        "[WARNING] HF_TOKEN env var not set. "
        "LLM calls may fail."
    )


# CONSTANTS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "deepseek-ai/DeepSeek-V4-Pro:novita"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

COLLECTION_NAME = "ragops-resume"


# GLOBAL SINGLETONS

_VECTOR_STORE = None
_EMBEDDINGS = None
_LOCK = threading.RLock()

# HELPERS

def get_db_path() -> str:

    return os.getenv(
        "CHROMA_DB_DIR",
        "/app/db" if os.path.exists("/app/db")
        else os.path.abspath("db"),
    )


def get_embeddings() -> HuggingFaceEmbeddings:

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

                
                print(
                    f"[ChromaDB] Connected. "
                    f"{count} documents in collection."
                )

    return _VECTOR_STORE


def get_vector_store_count() -> int:

    try:
        db = get_vector_store()

        count = db._collection.count()

        
        return count

    except Exception as e:

        print(f"[ChromaDB] Could not get document count: {e}")

        return 0

# PDF CHUNKING

def load_and_chunk_pdf(file_path: str) -> list:

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[PDF] Loaded {len(documents)} pages "
        f"→ {len(chunks)} chunks"
    )

    return chunks

# CREATE VECTOR STORE

def create_vector_store(file_path: str) -> Chroma:

    global _VECTOR_STORE

    chunks = load_and_chunk_pdf(file_path)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=get_db_path(),
        collection_name=COLLECTION_NAME,
    )

    with _LOCK:
        _VECTOR_STORE = vector_db

    count = vector_db._collection.count()

    
    print(
        f"[ChromaDB] Vector store created "
        f"with {count} documents."
    )

    return vector_db

# LLM

def call_llm(prompt: str) -> str:

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
                    "- Do not hallucinate\n"
                    "- If answer missing say: Not found"
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

# UTILS

def clean_output(answer: str) -> str:

    if not answer:
        return "Not found"

    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()


def unique_lines(lines: list) -> list:

    seen = set()

    result = []

    for line in lines:

        line = line.strip()

        if line and line not in seen:

            seen.add(line)

            result.append(line)

    return result


def relevance_score(query: str, context: str) -> float:

    stopwords = {
        "what", "is", "are", "his", "her",
        "the", "a", "an", "of", "to"
    }

    query = re.sub(r"[^\w\s]", "", query.lower())

    context = re.sub(r"[^\w\s]", "", context.lower())

    query_words = set(query.split()) - stopwords

    context_words = set(context.split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(context_words)

    return round(len(overlap) / len(query_words), 3)

# QUERY ROUTING

def route_query(query: str) -> tuple[str, int]:

    q = query.lower()

    if any(w in q for w in (
        "skill",
        "technology",
        "tool",
        "language",
        "framework",
        "library",
        "ml"
    )):
        return (
            query +
            " skills ML Libraries technologies tools languages",
            6
        )

    if any(w in q for w in (
        "experience",
        "work",
        "job",
        "intern",
        "role",
        "company"
    )):
        return query + " experience internship work history", 6

    if any(w in q for w in (
        "project",
        "built",
        "deployed",
        "model",
        "app"
    )):
        return query + " projects machine learning deployed", 6

    return query, 4

# FALLBACK ANSWER

def fallback_answer(query: str, context: str) -> str:

    lines = unique_lines(
        re.split(r"[\n\r]+", context)
    )

    snippets = lines[:5]

    if not snippets:
        return "Not found"

    return "\n".join(
        f"- {snippet}" for snippet in snippets
    )

# MLFLOW LOGGER

def _log_to_mlflow(params: dict, metrics: dict):

    try:

        with mlflow.start_run():

            for k, v in params.items():
                mlflow.log_param(k, v)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

    except Exception as e:

        print(f"[MLflow] Logging skipped: {e}")


def log_mlflow_async(params: dict, metrics: dict):

    t = threading.Thread(
        target=_log_to_mlflow,
        args=(params, metrics)
    )

    t.daemon = True

    t.start()

# MAIN RAG PIPELINE

def ask_rag(query: str, chat_history: list = None) -> str:

    retrieval_query, k = route_query(query)

    db = get_vector_store()

    docs = db.similarity_search(
        retrieval_query,
        k=k
    )

    if not docs:

        log_mlflow_async(
            params={
                "query": query,
                "retrieval_k": k
            },
            metrics={
                "num_chunks": 0,
                "context_relevance": 0.0,
                "answer_length": 0,
                "is_found": 0,
            },
        )

        return "Not found"

    context = "\n\n".join(
        unique_lines(
            [doc.page_content for doc in docs]
        )
    )

    rel_score = relevance_score(query, context)

    prompt = f"""
Context:
{context}

Question:
{query}

Provide a concise professional answer.
"""

    try:

        raw_answer = call_llm(prompt)

        answer = clean_output(raw_answer)

    except Exception as e:

        print(
            f"[LLM] Provider failed; using fallback: {e}"
        )

        answer = fallback_answer(query, context)

    log_mlflow_async(
        params={
            "query": query,
            "retrieval_query": retrieval_query,
            "retrieval_k": k,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
        },
        metrics={
            "num_chunks": len(docs),
            "context_relevance": rel_score,
            "answer_length": len(answer),
            "is_found": 0 if answer == "Not found" else 1,
        },
    )

    return answer