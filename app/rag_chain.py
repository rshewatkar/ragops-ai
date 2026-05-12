import os
import re
import threading
import warnings

# =========================
# ENV + WARNINGS
# =========================

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from .*",
    module="transformers.*",
)

# =========================
# MLFLOW
# =========================

import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("ragops-ai")

# =========================
# LANGCHAIN IMPORTS
# =========================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# OPENAI CLIENT
# =========================

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

# =========================
# GLOBAL SINGLETONS
# =========================

_VECTOR_STORE = None
_EMBEDDINGS = None

_LOCK = threading.Lock()

# =========================
# HELPERS
# =========================


def get_db_path():
    return os.getenv(
        "CHROMA_DB_DIR",
        "/app/db" if os.path.exists("/app/db") else os.path.abspath("db")
    )


def get_embeddings():
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        with _LOCK:
            if _EMBEDDINGS is None:

                print("Loading embedding model...")

                _EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

    return _EMBEDDINGS


def get_vector_store():
    global _VECTOR_STORE

    if _VECTOR_STORE is None:
        with _LOCK:
            if _VECTOR_STORE is None:

                print("Loading ChromaDB...")

                _VECTOR_STORE = Chroma(
                    persist_directory=get_db_path(),
                    embedding_function=get_embeddings()
                )

    return _VECTOR_STORE


# =========================
# PDF CHUNKING
# =========================

def load_and_chunk_pdf(file_path: str):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Loaded {len(documents)} pages")
    print(f"Created {len(chunks)} chunks")

    return chunks


# =========================
# CREATE VECTOR STORE
# =========================

def create_vector_store(file_path: str):

    chunks = load_and_chunk_pdf(file_path)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=get_db_path()
    )

    print("Vector DB created successfully")

    return vector_db


# =========================
# LLM
# =========================

def call_llm(prompt: str):

    try:

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",

            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI Resume Assistant.

Rules:
- Answer professionally
- Use bullet points when needed
- Keep answers concise
- Answer ONLY from context
- Do not hallucinate
- If answer not found say: Not found
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,
            max_tokens=300
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:

        print("LLM ERROR:", e)

        return "Not found"


# =========================
# CLEAN OUTPUT
# =========================

def clean_output(answer: str):

    if not answer:
        return "Not found"

    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()


# =========================
# UTILS
# =========================

def unique_lines(lines):

    seen = set()
    result = []

    for line in lines:

        line = line.strip()

        if line and line not in seen:
            seen.add(line)
            result.append(line)

    return result


def relevance_score(query, context):

    stopwords = {
        "what", "is", "are", "his",
        "her", "the", "a", "an",
        "of", "to"
    }

    query = re.sub(r"[^\w\s]", "", query.lower())
    context = re.sub(r"[^\w\s]", "", context.lower())

    query_words = set(query.split()) - stopwords
    context_words = set(context.split())

    overlap = query_words.intersection(context_words)

    if not query_words:
        return 0.0

    return round(len(overlap) / len(query_words), 3)


# =========================
# MAIN RAG
# =========================

def ask_rag(query: str, chat_history=None):

    if mlflow.active_run():
        return _ask_rag(query, chat_history)

    with mlflow.start_run():
        return _ask_rag(query, chat_history)


def _ask_rag(query: str, chat_history=None):

    query_lower = query.lower()

    # =========================
    # MLFLOW LOGGING
    # =========================

    mlflow.log_param("query", query)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm_model", "deepseek-ai/DeepSeek-V4-Pro:novita")
    mlflow.log_param("chunk_size", 1000)
    mlflow.log_param("chunk_overlap", 200)

    # =========================
    # QUERY ROUTING
    # =========================

    k = 4

    if "skill" in query_lower:
        retrieval_query = query + " skills technologies tools"
        k = 6

    elif "experience" in query_lower:
        retrieval_query = query + " experience internship work"
        k = 6

    elif "project" in query_lower:
        retrieval_query = query + " projects machine learning"
        k = 6

    elif "education" in query_lower:
        retrieval_query = query + " education degree university"
        k = 4

    else:
        retrieval_query = query

    mlflow.log_param("retrieval_k", k)

    # =========================
    # VECTOR SEARCH
    # =========================

    db = get_vector_store()

    docs = db.similarity_search(retrieval_query, k=k)

    context = "\n\n".join(
        unique_lines([doc.page_content for doc in docs])
    )

    # =========================
    # METRICS
    # =========================

    mlflow.log_metric("num_chunks", len(docs))

    rel_score = relevance_score(query, context)

    mlflow.log_metric("context_relevance", rel_score)

    # =========================
    # PROMPT
    # =========================

    history_text = ""

    if chat_history:

        last_turns = chat_history[-4:]

        for role, msg in last_turns:
            history_text += f"{role}: {msg}\n"

    prompt = f"""
Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Provide a professional answer.
"""

    # =========================
    # LLM CALL
    # =========================

    answer = call_llm(prompt)

    answer = clean_output(answer)

    # =========================
    # FINAL LOGGING
    # =========================

    mlflow.log_metric("answer_length", len(answer))

    mlflow.log_metric(
        "is_found",
        0 if answer == "Not found" else 1
    )

    return answer