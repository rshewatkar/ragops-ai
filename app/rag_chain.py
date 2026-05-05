import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from .*",
    module="transformers.*",
)
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ragops-ai")

import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load + chunk PDF
def load_and_chunk_pdf(file_path: str):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks

# creat Vector DB
def create_vector_store(file_path: str):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Store in ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Vector DB created successfully")

    return vector_db

# Hugging Face Setup (SECURE)

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

def call_llm(prompt: str):
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",          

            messages=[
                {
                    "role": "system",
                    "content": "You are a strict information extraction system. Only answer using given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("LLM Error:",e)
        return ""

def clean_output(answer: str):
    if not answer:
        return "Not found"

    # remove junk words
    bad_words = [
        "sure", "here", "answer", "assistant",
        "based on", "the context", ":"
    ]

    answer = answer.lower()

    for word in bad_words:
        answer = answer.replace(word, "")

    # remove extra spaces
    answer = " ".join(answer.split())

    # limit length
    return answer.strip()[:300] if answer else "Not found"


def unique_lines(lines):
    seen = set()
    result = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            result.append(line)
    return result


def extract_lines(context, keywords):
    lines = context.split("\n")
    matches = []

    for line in lines:
        if any(k in line.lower() for k in keywords):
            matches.append(line.strip())

    return unique_lines(matches)

def relevance_score(query, context):
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    overlap = query_words.intersection(context_words)
    return round(len(overlap) / (len(query_words) + 1), 3)


def answer_coverage(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = answer_words.intersection(context_words)
    return round(len(overlap) / (len(answer_words) + 1), 3)

# =========================
# 🚀 MAIN RAG
# =========================
def ask_rag(query: str, chat_history=None):
    if mlflow.active_run():
        return _ask_rag(query, chat_history)

    with mlflow.start_run():
        return _ask_rag(query, chat_history)
    
def _ask_rag(query: str, chat_history=None):
    # log input
    mlflow.log_param("query", query)
    mlflow.log_param("prompt_version", "v2_resume_assistant")
    mlflow.log_param("chunk_size", 200)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm_model", "deepseek-ai/DeepSeek-V4-Pro:novita")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    history_text = ""

    if chat_history:
        last_turns = chat_history[-4:]  # last 2 Q&A
        for role, msg in last_turns:
            history_text += f"{role}: {msg}\n"
    
    query_lower = query.lower()

    # QUERY 
    
    if any(k in query_lower for k in ["skill", "technology", "tools"]):
        retrieval_query = query + " resume skills programming languages ML tools"
        k = 6
        query_type = "skills"
    
    elif any(k in query_lower for k in ["education", "degree"]):
        retrieval_query = query + " education degree university"
        k = 4
        query_type = "education"
    
    elif any(k in query_lower for k in ["librar", "framework"]):
        retrieval_query = query + " machine learning libraries python"
        k = 5
        query_type = "libraries"
    
    elif any(k in query_lower for k in ["experience", "work", "internship"]):
        retrieval_query = query + " work experience company role internship"
        k = 6
        query_type = "experience"
    
    elif any(k in query_lower for k in ["project", "built", "developed"]):
        retrieval_query = query + " projects machine learning built developed"
        k = 6
        query_type = "projects"
    
    elif any(k in query_lower for k in ["profile", "about", "who is", "summary"]):
        retrieval_query = query + " resume summary profile about candidate"
        k = 5
        query_type = "profile"
    
    else:
        retrieval_query = query
        k = 4
        query_type = "general"
                                                                               
    # Log retrival config
    mlflow.log_param("retrieval_k", k)
    mlflow.log_param("query_type", query_type)
    

    # 🔍 RETRIEVE
    docs = db.similarity_search(retrieval_query, k=k)

    context = "\n\n".join(unique_lines([doc.page_content for doc in docs]))

    print("\n=== CONTEXT ===\n")
    print(context)
    
    # EVALUATION METRICS 

    mlflow.log_metric("num_chunks", len(docs))

    rel_score = relevance_score(query, context)
    mlflow.log_metric("context_relevance", rel_score)     
    
    
    # RULE-BASED FAST PATH
    
    if "skill" in query_lower:
        skills = extract_lines(context, [
            "python", "sql", "machine learning",
            "scikit", "xgboost", "tensorflow",
            "pandas", "numpy", "mlflow", "docker", "git"
        ])
        answer = "\n".join(skills) if skills else "Not found"
        
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer    

    if "education" in query_lower:
        edu = extract_lines(context, [
            "engineering", "diploma", "university", "bachelor"
        ])
        answer = "\n".join(edu) if edu else "Not found"
        
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer

    if "librar" in query_lower:
        libs = extract_lines(context, [
            "scikit", "xgboost", "tensorflow", "pandas", "numpy"
        ])
        answer =  "\n".join(libs) if libs else "Not found"
    
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer
    
    # PROFILE / SUMMARY
    if any(k in query_lower for k in ["profile", "about", "who is", "summary"]):
        lines = unique_lines(context.split("\n"))
        answer = "\n".join(lines[:5]) if lines else "Not found"
    
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer
    
    
    # EXPERIENCE
    if any(k in query_lower for k in ["experience", "work", "internship"]):
        exp = extract_lines(context, [
            "experience", "intern", "company", "worked", "role"
        ])
        answer = "\n".join(exp) if exp else "Not found"
    
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer
    
    
    # PROJECTS
    if "project" in query_lower:
        proj = extract_lines(context, [
            "project", "built", "developed", "model", "system"
        ])
        answer = "\n".join(proj) if proj else "Not found"
    
        mlflow.log_param("response_type", "rule-based")
        mlflow.log_metric("output_length", len(answer))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
        mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

        return answer
    
    # LLM FALLBACK
    
    prompt = f"""
    You are a helpful AI assistant answering questions about a candidate's resume.
    
    Conversation History:
    {history_text}
    
    Context:
    {context}
    
    User Question:
    {query}
    
    Rules:
    - Answer ONLY from context
    - Be concise but clear
    - If not found → say "Not found"
    - Do NOT hallucinate
    
    Answer:
    """  
                                   
    answer = call_llm(prompt)
    answer = clean_output(answer)
    
    # Log LL usage
    mlflow.log_param("response_type","llm")
    mlflow.log_metric("output_length", len(answer))
    mlflow.log_metric("answer_length", len(answer))
    mlflow.log_metric("is_found", 0 if answer == "Not found" else 1)
    mlflow.log_metric("answer_coverage", answer_coverage(answer, context))

    if not answer or "error" in answer.lower():
        return "Not Found"
    
    return answer.strip()
    
