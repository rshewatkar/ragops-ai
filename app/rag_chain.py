import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from .*",
    module="transformers.*",
)

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
            model="deepseek-ai/DeepSeek-V4-Pro:fastest",
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
        return f"Error: {str(e)}"



# Search 
def search_query(query: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    results = db.similarity_search(query +"resume skills", k=4)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---\n")
        print(doc.page_content)
        

def _unique_lines(lines):
    seen = set()
    unique = []
    for line in lines:
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _extract_matching_lines(context: str, keywords):
    matches = []
    for line in context.splitlines():
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in keywords):
            matches.append(line)
    return _unique_lines(matches)


def _first_context_lines(context: str, limit: int = 5):
    lines = _unique_lines(context.splitlines())
    return "\n".join(lines[:limit]) if lines else "Not found"


def _context_fallback(context: str):
    lines = _unique_lines(context.splitlines())
    useful_lines = [
        line for line in lines
        if len(line) > 20 and not line.lower().startswith(("page ", "http"))
    ]
    return "\n".join(useful_lines[:4]) if useful_lines else "Not found"


# Main RAG Function
def ask_rag(query: str):
    # Load embeddings + DB
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    query_lower = query.lower()

    retrieval_query = query
    if any(keyword in query_lower for keyword in ["skill", "librar", "technology", "tools"]):
        retrieval_query = (
            f"{query} technical skills programming languages machine learning "
            "ML libraries tools frameworks resume"
        )

    # Retrieve relevant chunks
    docs = db.similarity_search(retrieval_query, k=6)
    
    # Remove duplicate chunks
    unique_docs = _unique_lines([doc.page_content for doc in docs])
    context = "\n\n".join(unique_docs[:6])

    print("\n=== CONTEXT SENT TO LLM ===\n")
    print(context)
    
    # SIMPLE RULE-BASED EXTRACTION
    if any(keyword in query_lower for keyword in ["who is", "about rahul", "summary", "profile"]):
        return _first_context_lines(context)

    if "skill" in query_lower:
        skills = _extract_matching_lines(context, [
            "skills", "programming", "python", "sql", "machine learning",
            "scikit", "xgboost", "h2o", "tensorflow", "pandas", "numpy",
            "power bi", "tableau", "excel", "flask", "fastapi", "mlflow",
            "docker", "github", "git"
        ])
        if skills:
            return "\n".join(skills)[:500]
        else:
            return "Not found"

    elif "education" in query_lower:
        edu = _extract_matching_lines(context, [
            "engineering", "diploma", "university", "bachelor", "education"
        ])
        return "\n".join(set(edu)) if edu else "Not found"

    elif "librar" in query_lower:
        libs = _extract_matching_lines(context, [
            "ml libraries", "scikit", "xgboost", "h2o", "tensorflow", "pandas", "numpy"
        ])
        return "\n".join(set(libs)) if libs else "Not found" 

        
    # Create prompt
    prompt = f"""
    Extract answer from the context.

    Rules:
    - Only use context
    - No explanation
    - No extra words
    - If not found: Not found

    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """     
    answer = call_llm(prompt)

    # Cleanup
    answer = answer.replace("Answer:", "").strip()

    # Safety fallback
    if not answer or "error" in answer.lower():
        return _context_fallback(context)

    return answer                 

  
   
