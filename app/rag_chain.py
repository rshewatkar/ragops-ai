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
#def search_query(query: str):
#    embeddings = HuggingFaceEmbeddings(
#        model_name="all-MiniLM-L6-v2"
#    )
#
#    db = Chroma(
#        persist_directory="db",
#        embedding_function=embeddings
#    )
#
#    results = db.similarity_search(query +"resume skills", k=4)
#
#    for i, doc in enumerate(results):
#        print(f"\n--- Result {i+1} ---\n")
#        print(doc.page_content)
        

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


# =========================
# 🚀 MAIN RAG
# =========================
def ask_rag(query: str):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    query_lower = query.lower()

    # 🔥 QUERY IMPROVEMENT
    if "skill" in query_lower:
        retrieval_query = query + " skills programming languages ML tools"
        k = 6
    elif "education" in query_lower:
        retrieval_query = query + " education degree university"
        k = 4
    elif "librar" in query_lower:
        retrieval_query = query + " machine learning libraries python"
        k = 5
    else:
        retrieval_query = query
        k = 4

    # 🔍 RETRIEVE
    docs = db.similarity_search(retrieval_query, k=k)

    context = "\n\n".join(unique_lines([doc.page_content for doc in docs]))

    print("\n=== CONTEXT ===\n")
    print(context)

    # =========================
    # ⚡ RULE-BASED FAST PATH
    # =========================
    if "skill" in query_lower:
        skills = extract_lines(context, [
            "python", "sql", "machine learning",
            "scikit", "xgboost", "tensorflow",
            "pandas", "numpy", "mlflow", "docker", "git"
        ])
        return "\n".join(skills) if skills else "Not found"

    if "education" in query_lower:
        edu = extract_lines(context, [
            "engineering", "diploma", "university", "bachelor"
        ])
        return "\n".join(edu) if edu else "Not found"

    if "librar" in query_lower:
        libs = extract_lines(context, [
            "scikit", "xgboost", "tensorflow", "pandas", "numpy"
        ])
        return "\n".join(libs) if libs else "Not found"

    
    # LLM FALLBACK
    
    prompt = f"""
    Extract answer strictly from context.

    Rules:
    - No explanation
    - No extra words
    - If not found → Not found

    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """

    answer = call_llm(prompt)

    if not answer or "error" in answer.lower():
        return "Not found"

    return answer.strip()




































































































