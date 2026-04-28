import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

def search_query(query: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---\n")
        print(doc.page_content)
        

def ask_rag(query: str):
    # Load embeddings + DB
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    # Retrieve relevant chunks
    docs = db.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])
    print("\n=== CONTEXT SENT TO LLM ===\n")
    print(context)
    
    for i, doc in enumerate(docs):
        print(f"\n--- Retrieved Chunk {i+1} ---\n")
        print(doc.page_content)

    # Create prompt
    prompt = f"""
    You are a strict information extraction system.

    Extract ONLY the exact skills mentioned in the context.

    Rules:
    - Do NOT add any new skills
    - Do NOT assume anything
    - Do NOT include tools not explicitly written
    - If only Python and SQL are present, return only those
    - Answer in bullet points

    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """    
    
    # Call Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "option":{"temperature":0}
        }
    )

    return response.json()["response"]