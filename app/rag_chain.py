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

# Search 
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

    # Retrieve relevant chunks
    docs = db.similarity_search(query, k=4)
    
    # Remove duplicate chunks
    unique_docs = list(set([doc.page_content.strip() for doc in docs]))
    context = "\n\n".join(unique_docs[:2])

    print("\n=== CONTEXT SENT TO LLM ===\n")
    print(context)
    
    lines = context.split("\n")
 

    # SIMPLE RULE-BASED EXTRACTION


    if "skills" in query.lower():
        lines = context.split("\n")
        skills = []
        for line in lines:
            if any (keyword in line.lower() for keyword in [
                "python", "SQL", "machine Learnign", "scikit","xgboost", "pandas","numpy" 
                ]):       
                    skills.append(line.strip())
        if skills:
            return "\n".join(set(skills))[:300]


    elif "education" in query.lower():
       lines = context.split("\n")
       edu = [line.strip() for line in lines if "Engineering" in line or "Diploma" in line]
       return "\n".join(set(edu))[:300]
       if edu:
           return "\n".joint(set(edu))[:300]
    
    elif "libraries" in query.lower():
       lines = context.split("\n")
       libs = [line.strip() for line in lines if "Scikit" in line or "Tensor" in line or "NumPy" in line]
       return "\n".join(set(libs))[:300]
       if libs:
           return "\n".join(set(libs))[:300]

    # Create prompt
    prompt = f"""
        
    Context:
    {context}
    
    Question:
    {query}
    
    Final Answer:
    """     
                     
    # CALL OLLAMA
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
    )

    result = response.json()

    
    # SAFE PARSE
    
    answer = result.get("response", "").strip()

    
    # CLEAN OUTPUT
    
    bad_phrases = ["Sure", "Here", "I can", "assistant", "AI"]
    for phrase in bad_phrases:
        answer = answer.replace(phrase, "")

   