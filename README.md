## 🚀 RAGOps-AI: Production-Ready RAG System with MLOps 


### 📌 Project Overview
---
RAGOps-AI is an end-to-end Retrieval-Augmented Generation (RAG) system built using modern AI engineering and MLOps practices.

The project enables users to ask questions about a resume through a web interface while leveraging:

Semantic search with vector databases
LLM-powered response generation
Experiment tracking
Containerized deployment
CI/CD automation
Monitoring and observability
Load testing and system validation

This project was built following a structured 30-Day RAGOps roadmap to simulate real-world AI production workflows.


### 🎯 Problem Statement
---
Traditional resume search systems rely on keyword matching and fail to understand context.

This project solves that problem by:

* Converting resume content into vector embeddings
* Storing embeddings in ChromaDB
* Retrieving relevant context using semantic search
* Generating grounded answers using LLMs
* Tracking experiments and system performance

### 🏗️ Architecture Diagram
---
![Architecture Diagram](/docs/architecture.png.png "Project Architecture")

🛠 Tech Stack
---
**AI / LLM**
* LangChain
* Hugging Face Inference API
* DeepSeek V4
* Sentence Transformers

**Backend** 
* FastAPI
* Pydantic

**Vector Database**
* ChromaDB

**MLOps**
* MLflow

**Deployment**
* Docker
* Docker Compose
* Render

**CI/CD**
* GitHub Actions

**Testing**
* Pytest
* Locust