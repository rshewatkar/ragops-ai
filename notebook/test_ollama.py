import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi",   # 👈 use this
        "prompt": "Explain RAG simply",
        "stream": False
    }
)

print(response.json()["response"])
