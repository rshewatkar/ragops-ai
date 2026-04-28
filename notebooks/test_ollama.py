import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "tinyllama",   
        "prompt": "Explain AI simply",
        "stream": False,
        "options": {
            "temperature": 0}
    }
)

print(response.json()["response"])
