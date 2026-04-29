import requests

API_URL = "http://127.0.0.1:8000/ask"

queries = [
    "What are Rahul's skills?",
    "What is his education?",
    "Which ML libraries does he know?"
]

for i, query in enumerate(queries, 1):
    response = requests.post(API_URL, json={"query": query})

    print(f"\n--- Test Case {i} ---")
    print("Query:", query)

    try:
        result = response.json()
        print("Answer:", result["answer"])
    except Exception as e:
        print("Error:", e)