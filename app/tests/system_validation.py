import requests

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")

    assert response.status_code == 200

    print("Health endpoint OK")


def test_ask():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "query": "What are Rahul's skills?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data

    print("Ask endpoint OK")


if __name__ == "__main__":
    test_health()
    test_ask()

    print("System validation passed")