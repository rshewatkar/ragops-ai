from locust import HttpUser, task, between

class RagOpsUser(HttpUser):

    wait_time = between(1, 3)

    @task
    def ask_question(self):

        payload = {
            "query": "What are Rahul's machine learning skills?",
            "query":"What are his skills?",
            "query":"What is his experience?",
            "query":"What projects has he built?",
            "query":"Tell me about his profile",
            "query":"Which ML libraries does he know?"
        }

        self.client.post(
            "/ask",
            json=payload
        )