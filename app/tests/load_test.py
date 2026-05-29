from locust import HttpUser, task, between

class RagOpsUser(HttpUser):

    wait_time = between(1, 3)

    @task
    def ask_question(self):

        payload = {
            "query": "What are Rahul's machine learning skills?"
        }

        self.client.post(
            "/ask",
            json=payload
        )