from locust import HttpUser, task, between
import random
from locust.exception import RescheduleTask


similar_prompts = [
    ("Tell me about AI", "Explain artificial intelligence to me"),
    ("What is machine learning?", "Describe machine learning"),
    ("Define deep learning", "What does deep learning mean?")
]

different_prompts = [
    ("What's the weather like today?", "Explain quantum computing"),
    ("How do I cook pasta?", "Tell me about the stock market"),
    ("Describe a cat", "How does an airplane work?")
]

class PromptUser(HttpUser):
    wait_time = between(1, 5)  # simulate user wait time between requests

    @task(3)
    def send_similar_prompt(self):
        p1, p2 = random.choice(similar_prompts)
        payload = {
            "prompt1": p1,
            "prompt2": p2,
            "similarity_method": "cosine"
        }
        self.client.post("/check_prompt_similarity", json=payload)

    @task(1)
    def send_different_prompt(self):
        p1, p2 = random.choice(different_prompts)
        payload = {
            "prompt1": p1,
            "prompt2": p2,
            "similarity_method": "cosine"
        }
        self.client.post("/check_prompt_similarity", json=payload)

    @task
    def similar_prompt(self):
        try:
            payload = {
                "prompt1": "Tell me about AI",
                "prompt2": "Explain artificial intelligence to me",
                "similarity_method": "cosine"
            }
            with self.client.post("/check_prompt_similarity", json=payload, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Unexpected status code: {response.status_code}")
                elif response.elapsed.total_seconds() > 1.0:
                    response.failure("Response took too long")
                else:
                    response.success()
        except Exception as e:
            raise RescheduleTask(f"Request failed: {str(e)}")
