from locust import HttpUser, between, task
from random import randrange, random


class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def bloom_small(self):
        sentence = "Translate to chinese. EN: I like soup. CN: "
        self.client.post(
            "/generate",
            json={
                "inputs": sentence[: randrange(1, len(sentence))],
                "parameters": {"max_new_tokens": 20, "seed": random()},
            },
        )

    @task
    def bloom_small(self):
        sentence = "Translate to chinese. EN: I like soup. CN: "
        self.client.post(
            "/generate",
            json={
                "inputs": sentence[: randrange(1, len(sentence))],
                "parameters": {
                    "max_new_tokens": 20,
                    "do_sample": True,
                    "top_p": 0.9,
                    "seed": random(),
                },
            },
        )
