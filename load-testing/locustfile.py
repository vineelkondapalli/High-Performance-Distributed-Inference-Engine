"""
Locust load test for the C++ sidecar inference endpoint.

Usage:
  # Web UI (http://localhost:8089)
  locust -f load-testing/locustfile.py --host=http://localhost:8080

  # Headless — 50 users, 5 spawn/sec, run for 60 seconds
  locust -f load-testing/locustfile.py --host=http://localhost:8080 \
    --users 50 --spawn-rate 5 --run-time 60s --headless \
    --csv=load-testing/results/locust

  # Headless burst test — 200 users to stress the queue
  locust -f load-testing/locustfile.py --host=http://localhost:8080 \
    --users 200 --spawn-rate 50 --run-time 30s --headless
"""

import random
from locust import HttpUser, between, task

# Short prompts keep inference time lower, showing queue behavior more clearly.
PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing briefly.",
    "What is the capital of France?",
    "Describe the water cycle.",
    "What does CPU stand for?",
    "Name three programming languages.",
    "What is a neural network?",
    "How does HTTPS work?",
    "What is Docker?",
    "Explain REST APIs in one sentence.",
]


class InferenceUser(HttpUser):
    """Simulates a user sending inference requests."""

    # Each user waits 0.5–2s between requests (simulates real usage).
    wait_time = between(0.5, 2.0)

    @task(10)
    def infer(self):
        """Primary task: send an inference request."""
        payload = {
            "prompt":     random.choice(PROMPTS),
            "max_tokens": random.randint(30, 80),
            "priority":   0,
        }
        with self.client.post(
            "/infer",
            json=payload,
            name="/infer",
            catch_response=True,
            timeout=120,   # model inference can be slow on CPU
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get("error"):
                        resp.failure(f"Inference error: {data['error']}")
                    elif not data.get("text"):
                        resp.failure("Empty response text")
                except Exception as e:
                    resp.failure(f"Bad JSON: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """Occasional health check — low weight."""
        self.client.get("/health", name="/health")

    @task(1)
    def scrape_metrics(self):
        """Periodic metrics scrape — low weight."""
        self.client.get("/metrics", name="/metrics")


class BurstUser(HttpUser):
    """
    Simulates a burst of high-priority requests — useful for showing
    that the C++ queue handles concurrent load without dropping connections.
    Activate by running a second locust session or with tags.
    """

    wait_time = between(0.0, 0.1)  # near-zero wait = maximum concurrency

    @task
    def burst_infer(self):
        payload = {
            "prompt":     "Hello",
            "max_tokens": 20,
            "priority":   1,  # high priority — gets served first
        }
        self.client.post("/infer", json=payload, name="/infer [burst]", timeout=120)
