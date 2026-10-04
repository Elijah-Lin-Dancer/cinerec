"""Locust load profile for CineRec.

Exercises the same flows a reviewer clicks: sign in as a guest, browse the
library, and pull recommendations across the algorithm ladder.

Run against a **lite** server (no torch needed):

    APP_MODE=lite uvicorn api.main:app --port 8000 &
    locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s \
        --host http://127.0.0.1:8000 --html reports/locust_report.html

Or via ``make loadtest`` (the server must already be running).
"""
import random

from locust import HttpUser, between, task

ALGORITHMS = ["UserCF", "ItemCF", "SVD", "NeuMF", "LightGCN", "MultiModalNCF"]


class CineRecUser(HttpUser):
    """A guest browsing and requesting recommendations."""

    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Guest login yields a user id + session token without credentials."""
        with self.client.get("/api/auth/guest", catch_response=True, name="/api/auth/guest") as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.user_id = data.get("user_id")
                self.headers = {"Authorization": f"Bearer {data.get('token')}"}
                resp.success()
            else:
                self.user_id = None
                self.headers = {}
                resp.failure(f"guest login failed: {resp.status_code}")

    @task(1)
    def health(self):
        self.client.get("/api/health")

    @task(3)
    def browse_movies(self):
        self.client.get("/api/movies?per_page=20")

    @task(4)
    def get_recommendations(self):
        if self.user_id is None:
            return
        algorithm = random.choice(ALGORITHMS)
        with self.client.get(
            f"/api/recommend?algorithm={algorithm}&top_k=10",
            headers=self.headers,
            catch_response=True,
            name="/api/recommend",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code in (401, 403, 503):
                # Auth rejections and "algorithm not in this deployment" are
                # expected outcomes of the contract, not load-test failures.
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")
