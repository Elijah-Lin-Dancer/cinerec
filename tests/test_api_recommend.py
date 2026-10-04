"""Tests for the recommendation endpoint contract: auth, validation, shape."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.auth import make_token  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {make_token(1)}"}


def test_recommend_requires_session_token():
    assert client.get("/api/recommend?algorithm=SVD").status_code == 401


def test_recommend_rejects_requesting_another_user():
    resp = client.get("/api/recommend?user_id=2&algorithm=SVD", headers=AUTH)
    assert resp.status_code == 403


def test_recommend_rejects_unknown_algorithm():
    resp = client.get("/api/recommend?algorithm=DoesNotExist", headers=AUTH)
    assert resp.status_code == 400


def test_recommend_returns_documented_shape():
    resp = client.get("/api/recommend?algorithm=SVD&top_k=5", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 1
    assert data["algorithm"] == "SVD"
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["fallback"], bool)
    assert "note" in data


def test_explain_requires_session_token():
    assert client.get("/api/recommend/1/explain").status_code == 401


def test_rating_requires_session_token():
    assert client.post("/api/movies/1/rate?rating=5").status_code == 401


def test_movies_listing_is_public():
    resp = client.get("/api/movies")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_health_endpoint():
    assert client.get("/api/health").json() == {"status": "ok"}
