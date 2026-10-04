"""Tests for the movie browsing / rating endpoints."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.auth import make_token  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {make_token(1)}"}


def test_listing_paginates():
    resp = client.get("/api/movies?per_page=2&page=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["movies"]) <= 2
    assert data["page"] == 1
    assert data["pages"] >= 1


def test_search_filters_by_title():
    resp = client.get("/api/movies?search=Alpha")
    assert resp.status_code == 200
    titles = [m["title"] for m in resp.json()["movies"]]
    assert "Alpha" in titles


def test_genre_filter():
    resp = client.get("/api/movies?genre=Comedy")
    assert resp.status_code == 200
    assert all("Comedy" in (m["genres"] or "") for m in resp.json()["movies"])


def test_genres_endpoint_is_sorted_and_unique():
    genres = client.get("/api/movies/genres").json()["genres"]
    assert genres == sorted(set(genres))
    assert "Action" in genres


def test_movie_detail_and_404():
    assert client.get("/api/movies/1").json()["title"] == "Alpha"
    assert client.get("/api/movies/9999").status_code == 404


def test_rating_requires_token_and_persists():
    assert client.post("/api/movies/2/rate?rating=5").status_code == 401

    resp = client.post("/api/movies/2/rate?rating=4", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == 1

    detail = client.get("/api/movies/2").json()
    assert detail["rating_count"] >= 1
    assert detail["avg_rating"] is not None


def test_rating_rejects_out_of_range():
    assert client.post("/api/movies/2/rate?rating=9", headers=AUTH).status_code == 422
