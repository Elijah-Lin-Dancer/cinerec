"""Regression tests for the frontend catch-all route (path traversal, S1)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api.main import app, _safe_frontend_path, FRONTEND_DIR

client = TestClient(app)


def test_safe_frontend_path_blocks_traversal():
    assert _safe_frontend_path("../.env") is None
    assert _safe_frontend_path("../../etc/passwd") is None
    assert _safe_frontend_path("../db/cinerec.db") is None


def test_safe_frontend_path_allows_inside():
    assert _safe_frontend_path("index.html") == os.path.join(FRONTEND_DIR, "index.html")


def test_traversal_request_does_not_leak_env():
    """Encoded `../` must 404 and never read files outside the frontend directory."""
    response = client.get("/%2e%2e/.env")
    assert response.status_code == 404
    assert "TMDB_API_KEY" not in response.text
    assert "sqlite" not in response.text


def test_traversal_request_does_not_leak_source():
    response = client.get("/%2e%2e%2fdb/database.py")
    assert response.status_code == 404
    assert "import" not in response.text


def test_root_serves_spa_shell():
    """`/` must serve index.html, not fall through the catch-all (which 404s).

    The catch-all `/{path:path}` matches "/" with an empty path, so the explicit
    root route has to be registered first — this test locks that in.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text


def test_unknown_client_route_falls_back_to_spa():
    response = client.get("/some/client/route")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text
