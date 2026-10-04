"""Tests for the in-process request metrics registry and /api/metrics endpoint."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.auth import make_token  # noqa: E402
from api.main import app  # noqa: E402
from api.metrics import MetricsRegistry, _normalise_path, registry  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {make_token(1)}"}


def test_inference_cache_is_hit_on_repeat_requests():
    """Full-mode inference is memoised: a repeated identical request is a hit."""
    from api.recommend import inference_cache_info

    before = inference_cache_info()["hits"]
    assert client.get("/api/recommend?algorithm=SVD&top_k=5", headers=AUTH).status_code == 200
    assert client.get("/api/recommend?algorithm=SVD&top_k=5", headers=AUTH).status_code == 200
    assert inference_cache_info()["hits"] >= before + 1


def test_path_normalisation_collapses_numeric_ids():
    assert _normalise_path("/api/movies/42") == "/api/movies/{id}"
    assert _normalise_path("/api/recommend/7/explain") == "/api/recommend/{id}/explain"
    assert _normalise_path("/api/recommend") == "/api/recommend"


def test_registry_counts_status_and_percentiles():
    reg = MetricsRegistry()
    for ms in (1.0, 2.0, 3.0, 4.0, 100.0):
        reg.record("GET", "/x", 200, ms)
    reg.record("GET", "/x", 500, 5.0)

    snap = reg.snapshot()
    assert snap["total_requests"] == 6
    assert snap["total_5xx"] == 1
    assert snap["status_counts"] == {"200": 5, "500": 1}

    endpoint = snap["endpoints"][0]
    assert endpoint["method"] == "GET"
    assert endpoint["path"] == "/x"
    assert endpoint["count"] == 6
    assert endpoint["p50_ms"] is not None
    assert endpoint["p95_ms"] >= endpoint["p50_ms"]


def test_registry_empty_snapshot_is_safe():
    snap = MetricsRegistry().snapshot()
    assert snap["total_requests"] == 0
    assert snap["endpoints"] == []
    assert snap["avg_latency_ms"] == 0.0


def test_metrics_endpoint_reports_traffic_and_cache():
    before = registry.snapshot()["total_requests"]
    assert client.get("/api/health").status_code == 200

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_requests"] > before
    assert "inference_cache" in data
    assert any(e["path"] == "/api/health" for e in data["endpoints"])
