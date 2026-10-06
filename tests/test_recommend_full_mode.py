"""Full-mode /api/recommend branch coverage with the model layer stubbed.

The live-inference path (``full`` mode) is pinned here without torch or the
shipped artefacts by replacing the two seams the endpoint owns —
``_cached_model_recs`` (inference) and ``get_explainer`` (explanations). That
keeps the tests about the *contract*: an unavailable algorithm reports 503, an
unexpected inference error reports 500, empty model output degrades to a
labelled popularity fallback, and a failing explainer degrades to empty reasons
rather than taking the whole response down.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.recommend as rec  # noqa: E402
from api.auth import make_token  # noqa: E402
from api.main import app  # noqa: E402
from models.registry import AlgorithmUnavailable  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {make_token(1)}"}


@pytest.fixture(autouse=True)
def _clear_inference_lru():
    """The inference LRU is process-global; stop stubs leaking between tests."""
    rec._cached_model_recs.cache_clear()
    yield
    rec._cached_model_recs.cache_clear()


class _StubExplainer:
    def __init__(self, reasons):
        self._reasons = reasons

    def explain(self, user_id, movie_id):
        return {"reasons": self._reasons}


def _stub_recs(*items):
    """Stand in for ``_cached_model_recs`` with a fixed tuple of (id, score)."""
    return lambda *args, **kwargs: tuple(items)


def test_unavailable_algorithm_reports_503(monkeypatch):
    def _unavailable(name):
        raise AlgorithmUnavailable("torch is not installed in this deployment")

    monkeypatch.setattr(rec, "_get_model", _unavailable)
    resp = client.get("/api/recommend?algorithm=NeuMF&top_k=3", headers=AUTH)
    assert resp.status_code == 503
    # The endpoint surfaces the real reason rather than a generic message.
    assert resp.json()["detail"] == "torch is not installed in this deployment"


def test_unexpected_inference_error_reports_500(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("model blew up")

    monkeypatch.setattr(rec, "_cached_model_recs", _boom)
    resp = client.get("/api/recommend?algorithm=SVD&top_k=3", headers=AUTH)
    assert resp.status_code == 500
    assert "Inference failed" in resp.json()["detail"]


def test_empty_model_output_uses_labelled_popularity_fallback(monkeypatch):
    monkeypatch.setattr(rec, "_cached_model_recs", _stub_recs())  # model returns nothing
    monkeypatch.setattr(rec, "_popular_items", lambda conn, exclude, top_k: [(1, 4.5), (2, 4.0)])

    resp = client.get("/api/recommend?algorithm=SVD&top_k=3", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback"] is True
    assert body["note"]  # the fallback is labelled, never silent
    assert body["total"] == 2
    # Popularity rows must not masquerade as personalized recommendations.
    assert all("reasons" not in item for item in body["recommendations"])


def test_explainer_failure_degrades_to_empty_reasons(monkeypatch):
    monkeypatch.setattr(rec, "_cached_model_recs", _stub_recs((1, 0.9), (2, 0.8)))

    def _boom():
        raise RuntimeError("explainer unavailable")

    monkeypatch.setattr(rec, "get_explainer", _boom)
    resp = client.get("/api/recommend?algorithm=SVD&top_k=3", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback"] is False
    assert body["recommendations"]
    assert all(item["reasons"] == [] for item in body["recommendations"])


def test_explainer_success_attaches_reasons(monkeypatch):
    monkeypatch.setattr(rec, "_cached_model_recs", _stub_recs((1, 0.9)))
    monkeypatch.setattr(rec, "get_explainer", lambda: _StubExplainer(["you liked similar films"]))

    resp = client.get("/api/recommend?algorithm=SVD&top_k=3", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["recommendations"][0]["reasons"] == ["you liked similar films"]


def test_explain_endpoint_reports_failure_as_500(monkeypatch):
    def _boom():
        raise RuntimeError("explainer unavailable")

    monkeypatch.setattr(rec, "get_explainer", _boom)
    assert client.get("/api/recommend/1/explain", headers=AUTH).status_code == 500


def test_explain_endpoint_returns_explainer_payload(monkeypatch):
    monkeypatch.setattr(rec, "get_explainer", lambda: _StubExplainer(["because"]))
    resp = client.get("/api/recommend/1/explain", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["reasons"] == ["because"]


def test_inference_lru_is_keyed_by_the_exclusion_set(monkeypatch):
    calls = []

    class _CountingModel:
        def recommend(self, user_id, top_k=10, exclude_items=None):
            calls.append((user_id, tuple(sorted(exclude_items or ()))))
            return [(3, 0.5)]

    monkeypatch.setattr(rec, "_get_model", lambda name: _CountingModel())

    rec._cached_model_recs("SVD", 1, ())
    rec._cached_model_recs("SVD", 1, ())  # identical key -> served from the LRU
    assert len(calls) == 1
    rec._cached_model_recs("SVD", 1, (3,))  # a new rating changes the key
    assert len(calls) == 2
