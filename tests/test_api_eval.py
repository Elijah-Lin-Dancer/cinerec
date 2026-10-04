"""Tests for the evaluation-results endpoints.

The contract is honesty: measured artefacts are returned, and a missing artefact
is reported as ``available: false`` rather than substituted with invented values.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_results_endpoint_shape():
    resp = client.get("/api/eval/results")
    assert resp.status_code == 200
    data = resp.json()
    if data.get("available") is False:
        assert "note" in data
        return
    # The real artefact ships with the repo, so this branch is the norm.
    for model in ("UserCF", "ItemCF", "SVD", "NeuMF", "MultiModalNCF"):
        assert model in data
        assert "HR@10" in data[model]


def test_ablation_endpoint_shape():
    resp = client.get("/api/eval/ablation")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("available") is True
    assert any("Full" in key for key in data)
