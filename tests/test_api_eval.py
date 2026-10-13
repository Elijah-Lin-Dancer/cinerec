"""Tests for the evaluation-results endpoints.

The contract is honesty: measured artefacts are returned, and a missing artefact
is reported as ``available: false`` rather than substituted with invented values.
The multiseed artefacts (mean ± std over seeds) take priority over the
single-seed fallbacks, and the cross-dataset endpoint pairs both domains.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)

EVAL_MODELS = ("UserCF", "ItemCF", "SVD", "NeuMF", "LightGCN", "MultiModalNCF")


def test_results_endpoint_shape():
    resp = client.get("/api/eval/results")
    assert resp.status_code == 200
    data = resp.json()
    if data.get("available") is False:
        assert "note" in data
        return
    # The real artefacts ship with the repo, so this branch is the norm.
    for model in EVAL_MODELS:
        assert model in data
        assert "HR@10" in data[model]
    # The multiseed artefacts ship too, so the endpoint prefers them.
    assert data["multiseed"] is True
    assert isinstance(data["MultiModalNCF"]["HR@10"], dict)


def test_results_multiseed_preferred_over_single_seed():
    """If both artefacts exist, mean ± std wins over the flat values."""
    from api import eval_api
    assert eval_api._load_multiseed("eval")["multiseed"] is True


def test_ablation_endpoint_shape():
    resp = client.get("/api/eval/ablation")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("available") is True
    assert any("Full" in key for key in data)
    assert data["multiseed"] is True


def test_coldstart_endpoint_shape():
    resp = client.get("/api/eval/coldstart")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("available") is True
    metrics = data["metrics"]
    for key in ("MultiModalNCF (content)", "NeuMF (content-blind)"):
        assert key in metrics
        assert "HR@10" in metrics[key]


def test_cross_endpoint_pairs_both_domains():
    resp = client.get("/api/eval/cross")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    # MovieLens side: always present in this repo.
    assert data["movielens"]["eval"]["available"] is True
    # Amazon side: committed with the RQ5 study; either state is honest, but a
    # present artefact must carry the same multiseed marker.
    amz = data["amazon"]["eval"]
    assert amz["available"] is True
    assert amz["multiseed"] is True
    for model in ("LightGCN", "BPRMF", "VBPR"):
        assert model in amz
