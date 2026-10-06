"""Tests for the evaluation charts (``evaluation/visualize.py``).

Every chart is driven by an on-disk artefact and must be *skipped* — never
drawn from placeholder numbers — when that artefact is missing. These tests
assert both the skip paths and that the chart files are really written from
supplied data.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

pytest.importorskip("matplotlib", reason="visualizations need matplotlib")

import evaluation.visualize as viz  # noqa: E402

RESULTS = {
    name: {"HR@10": 0.30, "NDCG@10": 0.25, "Recall@10": 0.32, "train_time": 1.5}
    for name in viz.MODEL_NAMES_ORDER
}

ABLATION = {
    "available": True,
    "_note": "measured variants only",
    "full": {"HR@10": 0.35, "NDCG@10": 0.30},
    "no_text": {"HR@10": 0.30, "NDCG@10": 0.26},
}

COLDSTART = {
    "n_cold_items": 50,
    "metrics": {
        "content_aware": {"HR@10": 0.22, "NDCG@10": 0.16},
        "content_blind": {"HR@10": 0.09, "NDCG@10": 0.07},
    },
}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    docs = tmp_path / "docs"
    processed.mkdir()
    docs.mkdir()
    monkeypatch.setattr(viz, "PROCESSED_DIR", str(processed))
    monkeypatch.setattr(viz, "DOCS_DIR", str(docs))
    return processed, docs


def test_load_json_returns_none_for_missing_artifact(dirs):
    processed, _ = dirs
    assert viz.load_json("eval_results.json") is None

    (processed / "eval_results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    assert viz.load_json("eval_results.json") == RESULTS


def test_generate_all_skips_everything_without_artifacts(dirs):
    _, docs = dirs
    viz.generate_all()
    # No artefact → no chart: nothing is drawn from placeholder numbers.
    assert list(docs.iterdir()) == []


def test_plot_model_comparison_and_training_time_write_charts(dirs):
    _, docs = dirs
    viz.plot_model_comparison(RESULTS)
    viz.plot_training_time(RESULTS)
    assert (docs / "model_comparison.png").exists()
    assert (docs / "training_time.png").exists()


def test_plot_model_comparison_is_a_noop_without_results(dirs):
    _, docs = dirs
    viz.plot_model_comparison(None)
    assert not (docs / "model_comparison.png").exists()


def test_plot_training_time_skips_when_no_train_time_recorded(dirs):
    _, docs = dirs
    viz.plot_training_time({name: {"HR@10": 0.3} for name in viz.MODEL_NAMES_ORDER})
    assert not (docs / "training_time.png").exists()


def test_plot_ablation_skips_when_artifact_unavailable(dirs):
    _, docs = dirs
    viz.plot_ablation({"available": False})
    assert not (docs / "ablation_study.png").exists()


def test_plot_ablation_writes_chart_when_measured(dirs):
    _, docs = dirs
    viz.plot_ablation(ABLATION)
    assert (docs / "ablation_study.png").exists()


def test_plot_coldstart_writes_chart(dirs):
    _, docs = dirs
    viz.plot_coldstart(COLDSTART)
    assert (docs / "coldstart.png").exists()


def test_generate_all_writes_every_chart_from_artifacts(dirs):
    processed, docs = dirs
    (processed / "eval_results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    (processed / "ablation_results.json").write_text(json.dumps(ABLATION), encoding="utf-8")
    (processed / "coldstart_results.json").write_text(json.dumps(COLDSTART), encoding="utf-8")

    viz.generate_all()

    assert (docs / "model_comparison.png").exists()
    assert (docs / "training_time.png").exists()
    assert (docs / "ablation_study.png").exists()
    assert (docs / "coldstart.png").exists()


def test_plot_model_comparison_pads_models_and_metrics_that_are_missing(dirs):
    """A partial results file still charts: gaps read as zero, not as a crash."""
    _, docs = dirs
    viz.plot_model_comparison({"UserCF": {"HR@10": 0.3}})
    assert (docs / "model_comparison.png").exists()


def test_plot_training_time_is_a_noop_without_results(dirs):
    _, docs = dirs
    viz.plot_training_time(None)
    assert not (docs / "training_time.png").exists()


def test_plot_ablation_is_a_noop_without_variants(dirs):
    _, docs = dirs
    viz.plot_ablation({"available": True})
    assert not (docs / "ablation_study.png").exists()


def test_plot_coldstart_is_a_noop_without_metrics(dirs):
    _, docs = dirs
    viz.plot_coldstart({"n_cold_items": 5})
    assert not (docs / "coldstart.png").exists()
