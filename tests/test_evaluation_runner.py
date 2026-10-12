"""Tests for the offline evaluation runner (``evaluation/runner.py``).

``run_evaluation`` trains six models end-to-end, which is far too slow for the
unit suite. These tests instead pin the two things that must hold: the
leave-last-N-out split, and the runner's persistence / error contract — driven
by stand-in model classes and a stubbed metric, so no training happens.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

# ``evaluation/runner.py`` imports the torch-backed models at module scope, so
# this file can only load where torch is installed. Without the gate the e2e job
# — which deliberately installs no torch — dies at *collection* with
# ModuleNotFoundError instead of skipping the module.
pytest.importorskip("torch", reason="evaluation.runner imports the torch-backed models")

import evaluation.runner as runner  # noqa: E402

RATINGS_CSV = """user_id,item_id,rating,timestamp
1,10,4.0,100
1,11,5.0,200
1,12,3.0,300
1,13,2.0,400
2,10,5.0,100
2,11,2.0,200
2,12,4.0,300
2,13,3.0,400
"""

# The class globals ``run_evaluation`` instantiates (note "SVD" is produced by
# ``SVDALS``, so the key in ``MODEL_FILES`` is not the attribute name).
MODEL_GLOBALS = ["UserCF", "ItemCF", "SVDALS", "NeuMF", "LightGCN", "MultiModalNCF"]


def test_load_data_builds_leave_last_n_out_split(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ratings.csv").write_text(RATINGS_CSV, encoding="utf-8")
    monkeypatch.setattr(runner, "RAW_DIR", str(raw))

    train_data, test_pairs, train_by_user, full_df = runner.load_data(n_test=2)

    # 8 interactions total, 2 held out per user → 4 training rows.
    assert len(full_df) == 8
    assert len(train_data["user_id"]) == 4

    assert sorted(train_by_user) == [1, 2]
    assert train_by_user[1] == {10, 11}
    assert train_by_user[2] == {10, 11}

    # The two most recent interactions per user form the test set.
    assert dict(test_pairs) == {1: {12, 13}, 2: {12, 13}}


class _StubModel:
    """Records the training call and writes a stand-in artefact."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_with = None

    def fit(self, train_data):
        self.fitted_with = train_data

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("stub")


class _FailingModel(_StubModel):
    def fit(self, train_data):
        raise RuntimeError("boom")


def _patch_runner(monkeypatch, tmp_path, failing=None):
    results_dir = tmp_path / "processed"
    results_dir.mkdir()
    monkeypatch.setattr(runner, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(
        runner,
        "load_data",
        lambda n_test=5: ({"user_id": [], "item_id": [], "rating": []}, [], {}, None),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_model",
        lambda model, test_pairs, train_by_user, k_values: {
            "HR@10": 0.5,
            "NDCG@10": 0.4,
            "Recall@10": 0.3,
        },
    )
    for name in MODEL_GLOBALS:
        monkeypatch.setattr(runner, name, _FailingModel if name == failing else _StubModel)
    return results_dir


def test_run_evaluation_saves_metrics_and_artifacts(tmp_path, monkeypatch):
    results_dir = _patch_runner(monkeypatch, tmp_path)

    all_results = runner.run_evaluation()

    assert set(all_results) == set(runner.MODEL_FILES)
    for name, res in all_results.items():
        assert res["HR@10"] == 0.5
        assert res["NDCG@10"] == 0.4
        assert res["Recall@10"] == 0.3
        assert "train_time" in res
        # Each model persisted its artefact under the configured results dir.
        assert os.path.exists(os.path.join(str(results_dir), runner.MODEL_FILES[name]))

    on_disk = json.loads((results_dir / "eval_results.json").read_text(encoding="utf-8"))
    assert on_disk == all_results


def test_run_evaluation_records_a_failing_model_without_aborting(tmp_path, monkeypatch):
    """One model blowing up must be recorded, not kill the whole run."""
    _patch_runner(monkeypatch, tmp_path, failing="NeuMF")

    all_results = runner.run_evaluation()

    assert "boom" in all_results["NeuMF"]["error"]
    # The other five still produced metrics.
    assert "HR@10" in all_results["UserCF"]
    assert "HR@10" in all_results["MultiModalNCF"]


class _SeedAwareStub(_StubModel):
    """Stand-in whose metric is derived from the seed it was constructed with,
    so the multi-seed aggregation can be checked without training anything."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = kwargs.get("seed", 42)

    def evaluate(self):
        return {"HR@10": self.seed / 100}


class _FailOnSecondSeed(_SeedAwareStub):
    def fit(self, train_data):
        if self.seed == 2:
            raise RuntimeError("boom")


def _patch_seed_stubs(monkeypatch, tmp_path, failing=None):
    results_dir = tmp_path / "processed"
    results_dir.mkdir()
    monkeypatch.setattr(runner, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(
        runner,
        "load_data",
        lambda n_test=5: ({"user_id": [], "item_id": [], "rating": []}, [], {}, None),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_model",
        lambda model, test_pairs, train_by_user, k_values: model.evaluate(),
    )
    for name in MODEL_GLOBALS:
        monkeypatch.setattr(runner, name, _FailOnSecondSeed if name == failing else _SeedAwareStub)
    return results_dir


def test_run_evaluation_multi_seed_aggregates_mean_std(tmp_path, monkeypatch):
    """``n_seeds>1`` must vary the training seed and report mean/std per metric."""
    _patch_seed_stubs(monkeypatch, tmp_path)

    all_results = runner.run_evaluation(n_seeds=2, seeds=[1, 2])

    res = all_results["NeuMF"]
    assert res["HR@10"] == {"mean": 0.015, "std": 0.005}
    assert res["seed_list"] == [1, 2]
    assert res["n_seeds"] == 2


def test_run_evaluation_multi_seed_records_a_failing_model_without_aborting(tmp_path, monkeypatch):
    """A seed blowing up is recorded for that model; the rest still aggregate."""
    _patch_seed_stubs(monkeypatch, tmp_path, failing="NeuMF")

    all_results = runner.run_evaluation(n_seeds=2, seeds=[1, 2])

    assert "boom" in all_results["NeuMF"]["error"]
    assert all_results["UserCF"]["n_seeds"] == 2
    assert all_results["MultiModalNCF"]["HR@10"] == {"mean": 0.015, "std": 0.005}
