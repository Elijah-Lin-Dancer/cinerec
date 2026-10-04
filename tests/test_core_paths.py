"""Core-path tests: base class, metrics runner, registry, schema, XAI and NeuMF.

These target the modules the API and evaluation pipeline actually depend on, so
the suite covers real behaviour rather than only the leaf metric helpers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from models.base import Recommender  # noqa: E402

# A tiny deterministic interaction set: 3 users × 5 items.
TRAIN = {
    "user_id": np.array([0, 0, 1, 1, 2, 2, 2]),
    "item_id": np.array([1, 2, 2, 3, 1, 3, 4]),
    "rating": np.array([5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 5.0]),
}


# --- models.base -----------------------------------------------------------

class _Dummy(Recommender):
    """Minimal concrete recommender used to exercise the base-class helpers."""

    def fit(self, train_data):
        self.matrix = np.ones((2, 2), dtype=np.float64)

    def predict(self, user_id, item_id):
        return 1.0

    def recommend(self, user_id, top_k=10, exclude_items=None):
        return [(i, 1.0) for i in range(top_k)]


def test_base_save_load_roundtrip_downcasts_float64(tmp_path):
    model = _Dummy()
    model.fit(TRAIN)
    assert model.recommend_ids(0, top_k=3) == [0, 1, 2]

    path = tmp_path / "dummy.pkl"
    model.save(str(path))
    loaded = Recommender.load(str(path))
    assert loaded.name == "_Dummy"
    assert loaded.matrix.dtype == np.float32  # float64 downcast keeps artefacts small


# --- evaluation.metrics ----------------------------------------------------

def test_evaluate_model_scores_against_filtered_candidates():
    from evaluation.metrics import evaluate_model

    class Stub:
        def recommend_ids(self, user_id, top_k=10, exclude_items=None):
            exclude = exclude_items or set()
            pool = [i for i in range(1, top_k + 20) if i not in exclude]
            return pool[:top_k]

    result = evaluate_model(Stub(), [(0, {1})], {0: {5}}, k_values=(5,))
    # Item 1 is relevant, unseen in training, and returned in the top-5.
    assert result["HR@5"] == 1.0
    assert result["Recall@5"] == 1.0
    assert 0.0 < result["NDCG@5"] <= 1.0


# --- models.registry -------------------------------------------------------

def test_registry_unknown_algorithm_returns_none():
    from models.registry import load_model

    assert load_model("NotAnAlgorithm") is None


def test_registry_missing_artifact_raises_unavailable(monkeypatch, tmp_path):
    from models import registry

    monkeypatch.setattr(registry, "PROCESSED_DIR", str(tmp_path))
    with pytest.raises(registry.AlgorithmUnavailable):
        registry.load_model("SVD")


# --- db schema -------------------------------------------------------------

def test_database_schema_is_idempotent():
    from db.database import get_connection, init_db

    init_db()
    init_db()  # a second run must not raise (fresh-clone safety)

    conn = get_connection()
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"users", "movies", "ratings"} <= names


# --- explainability --------------------------------------------------------

def test_explainer_returns_reasons_for_a_recommendation():
    from models.explain import RecommenderExplainer

    explainer = RecommenderExplainer()
    explainer.load_data()
    explainer.load_user_ratings(
        {"user_id": np.array([1]), "item_id": np.array([1]), "rating": np.array([5.0])}
    )

    out = explainer.explain(1, 1)
    assert out["item_id"] == 1
    assert isinstance(out["reasons"], list) and out["reasons"]


# --- NeuMF (ladder member) --------------------------------------------------

def test_neumf_fit_predict_recommend_smoke():
    pytest.importorskip("torch")
    from models.neumf import NeuMF

    model = NeuMF(embedding_dim=8, mlp_dims=(16, 8), epochs=2)
    model.fit(TRAIN)

    score = model.predict(0, 1)
    assert 0.0 <= score <= 1.0

    recs = model.recommend(0, top_k=3, exclude_items={1})
    assert 1 not in [item for item, _ in recs]


# --- api.recommend cold-start fallback --------------------------------------

def test_popular_items_fallback_returns_well_rated_unseen_items():
    from api.recommend import _popular_items
    from db.database import get_connection

    conn = get_connection()
    # 25 ratings (avg 5.0) for movie 1 clears the min-support threshold (n >= 20).
    conn.executemany(
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
        [(1000 + i, 1, 5.0, i) for i in range(25)],
    )
    conn.commit()

    out = _popular_items(conn, exclude=set(), top_k=3)
    excluded = _popular_items(conn, exclude={1}, top_k=3)
    conn.close()

    assert out and out[0][0] == 1 and out[0][1] == 5.0
    assert all(item_id != 1 for item_id, _ in excluded)
