"""Unit tests for the recommendation explainer (``models/explain.py``).

The explainer turns a recommendation into human-readable reasons. These tests
drive both reason families (content + collaborative, in their UserCF and ItemCF
variants) and the guard branches that keep a malformed index or an unloaded
feature set from raising.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from models.explain import RecommenderExplainer  # noqa: E402

# Three genres → a small, hand-checkable similarity matrix.
MOVIES = [
    {"id": 0, "title": "Alpha", "genres": "Action|Comedy"},
    {"id": 1, "title": "Beta", "genres": "Action|Drama"},
    {"id": 2, "title": "Gamma", "genres": "Action|Comedy"},
    {"id": 3, "title": "Delta", "genres": "Comedy|Drama"},
    {"id": 4, "title": "Epsilon", "genres": "Drama"},
]

# item0 and item2 are identical; item1/item3 overlap partially; item4 stands alone.
GENRES = np.array(
    [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

RATINGS = {
    "user_id": np.array([0, 0, 1]),
    "item_id": np.array([0, 1, 2]),
    "rating": np.array([5.0, 4.0, 3.0]),
}


def _loaded(tmp_path):
    """Explainer with the tiny catalogue above loaded from disk."""
    enriched = tmp_path / "movies_enriched.json"
    enriched.write_text(json.dumps(MOVIES), encoding="utf-8")
    genre_path = tmp_path / "genre_vectors.npy"
    np.save(genre_path, GENRES)

    exp = RecommenderExplainer()
    exp.load_data(str(enriched), str(genre_path))
    exp.load_user_ratings(RATINGS)
    return exp


def test_load_data_with_no_artifacts_stays_usable(tmp_path):
    """A missing artefact must leave the explainer usable, not crashing."""
    exp = RecommenderExplainer()
    exp.load_data(str(tmp_path / "nope.json"), str(tmp_path / "nope.npy"))

    assert exp.movies == []
    assert exp.genre_vecs is None
    assert exp.content_embs is None
    assert exp.item_similarity is None
    # Unknown ids fall back to a placeholder title / empty genres.
    assert exp.get_movie_title(42) == "Movie 42"
    assert exp.get_movie_genres(42) == ""


def test_load_data_builds_similarity_matrix(tmp_path):
    exp = _loaded(tmp_path)
    assert exp.item_similarity.shape == (5, 5)
    # Identical genre vectors → similarity of exactly 1.
    assert exp.item_similarity[0][2] == pytest.approx(1.0)


def test_content_reason_returns_similarity_and_genre_overlap(tmp_path):
    exp = _loaded(tmp_path)
    reasons = exp.content_reason(0, 2)
    kinds = {r["type"] for r in reasons}
    assert "content" in kinds
    assert "genre_match" in kinds

    content = next(r for r in reasons if r["type"] == "content")
    assert content["source_item"] == 0  # item0 is the closest match
    assert content["score"] == pytest.approx(1.0)
    assert "Alpha" in content["reason_en"]

    genre = next(r for r in reasons if r["type"] == "genre_match")
    assert set(genre["genres"]) == {"Action", "Comedy"}


def test_content_reason_needs_history_and_in_range_item(tmp_path):
    exp = _loaded(tmp_path)
    # No history for user 9 → nothing to compare against.
    assert exp.content_reason(9, 2) == []
    # Out-of-range recommendation → guard returns early.
    assert exp.content_reason(0, 99) == []


def test_collaborative_reason_user_cf_path(tmp_path):
    exp = _loaded(tmp_path)
    user_sim = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )

    reasons = exp.collaborative_reason(0, 1, sim_matrix=user_sim, is_user_cf=True, top_k=1)

    assert len(reasons) == 1
    reason = reasons[0]
    assert reason["type"] == "collaborative"
    assert reason["source_user"] == 1
    # User 1's highest-rated item is 2 ("Gamma").
    assert reason["source_item"] == 2
    assert "Gamma" in reason["reason_en"]


def test_collaborative_reason_guards_missing_matrix_and_bad_user(tmp_path):
    exp = _loaded(tmp_path)
    assert exp.collaborative_reason(0, 1, sim_matrix=None, is_user_cf=True) == []
    # user id beyond the matrix width must not index out of bounds.
    assert exp.collaborative_reason(99, 1, sim_matrix=np.eye(3), is_user_cf=True) == []


def test_collaborative_reason_item_cf_only_cites_rated_items(tmp_path):
    exp = _loaded(tmp_path)
    # ItemCF cites similar items the user actually rated; item3 is nearest to
    # item2 but unrated, so it must be skipped and item0 cited instead.
    reasons = exp.collaborative_reason(0, 2, sim_matrix=exp.item_similarity, is_user_cf=False, top_k=2)

    assert len(reasons) == 1
    reason = reasons[0]
    assert reason["type"] == "collaborative"
    assert reason["source_item"] == 0
    assert "Alpha" in reason["reason_en"]


def test_collaborative_reason_item_cf_guards(tmp_path):
    exp = _loaded(tmp_path)
    assert exp.collaborative_reason(0, 2, sim_matrix=None, is_user_cf=False) == []
    assert exp.collaborative_reason(0, 99, sim_matrix=exp.item_similarity, is_user_cf=False) == []


def test_explain_returns_title_and_reasons(tmp_path):
    exp = _loaded(tmp_path)
    result = exp.explain(0, 2)

    assert result["user_id"] == 0
    assert result["item_id"] == 2
    assert result["item_title"] == "Gamma"
    assert result["reasons"]


def test_explain_falls_back_to_default_reason(tmp_path):
    """With no features and no history there is nothing to cite → default."""
    exp = RecommenderExplainer()
    exp.load_data(str(tmp_path / "nope.json"), str(tmp_path / "nope.npy"))

    result = exp.explain(0, 2)

    assert len(result["reasons"]) == 1
    assert result["reasons"][0]["type"] == "default"
    assert result["item_title"] == "Movie 2"


def test_top_rated_items_is_memoised_and_invalidated(tmp_path):
    exp = _loaded(tmp_path)
    first = exp._top_rated_items(0, 20)
    assert exp._top_rated_items(0, 20) is first  # served from the per-user cache

    # Reloading the rating history must invalidate the memoised view.
    exp.load_user_ratings(
        {"user_id": np.array([0]), "item_id": np.array([4]), "rating": np.array([5.0])}
    )
    assert (4, 5.0) in exp._top_rated_items(0, 20)


def test_content_reason_skips_history_beyond_the_feature_matrix(tmp_path):
    """A rated id with no genre row is skipped, never used to index the matrix."""
    exp = _loaded(tmp_path)
    exp.user_ratings = {0: {99: 5.0}}  # item 99 has no feature row
    exp._top_rated_cache.clear()

    assert exp.content_reason(0, 2) == []


def test_content_reason_reuses_memoised_genre_preferences(tmp_path):
    exp = _loaded(tmp_path)
    first = exp.content_reason(0, 2)
    second = exp.content_reason(0, 2)

    assert first == second
    # The second pass is served from the per-user genre-preference cache.
    assert (0, 4.0) in exp._genre_pref_cache


def test_collaborative_reason_user_cf_skips_negligible_similarity(tmp_path):
    exp = _loaded(tmp_path)
    user_sim = np.array(
        [
            [1.0, 0.005, 0.001],
            [0.005, 1.0, 0.0],
            [0.001, 0.0, 1.0],
        ]
    )

    assert exp.collaborative_reason(0, 1, sim_matrix=user_sim, is_user_cf=True, top_k=1) == []


def test_collaborative_reason_item_cf_stops_once_top_k_is_reached(tmp_path):
    exp = _loaded(tmp_path)
    item_sim = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9, 0.3, 1.0, 0.2, 0.1],  # row for the recommended item (2)
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    reasons = exp.collaborative_reason(0, 2, sim_matrix=item_sim, is_user_cf=False, top_k=1)

    assert len(reasons) == 1
    assert reasons[0]["source_item"] == 0
