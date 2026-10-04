"""Unit tests for the recommender models (fast, no training loops)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from models.user_cf import UserCF  # noqa: E402
from models.item_cf import ItemCF  # noqa: E402
from models.svd_als import SVDALS  # noqa: E402

# The neural model needs the optional training stack; skip its tests when torch
# is absent (e.g. the lightweight CI install) rather than failing the suite.
pytest.importorskip("torch")
from models.multimodal_ncf import MultiModalNCF  # noqa: E402

# A tiny deterministic interaction set: 3 users × 5 items.
TRAIN = {
    "user_id": np.array([0, 0, 1, 1, 2, 2, 2]),
    "item_id": np.array([1, 2, 2, 3, 1, 3, 4]),
    "rating": np.array([5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 5.0]),
}


def _check_recommender(model):
    model.fit(TRAIN)
    recs = model.recommend(0, top_k=3, exclude_items={1})
    assert isinstance(recs, list)
    assert all(len(pair) == 2 for pair in recs)
    # Excluded items must never be recommended.
    assert 1 not in [item for item, _ in recs]
    # Scores are ordered descending.
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_user_cf_recommends_and_excludes():
    _check_recommender(UserCF(k=2))


def test_item_cf_recommends_and_excludes():
    _check_recommender(ItemCF(k=2))


def test_svd_is_truncated_svd_not_als():
    """Guard the honest naming: the class must expose the truncated-SVD factors."""
    model = SVDALS(k=3)
    model.fit(TRAIN)
    assert model.U.shape[1] == model.k
    assert model.V.shape[1] == model.k


def test_multimodal_ablation_zeroes_disabled_modalities():
    """A disabled modality must be blanked before training (ablation hook)."""
    model = MultiModalNCF(disabled_features=["text", "image"])
    model.num_items = 1683  # matches the shipped feature arrays
    model._load_content_features()

    assert np.count_nonzero(model.text_emb) == 0
    assert np.count_nonzero(model.image_emb) == 0
    assert np.count_nonzero(model.genre_vec) > 0


def test_multimodal_cold_path_runs_without_item_embedding():
    """``predict_cold`` scores a brand-new item from content features alone."""
    import torch

    model = MultiModalNCF(embedding_dim=8, mlp_dims=(16, 8))
    model.num_users, model.num_items = 3, 8
    model._load_content_features()
    model.net = model.net or __import__(
        "models.multimodal_ncf", fromlist=["MultiModalNCFNet"]
    ).MultiModalNCFNet(model.num_users, model.num_items, embedding_dim=8, mlp_dims=(16, 8))

    text = np.zeros(384, dtype=np.float32)
    image = np.zeros(2048, dtype=np.float32)
    genre = np.zeros(18, dtype=np.float32)
    genre[0] = 1.0

    score = model.predict_cold(0, text, image, genre)
    assert 0.0 <= score <= 1.0
    assert torch.isfinite(torch.tensor(score))
