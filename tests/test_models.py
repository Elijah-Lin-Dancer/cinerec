"""Unit tests for the recommender models (fast, no training loops)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from models.user_cf import UserCF  # noqa: E402
from models.item_cf import ItemCF  # noqa: E402
from models.svd_als import SVDALS  # noqa: E402

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


def test_lightgcn_recommends_and_excludes():
    pytest.importorskip("torch")
    from models.lightgcn import LightGCN
    model = LightGCN(embedding_dim=8, num_layers=2, epochs=2, batch_size=4)
    _check_recommender(model)
    # Same boundary contract as the numpy models (neural recommend path).
    assert model.recommend(-1) == []
    assert model.recommend(model.num_users) == []
    assert model.recommend(0, top_k=0) == []


def test_svd_is_truncated_svd_not_als():
    """Guard the honest naming: the class must expose the truncated-SVD factors."""
    model = SVDALS(k=3)
    model.fit(TRAIN)
    assert model.U.shape[1] == model.k
    assert model.V.shape[1] == model.k


@pytest.mark.parametrize(
    "factory",
    [lambda: UserCF(k=2), lambda: ItemCF(k=2), lambda: SVDALS(k=3)],
)
def test_models_guard_out_of_range_and_nonpositive_top_k(factory):
    """Malformed indices and non-positive ``top_k`` must degrade safely.

    Regression: the numpy models sliced with ``[-top_k:]``, so ``top_k=0``
    silently returned *every* item, and negative user ids wrapped around via
    Python's negative indexing.
    """
    model = factory()
    model.fit(TRAIN)

    assert model.recommend(-1) == []
    assert model.recommend(model.num_users) == []
    assert model.recommend(0, top_k=0) == []
    assert model.recommend(0, top_k=-5) == []

    # Out-of-range exclusion ids are ignored, not wrapped around.
    assert isinstance(model.recommend(0, top_k=3, exclude_items={-1, 10_000}), list)

    # Prediction must never raise on a malformed index.
    assert np.isfinite(model.predict(-1, 0))
    assert np.isfinite(model.predict(0, 10_000))


def test_multimodal_ablation_zeroes_disabled_modalities():
    """A disabled modality must be blanked before training (ablation hook)."""
    pytest.importorskip("torch")
    from models.multimodal_ncf import MultiModalNCF

    model = MultiModalNCF(disabled_features=["text", "image"])
    model.num_items = 1683  # matches the shipped feature arrays
    model._load_content_features()

    assert np.count_nonzero(model.text_emb) == 0
    assert np.count_nonzero(model.image_emb) == 0
    assert np.count_nonzero(model.genre_vec) > 0


def test_multimodal_cold_path_runs_without_item_embedding():
    """``predict_cold`` scores a brand-new item from content features alone."""
    torch = pytest.importorskip("torch")
    from models.multimodal_ncf import MultiModalNCF, MultiModalNCFNet

    model = MultiModalNCF(embedding_dim=8, mlp_dims=(16, 8))
    model.num_users, model.num_items = 3, 8
    model._load_content_features()
    model.net = MultiModalNCFNet(model.num_users, model.num_items, embedding_dim=8, mlp_dims=(16, 8))

    text = np.zeros(384, dtype=np.float32)
    image = np.zeros(2048, dtype=np.float32)
    genre = np.zeros(18, dtype=np.float32)
    genre[0] = 1.0

    score = model.predict_cold(0, text, image, genre)
    assert 0.0 <= score <= 1.0
    assert torch.isfinite(torch.tensor(score))
