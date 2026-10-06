"""Minimal forward / round-trip tests for the three torch-backed models.

The numpy models already have a contract test in ``test_models.py``; these are
its neural counterpart for NeuMF, LightGCN and MultiModalNCF — the modules that
otherwise sit near zero coverage because they need torch.

Gated with ``pytest.importorskip``: CI installs the CPU wheel so they run there,
while lite-mode environments (no torch) skip the whole file. Every case trains a
single epoch on a handful of interactions, so the file stays fast.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

#: 3 users x 5 items; all three models treat rating >= 4.0 as a positive.
TRAIN = {
    "user_id": np.array([0, 0, 1, 1, 2, 2, 2]),
    "item_id": np.array([1, 2, 2, 3, 1, 3, 4]),
    "rating": np.array([5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 5.0]),
}
NUM_USERS, NUM_ITEMS = 3, 5


def _assert_recommend_contract(model):
    """The boundary contract every model in the ladder is expected to honour."""
    assert model.recommend(-1) == []
    assert model.recommend(model.num_users) == []
    assert model.recommend(0, top_k=0) == []
    assert model.recommend(0, top_k=-5) == []

    recs = model.recommend(0, top_k=3, exclude_items={1})
    assert isinstance(recs, list)
    assert all(len(pair) == 2 for pair in recs)
    assert 1 not in [item for item, _ in recs]
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


# --- NeuMF -------------------------------------------------------------------


def test_neumf_net_forward_returns_probabilities():
    from models.neumf import NeuMFNet

    net = NeuMFNet(NUM_USERS, NUM_ITEMS, embedding_dim=8, mlp_dims=(16, 8)).eval()
    with torch.no_grad():
        scores = net(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))

    assert scores.shape == (3,)
    assert bool((scores >= 0).all()) and bool((scores <= 1).all())


def test_neumf_fit_predict_recommend_and_roundtrip(tmp_path):
    from models.neumf import NeuMF

    model = NeuMF(embedding_dim=8, mlp_dims=(16, 8), epochs=1, batch_size=4, num_neg=1)
    model.fit(TRAIN)
    assert (model.num_users, model.num_items) == (NUM_USERS, NUM_ITEMS)

    assert 0.0 <= model.predict(0, 1) <= 1.0
    # Out-of-range indices cannot be embedded, so they are refused, not wrapped.
    assert model.predict(-1, 0) == 0.0
    assert model.predict(0, 10_000) == 0.0
    _assert_recommend_contract(model)

    path = os.path.join(tmp_path, "neumf.pt")
    model.save(path)
    reloaded = NeuMF(embedding_dim=8, mlp_dims=(16, 8))
    reloaded.load(path)
    assert (reloaded.num_users, reloaded.num_items) == (NUM_USERS, NUM_ITEMS)
    assert reloaded.recommend(0, top_k=3) == model.recommend(0, top_k=3)


# --- LightGCN ----------------------------------------------------------------


def test_lightgcn_fit_predict_recommend_and_roundtrip(tmp_path):
    from models.lightgcn import LightGCN

    model = LightGCN(embedding_dim=8, num_layers=2, epochs=2, batch_size=4)
    model.fit(TRAIN)
    assert (model.num_users, model.num_items) == (NUM_USERS, NUM_ITEMS)
    # The layer-averaged embeddings are the persisted artefact.
    assert model.user_emb.shape == (NUM_USERS, 8)
    assert model.item_emb.shape == (NUM_ITEMS, 8)

    assert np.isfinite(model.predict(0, 1))
    assert model.predict(-1, 0) == 0.0
    _assert_recommend_contract(model)

    path = os.path.join(tmp_path, "lightgcn.pkl")
    model.save(path)
    reloaded = LightGCN()
    reloaded.load(path)
    assert (reloaded.num_users, reloaded.num_items) == (NUM_USERS, NUM_ITEMS)
    np.testing.assert_allclose(reloaded.user_emb, model.user_emb)


# --- MultiModalNCF -----------------------------------------------------------


@pytest.fixture
def tiny_content_features(tmp_path, monkeypatch):
    """Point MultiModalNCF at tiny features instead of the shipped (large) arrays."""
    from models import multimodal_ncf as mm

    monkeypatch.setattr(mm, "PROCESSED_DIR", str(tmp_path))
    np.save(os.path.join(tmp_path, "text_embeddings.npy"), np.zeros((NUM_ITEMS, 384), np.float32))
    np.save(os.path.join(tmp_path, "image_embeddings.npy"), np.zeros((NUM_ITEMS, 2048), np.float32))
    genre = np.zeros((NUM_ITEMS, 18), np.float32)
    genre[:, 0] = 1.0
    np.save(os.path.join(tmp_path, "genre_vectors.npy"), genre)
    return genre


def test_multimodal_fit_recommend_cold_and_roundtrip(tmp_path, tiny_content_features):
    from models.multimodal_ncf import MultiModalNCF

    genre = tiny_content_features
    model = MultiModalNCF(embedding_dim=8, mlp_dims=(16, 8), epochs=1, batch_size=4, num_neg=1)
    model.fit(TRAIN)
    assert (model.num_users, model.num_items) == (NUM_USERS, NUM_ITEMS)
    assert model.text_emb.shape == (NUM_ITEMS, 384)

    assert 0.0 <= model.predict(0, 1) <= 1.0
    _assert_recommend_contract(model)

    # Cold path: a brand-new item is scored from its raw content alone.
    cold = model.predict_cold(0, np.zeros(384, np.float32), np.zeros(2048, np.float32), genre[0])
    assert 0.0 <= cold <= 1.0

    path = os.path.join(tmp_path, "multimodal.pt")
    model.save(path)
    reloaded = MultiModalNCF(embedding_dim=8, mlp_dims=(16, 8))
    reloaded.load(path)
    assert (reloaded.num_users, reloaded.num_items) == (NUM_USERS, NUM_ITEMS)
    assert reloaded.text_emb.shape == (NUM_ITEMS, 384)
