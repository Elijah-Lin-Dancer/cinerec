"""Tests for the feature-engineering id alignment.

The content feature arrays are indexed by raw movie id (row ``i`` = movie ``i``),
so a model can index features directly with ``item_id``. This was historically
wrong (positional indexing), which silently mis-aligned every feature — these
tests pin the contract.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

import data.preprocess as preprocess  # noqa: E402


def test_encode_genres_is_id_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))

    item_ids = [1, 4]  # note: sparse, non-contiguous ids
    result = preprocess.encode_genres(["Action|Comedy", "Drama"], item_ids)

    assert result.shape == (5, preprocess.NUM_GENRES)
    # Row 0 (an unused id) must stay zero — proves we index by id, not position.
    assert np.count_nonzero(result[0]) == 0
    assert result[1, preprocess.GENRE_LIST.index("Action")] == 1.0
    assert result[1, preprocess.GENRE_LIST.index("Comedy")] == 1.0
    assert result[4, preprocess.GENRE_LIST.index("Drama")] == 1.0
    # And the artefact is actually written for downstream models.
    assert (tmp_path / "genre_vectors.npy").exists()


def test_encode_genres_ignores_unknown_and_empty():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        original = preprocess.PROCESSED_DIR
        preprocess.PROCESSED_DIR = d
        try:
            result = preprocess.encode_genres(["Nonsense", "", None], [1, 2, 3])
        finally:
            preprocess.PROCESSED_DIR = original
    assert np.count_nonzero(result) == 0
