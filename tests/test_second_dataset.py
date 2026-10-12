"""Tests for the second-dataset machinery (CINEREC_DATASET).

Pins three contracts the cross-dataset study depends on:

1. ``config._dataset_dirs`` — a non-default dataset resolves to an isolated
   ``data/datasets/<name>`` subtree so it can never overwrite the served
   MovieLens artefacts.
2. ``amazon_dataset._apply_slice`` — the heavy-user slice is deterministic,
   enforces its user/item thresholds, and its thresholds are decoupled (a
   shared threshold would death-spiral sparse Amazon data to zero rows).
3. ``preprocess._load_vocab`` — a dataset may define its own 18-way category
   vocabulary, and a mismatched length fails loudly instead of silently
   mis-shaping the genre array.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import data.preprocess as preprocess  # noqa: E402
from config import _dataset_dirs  # noqa: E402
from data.amazon_dataset import _apply_slice  # noqa: E402


# --- config._dataset_dirs ---------------------------------------------------

def test_default_dataset_keeps_historical_paths():
    raw, processed = _dataset_dirs("ml-100k")
    assert raw.endswith(os.path.join("data", "raw"))
    assert processed.endswith(os.path.join("data", "processed"))


def test_second_dataset_is_isolated_subtree():
    raw, processed = _dataset_dirs("amazon-video-games")
    assert os.path.join("datasets", "amazon-video-games") in raw
    assert os.path.join("datasets", "amazon-video-games") in processed
    assert raw.endswith("raw") and processed.endswith("processed")
    # Neither path may resolve inside the served MovieLens tree.
    assert not processed.endswith(os.path.join("data", "processed"))


# --- amazon_dataset._apply_slice ---------------------------------------------

def _synthetic_ratings():
    """4 heavy users (20 ratings each over 10 shared items) + 4 light users (3 each)."""
    rows = []
    for u in range(4):
        for step in range(20):
            rows.append((f"u{u}", f"i{step % 10}", 4.0, step))
    for u in range(4, 8):
        for step in range(3):
            rows.append((f"u{u}", f"j{step}", 3.0, step))
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])


def test_slice_enforces_user_and_item_thresholds():
    out = _apply_slice(_synthetic_ratings(), max_users=0, min_user=20, min_item=5)
    assert len(out) > 0
    assert out.groupby("user_id").size().min() >= 20
    assert out.groupby("parent_asin").size().min() >= 5
    # Light users are gone entirely.
    assert not out["user_id"].str.startswith("u4").any()
    assert set(out["user_id"].unique()) == {"u0", "u1", "u2", "u3"}


def test_slice_is_deterministic():
    a = _apply_slice(_synthetic_ratings(), max_users=0, min_user=20, min_item=5)
    b = _apply_slice(_synthetic_ratings(), max_users=0, min_user=20, min_item=5)
    pd.testing.assert_frame_equal(a, b)


def test_slice_sampling_is_seeded_and_capped():
    kwargs = dict(min_user=20, min_item=5)
    a = _apply_slice(_synthetic_ratings(), max_users=2, **kwargs)
    b = _apply_slice(_synthetic_ratings(), max_users=2, **kwargs)
    pd.testing.assert_frame_equal(a, b)  # same seed → same sample
    assert a["user_id"].nunique() <= 2


def test_slice_thresholds_are_decoupled_not_death_spiralling():
    """A single shared user/item threshold empties this data; the split
    thresholds keep the heavy users. Guards the cascade against the
    collapse observed when both sides filter at the same bar."""
    shared = _apply_slice(_synthetic_ratings(), max_users=0, min_user=20, min_item=20)
    assert len(shared) == 0  # items never reach 20 → documented collapse
    split = _apply_slice(_synthetic_ratings(), max_users=0, min_user=20, min_item=5)
    assert len(split) > 0


# --- preprocess._load_vocab ---------------------------------------------------

def test_load_vocab_defaults_to_movieledgeres_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))
    assert preprocess._load_vocab() == preprocess.GENRE_LIST


def test_load_vocab_reads_dataset_vocabulary(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))
    vocab = [f"cat{i}" for i in range(preprocess.NUM_GENRES)]
    (tmp_path / "category_vocab.json").write_text(json.dumps(vocab))
    assert preprocess._load_vocab() == vocab


def test_load_vocab_rejects_wrong_length(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))
    (tmp_path / "category_vocab.json").write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError, match="content tower is fixed"):
        preprocess._load_vocab()
