"""Tests for evaluation metrics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k


def test_hit_rate_found():
    assert hit_rate_at_k([1, 2, 3, 4, 5], {3}, 5) == 1.0


def test_hit_rate_not_found():
    assert hit_rate_at_k([1, 2, 4, 5], {3}, 5) == 0.0


def test_hit_rate_empty_relevant():
    assert hit_rate_at_k([1, 2, 3], set(), 5) == 0.0


def test_ndcg_perfect():
    """When the single relevant item is at position 1, NDCG should be 1.0."""
    assert abs(ndcg_at_k([3, 1, 2], {3}, 3) - 1.0) < 1e-9


def test_ndcg_partial():
    score = ndcg_at_k([1, 2, 3], {3}, 3)
    assert 0 < score < 1.0


def test_ndcg_empty_relevant():
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0


def test_recall():
    assert abs(recall_at_k([1, 2, 3, 4, 5], {3, 6}, 5) - 0.5) < 1e-9


def test_recall_empty():
    assert recall_at_k([1, 2, 3], set(), 5) == 0.0


if __name__ == "__main__":
    test_hit_rate_found()
    test_hit_rate_not_found()
    test_hit_rate_empty_relevant()
    test_ndcg_perfect()
    test_ndcg_partial()
    test_ndcg_empty_relevant()
    test_recall()
    test_recall_empty()
    print("All 8 metric tests passed!")
