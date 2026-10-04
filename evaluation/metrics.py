"""
Evaluation metrics for recommender systems.
Supports HR@K, NDCG@K, and Recall@K.
"""
import numpy as np


def hit_rate_at_k(recommended, relevant, k):
    """
    Hit Rate @ K: 1 if any relevant item appears in top-k recommendations.

    Args:
        recommended: list of recommended item IDs
        relevant: set of relevant (ground truth) item IDs
        k: cutoff rank
    """
    if not relevant:
        return 0.0
    return float(len(set(recommended[:k]) & set(relevant)) > 0)


def ndcg_at_k(recommended, relevant, k):
    """
    Normalized Discounted Cumulative Gain @ K.

    Args:
        recommended: list of recommended item IDs
        relevant: set of relevant (ground truth) item IDs
        k: cutoff rank
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    """
    Recall @ K: fraction of relevant items found in top-k.

    Args:
        recommended: list of recommended item IDs
        relevant: set of relevant (ground truth) item IDs
        k: cutoff rank
    """
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def evaluate_model(model, test_pairs, train_by_user=None, k_values=(5, 10, 20)):
    """
    Evaluate a recommender model on test data.

    Args:
        model: Recommender instance with recommend_ids() method
        test_pairs: list of (user_id, set_of_relevant_items)
        train_by_user: optional {user_id: set(train item ids)}. When given, every
            model is evaluated against the *same* candidate set (all items minus
            the user's training items), removing the protocol inconsistency where
            some models could re-recommend already-seen items.
        k_values: list of K values to evaluate at

    Returns:
        dict: {metric_name: mean_value}
    """
    results = {}
    for k in k_values:
        results[f"HR@{k}"] = []
        results[f"NDCG@{k}"] = []
        results[f"Recall@{k}"] = []

    train_by_user = train_by_user or {}
    max_k = max(k_values)

    for user_id, relevant in test_pairs:
        if not relevant:
            continue
        exclude = train_by_user.get(user_id, set())
        recs = model.recommend_ids(user_id, top_k=max_k, exclude_items=exclude)
        for k in k_values:
            results[f"HR@{k}"].append(hit_rate_at_k(recs, relevant, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recs, relevant, k))
            results[f"Recall@{k}"].append(recall_at_k(recs, relevant, k))

    return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in results.items()}
