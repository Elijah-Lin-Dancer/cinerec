"""
Cold-start experiment — can content features score items the model never saw?

Protocol
--------
1. Take the standard leave-last-5-out split, then pick the ``N_COLD_ITEMS`` items
   that are rarest in the training set and remove *all* of their training
   interactions. They become genuinely cold: no collaborative signal exists for
   them, so their learned embeddings are never updated.
2. Retrain two models on the reduced training set:
     - MultiModalNCF — scored through the content tower (``predict_cold``), which
       needs only the item's text/image/genre vectors and no trained embedding.
     - NeuMF — the content-blind control: cold items fall back on their untrained
       (random) embeddings, which is the honest "no content, no history" case.
3. Evaluate both on test interactions whose relevant item is cold. The candidate
   set is the *cold pool itself*: the task is to identify which unseen items a
   user would like, which is exactly what a content-aware model claims to do.
   Ranking cold items against the fully-trained warm catalogue would be an unfair
   test that no content-only path could pass, and would not isolate the claim.

   A ``Random`` baseline (fixed seed) is included as the reference floor.

Output: data/processed/coldstart_results.json — consumed by evaluation/visualize.py.
Small single-seed study on MovieLens-100K; no significance claims. See ROADMAP Phase 4.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from evaluation.runner import load_data, RESULTS_DIR
from evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from models.multimodal_ncf import MultiModalNCF
from models.neumf import NeuMF

N_COLD_ITEMS = 100
K_VALUES = (10, 20)
TRAIN_KWARGS = dict(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)


def choose_cold_items(train_by_user, test_pairs, n=N_COLD_ITEMS):
    """Rarest-in-training items that still appear in the test set."""
    counts = {}
    for items in train_by_user.values():
        for i in items:
            counts[i] = counts.get(i, 0) + 1

    test_items = set()
    for _, relevant in test_pairs:
        test_items |= relevant

    ranked = sorted(test_items, key=lambda i: (counts.get(i, 0), i))
    return set(ranked[:n])


def remove_items_from_train(train_data, cold_items):
    """Drop every training interaction touching a cold item."""
    mask = np.array([i not in cold_items for i in train_data["item_id"]])
    return {
        "user_id": train_data["user_id"][mask],
        "item_id": train_data["item_id"][mask],
        "rating": train_data["rating"][mask],
    }


def _cold_mm_scores(model, user_id, cold_idx):
    """Content-only score for every cold item (zero GMF item vector)."""
    net = model.net
    net.eval()
    with torch.no_grad():
        cu = torch.full((len(cold_idx),), user_id, dtype=torch.long).to(model.device)
        return net.forward_cold(
            cu,
            torch.FloatTensor(model.text_emb[cold_idx]).to(model.device),
            torch.FloatTensor(model.image_emb[cold_idx]).to(model.device),
            torch.FloatTensor(model.genre_vec[cold_idx]).to(model.device),
        ).cpu().numpy()


def _cold_neumf_scores(model, user_id, cold_idx):
    """NeuMF score for cold items: untrained embeddings — the content-blind case."""
    net = model.net
    net.eval()
    with torch.no_grad():
        cu = torch.full((len(cold_idx),), user_id, dtype=torch.long).to(model.device)
        ci = torch.LongTensor(cold_idx).to(model.device)
        return net(cu, ci).cpu().numpy()


def _rank(cold_idx, scores, top_k):
    """Rank the cold pool by score; return item ids best-first."""
    order = np.argsort(scores)[::-1][:top_k]
    return [int(cold_idx[i]) for i in order]


def evaluate_cold(rank_fn, pairs, cold_idx, max_k):
    sums = {f"{m}@{k}": [] for k in K_VALUES for m in ("HR", "NDCG", "Recall")}
    for user_id, relevant in pairs:
        recs = rank_fn(user_id, cold_idx, max_k)
        for k in K_VALUES:
            sums[f"HR@{k}"].append(hit_rate_at_k(recs, relevant, k))
            sums[f"NDCG@{k}"].append(ndcg_at_k(recs, relevant, k))
            sums[f"Recall@{k}"].append(recall_at_k(recs, relevant, k))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in sums.items()}


def main():
    train_data, test_pairs, train_by_user, _ = load_data()
    cold_items = choose_cold_items(train_by_user, test_pairs)
    cold_idx = np.array(sorted(cold_items), dtype=int)
    print(f"Cold items: {len(cold_items)} (removed from training)")

    reduced = remove_items_from_train(train_data, cold_items)

    # Only test interactions whose relevant item is cold.
    cold_pairs = []
    for user_id, relevant in test_pairs:
        rel_cold = relevant & cold_items
        if rel_cold:
            cold_pairs.append((user_id, rel_cold))
    print(f"Cold test pairs: {len(cold_pairs)}  (candidate pool = {len(cold_idx)} cold items)")

    results = {}
    max_k = max(K_VALUES)

    # --- MultiModalNCF (content tower) ---
    print("\nTraining MultiModalNCF on the reduced set...")
    np.random.seed(42)
    torch.manual_seed(42)
    mm = MultiModalNCF(**TRAIN_KWARGS)
    t = time.time()
    mm.fit(reduced)
    mm_time = time.time() - t

    def mm_rank(user_id, idx, top_k):
        return _rank(idx, _cold_mm_scores(mm, user_id, idx), top_k)

    results["MultiModalNCF (content)"] = evaluate_cold(mm_rank, cold_pairs, cold_idx, max_k)
    results["MultiModalNCF (content)"]["train_time"] = round(mm_time, 2)

    # --- NeuMF (content-blind control) ---
    print("\nTraining NeuMF on the reduced set...")
    np.random.seed(42)
    torch.manual_seed(42)
    ncf = NeuMF(**TRAIN_KWARGS)
    t = time.time()
    ncf.fit(reduced)
    ncf_time = time.time() - t

    def ncf_rank(user_id, idx, top_k):
        return _rank(idx, _cold_neumf_scores(ncf, user_id, idx), top_k)

    results["NeuMF (content-blind)"] = evaluate_cold(ncf_rank, cold_pairs, cold_idx, max_k)
    results["NeuMF (content-blind)"]["train_time"] = round(ncf_time, 2)

    # --- Random floor ---
    def random_rank(user_id, idx, top_k):
        rng = np.random.default_rng(user_id)
        return [int(i) for i in rng.permutation(idx)[:top_k]]

    results["Random"] = evaluate_cold(random_rank, cold_pairs, cold_idx, max_k)

    payload = {
        "n_cold_items": len(cold_items),
        "n_cold_test_pairs": len(cold_pairs),
        "candidate_pool": "cold items only",
        "metrics": results,
    }
    out_path = os.path.join(RESULTS_DIR, "coldstart_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nCold-start results:")
    for name, m in results.items():
        print(f"  {name}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(m.items()) if k.startswith(("HR", "NDCG", "Recall"))
        ))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
