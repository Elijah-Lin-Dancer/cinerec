"""
Offline Evaluation Runner — Train and evaluate every model on MovieLens 100K.
Uses a leave-last-5-out split: each user's 5 most recent interactions form the
test set, the rest the training set.
Evaluates HR@K, NDCG@K, Recall@K at K={5, 10, 20}.
Saves results to data/processed/eval_results.json.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.user_cf import UserCF
from models.item_cf import ItemCF
from models.svd_als import SVDALS
from models.bpr_mf import BPRMF
from models.neumf import NeuMF
from models.lightgcn import LightGCN
from models.vbpr import VBPR
from models.multimodal_ncf import MultiModalNCF
from evaluation.metrics import evaluate_model

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# BPR-MF and VBPR are the published SOTA baselines (Rendle 2009; He & McAuley
# 2016). They are evaluated here for comparison but are not part of the servable
# ladder in ``models/registry.py`` — the app only ships the six ladder models.
MODEL_FILES = {
    "UserCF": "model_usercf.pkl",
    "ItemCF": "model_itemcf.pkl",
    "SVD": "model_svd.pkl",
    "BPRMF": "model_bprmf.pkl",
    "NeuMF": "model_neumf.pt",
    "LightGCN": "model_lightgcn.pkl",
    "VBPR": "model_vbpr.pkl",
    "MultiModalNCF": "model_multimodalncf.pt",
}


def load_data(n_test=5):
    """Load MovieLens ratings and build a leave-last-N-out split.

    Each user's most recent ``n_test`` interactions form the test set; the rest
    form the training set. Using N>1 relevant items makes Recall@K distinct from
    HR@K (with a single relevant item they are mathematically identical).
    """
    csv_path = os.path.join(RAW_DIR, "ratings.csv")
    df = pd.read_csv(csv_path)

    # Rank interactions from the end per user: 0 = most recent.
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["_rev_rank"] = df.groupby("user_id").cumcount(ascending=False)

    test_df = df[df["_rev_rank"] < n_test].drop(columns=["_rev_rank"])
    train_df = df[df["_rev_rank"] >= n_test].drop(columns=["_rev_rank"])

    train_data = {
        "user_id": train_df["user_id"].values,
        "item_id": train_df["item_id"].values,
        "rating": train_df["rating"].values,
    }

    # Per-user training items, used to align the candidate set across models.
    train_by_user = {
        int(u): set(g["item_id"].tolist())
        for u, g in train_df.groupby("user_id")
    }

    # Test pairs: (user_id, {relevant_item_ids})
    test_pairs = [
        (int(u), set(g["item_id"].tolist()))
        for u, g in test_df.groupby("user_id")
    ]

    print(f"Data: {len(train_df)} train, {len(test_df)} test "
          f"({n_test} held-out items/user)")
    print(f"Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")

    return train_data, test_pairs, train_by_user, df


def _print_results(name, results):
    """Print one model's metrics; ``mean ± std`` when aggregated across seeds."""
    print(f"\n{name} Results:")
    if "error" in results:
        print(f"  error: {results['error']}")
        return
    for metric, value in sorted(results.items()):
        if metric in ("seed_list", "n_seeds"):
            continue
        if metric == "train_time":
            print(f"  {metric}: {value}s")
        elif isinstance(value, dict):
            print(f"  {metric}: {value['mean']:.4f} ± {value['std']:.4f}")
        else:
            print(f"  {metric}: {value:.4f}")


def run_evaluation(n_seeds=1, seeds=None):
    """Train all models and evaluate.

    With ``n_seeds=1`` (the default) each model is trained once with its default
    seed and the JSON contains the raw metrics, as before. With ``n_seeds>1`` the
    training seeds are varied and the JSON reports ``mean``/``std`` across the
    runs — a single-seed table makes no claim about seed noise, so numbers that
    are supposed to support a "best model" statement should come from the
    multi-seed path.
    """
    train_data, test_pairs, train_by_user, full_df = load_data()
    k_values = [5, 10, 20]
    all_results = {}

    models_to_train = [
        ("UserCF", lambda: UserCF(k=50)),
        ("ItemCF", lambda: ItemCF(k=50)),
        ("SVD", lambda: SVDALS(k=64)),
        ("BPRMF", lambda: BPRMF(embedding_dim=64, epochs=50)),
        ("NeuMF", lambda: NeuMF(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)),
        ("LightGCN", lambda: LightGCN(embedding_dim=64, num_layers=3, epochs=50, batch_size=4096)),
        ("VBPR", lambda: VBPR(embedding_dim=64, visual_dim=64, epochs=30)),
        ("MultiModalNCF", lambda: MultiModalNCF(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)),
    ]

    seed_list = list(seeds) if seeds is not None else list(range(1, n_seeds + 1))

    for name, model_factory in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        try:
            per_seed = []
            first_train_time = None
            for idx, seed in enumerate(seed_list):
                model = model_factory()
                if hasattr(model, "seed"):
                    model.seed = seed

                start_time = time.time()
                model.fit(train_data)
                train_time = time.time() - start_time

                if idx == 0:
                    # Persist the first seed's model, so the artefact shipped to a
                    # fresh clone is the one the reported metrics start from.
                    save_path = os.path.join(RESULTS_DIR, MODEL_FILES[name])
                    model.save(save_path)
                    print(f"Saved {name} → {save_path}")
                    first_train_time = train_time

                print(f"Evaluating {name} (seed={seed})...")
                per_seed.append(evaluate_model(model, test_pairs, train_by_user, k_values=k_values))

            results = dict(per_seed[0])
            results["train_time"] = round(first_train_time, 2)

            if len(seed_list) > 1:
                for metric in list(results):
                    if metric == "train_time":
                        continue
                    vals = [r[metric] for r in per_seed if metric in r]
                    results[metric] = {
                        "mean": round(sum(vals) / len(vals), 4),
                        "std": round(float(np.std(vals)), 4),
                    }
                results["seed_list"] = seed_list
                results["n_seeds"] = len(seed_list)

            all_results[name] = results

        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

        _print_results(name, all_results[name])

    # Save results. A single-seed run keeps writing eval_results.json (the
    # artefact the dashboard reads); a multi-seed run adds eval_multiseed.json
    # instead of overwriting it, so the served numbers stay raw metrics.
    if len(seed_list) > 1:
        results_path = os.path.join(RESULTS_DIR, "eval_multiseed.json")
    else:
        results_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
