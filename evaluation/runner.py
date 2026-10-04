"""
Offline Evaluation Runner — Train and evaluate all 5 models on MovieLens 100K.
Uses a leave-last-5-out split: each user's 5 most recent interactions form the
test set, the rest the training set.
Evaluates HR@K, NDCG@K, Recall@K at K={5, 10, 20}.
Saves results to data/processed/eval_results.json.
"""
import os
import sys
import json
import time
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.user_cf import UserCF
from models.item_cf import ItemCF
from models.svd_als import SVDALS
from models.neumf import NeuMF
from models.multimodal_ncf import MultiModalNCF
from evaluation.metrics import evaluate_model

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MODEL_FILES = {
    "UserCF": "model_usercf.pkl",
    "ItemCF": "model_itemcf.pkl",
    "SVD": "model_svd.pkl",
    "NeuMF": "model_neumf.pt",
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


def run_evaluation():
    """Train all models and evaluate."""
    train_data, test_pairs, train_by_user, full_df = load_data()
    k_values = [5, 10, 20]
    all_results = {}

    models_to_train = [
        ("UserCF", lambda: UserCF(k=50)),
        ("ItemCF", lambda: ItemCF(k=50)),
        ("SVD", lambda: SVDALS(k=64)),
        ("NeuMF", lambda: NeuMF(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)),
        ("MultiModalNCF", lambda: MultiModalNCF(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)),
    ]

    for name, model_factory in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        model = model_factory()
        start_time = time.time()

        try:
            model.fit(train_data)
            train_time = time.time() - start_time

            # Persist the trained artifact so a fresh clone can serve it.
            save_path = os.path.join(RESULTS_DIR, MODEL_FILES[name])
            model.save(save_path)
            print(f"Saved {name} → {save_path}")

            print(f"Evaluating {name}...")
            results = evaluate_model(model, test_pairs, train_by_user, k_values=k_values)
            results["train_time"] = round(train_time, 2)

            # Print results
            print(f"\n{name} Results:")
            for metric, value in sorted(results.items()):
                if metric != "train_time":
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}s")

            all_results[name] = results

        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Save results
    results_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
