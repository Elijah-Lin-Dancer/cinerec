"""
Ablation study — what does each content modality actually contribute?

Trains the Multi-Modal NCF five times, each time blanking a different subset of
the content features (text / image / genre) at the input, and evaluates every
variant on the *same* leave-last-5-out split. Because the split, the architecture
and the training budget are held fixed, the delta between variants is attributable
to the modality that was removed.

Output
------
- single seed (default) → ``data/processed/ablation_results.json`` (flat metrics,
  consumed by evaluation/visualize.py)
- several seeds → ``data/processed/ablation_multiseed.json`` with ``mean``/``std``
  per variant, so the deltas between variants can be read against seed noise.

Run several seeds with ``make ablation-seeds``.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from evaluation.runner import load_data, RESULTS_DIR
from evaluation.metrics import evaluate_model
from models.multimodal_ncf import MultiModalNCF

# Same budget as the main run so the ablation is comparable to eval_results.json.
TRAIN_KWARGS = dict(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)

#: Default seed for the single-seed run — matches the main evaluation's seed 42.
DEFAULT_SEED = 42

VARIANTS = [
    ("Full (Text+Image+Genre)", []),
    ("w/o Text", ["text"]),
    ("w/o Image", ["image"]),
    ("w/o Genre", ["genre"]),
    ("Behavior only (no content)", ["text", "image", "genre"]),
]

METRIC_KEYS = ("HR@5", "NDCG@5", "Recall@5", "HR@10", "NDCG@10", "Recall@10",
               "HR@20", "NDCG@20", "Recall@20")


def _aggregate(per_seed, seed_list):
    """Mean/std per metric across seeds, or the raw metrics for a single seed."""
    agg = dict(per_seed[0])
    agg["train_time"] = per_seed[0]["train_time"]
    if len(seed_list) > 1:
        for metric in METRIC_KEYS:
            vals = [r[metric] for r in per_seed]
            agg[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        agg["seed_list"] = seed_list
        agg["n_seeds"] = len(seed_list)
    return agg


def difficulty_report(results, seed_list):
    """Report the full-content HR@10 std as the scale for reading the deltas."""
    if len(seed_list) == 1:
        return
    full = results["Full (Text+Image+Genre)"]["HR@10"]
    print(f"\nSeed noise (Full HR@10 std) = {full['std']:.4f}; "
          "deltas smaller than this are not distinguishable from seed variation.")


def main(seeds=None):
    seed_list = list(seeds) if seeds else [DEFAULT_SEED]
    train_data, test_pairs, train_by_user, _ = load_data()
    results = {}

    for label, disabled in VARIANTS:
        print(f"\n{'=' * 60}\nAblation: {label}  (seeds={seed_list})\n{'=' * 60}")
        per_seed = []
        for seed in seed_list:
            model = MultiModalNCF(disabled_features=disabled, seed=seed, **TRAIN_KWARGS)
            start = time.time()
            model.fit(train_data)
            train_time = time.time() - start
            metrics = evaluate_model(model, test_pairs, train_by_user, k_values=(5, 10, 20))
            metrics["train_time"] = round(train_time, 2)
            per_seed.append(metrics)

        agg = _aggregate(per_seed, seed_list)
        agg["disabled_features"] = sorted(disabled)
        results[label] = agg

        hr = agg["HR@10"]
        shown = f"{hr['mean']:.4f} ± {hr['std']:.4f}" if isinstance(hr, dict) else f"{hr:.4f}"
        print(f"{label}: HR@10={shown}, train_time={agg['train_time']}s")

    difficulty_report(results, seed_list)

    filename = "ablation_multiseed.json" if len(seed_list) > 1 else "ablation_results.json"
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultiModalNCF content-modality ablation study.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Training seeds; more than one writes ablation_multiseed.json.")
    main(seeds=parser.parse_args().seeds)
