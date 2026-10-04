"""
Ablation study — what does each content modality actually contribute?

Trains the Multi-Modal NCF five times, each time blanking a different subset of
the content features (text / image / genre) at the input, and evaluates every
variant on the *same* leave-last-5-out split. Because the split, the architecture
and the training budget are held fixed, the delta between variants is attributable
to the modality that was removed.

Output: data/processed/ablation_results.json (consumed by evaluation/visualize.py).

This is a small self-contained study on MovieLens-100K; it is not multi-seed and
makes no significance claims. See docs/ROADMAP.md Phase 4.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from evaluation.runner import load_data, RESULTS_DIR
from evaluation.metrics import evaluate_model
from models.multimodal_ncf import MultiModalNCF

# Same budget as the main run so the ablation is comparable to eval_results.json.
TRAIN_KWARGS = dict(embedding_dim=32, mlp_dims=(128, 64, 32), epochs=10, batch_size=512)

VARIANTS = [
    ("Full (Text+Image+Genre)", []),
    ("w/o Text", ["text"]),
    ("w/o Image", ["image"]),
    ("w/o Genre", ["genre"]),
    ("Behavior only (no content)", ["text", "image", "genre"]),
]


def main():
    train_data, test_pairs, train_by_user, _ = load_data()
    results = {}

    for label, disabled in VARIANTS:
        print(f"\n{'=' * 60}\nAblation: {label}\n{'=' * 60}")
        # Fixed seed per variant so differences are not seed noise.
        np.random.seed(42)
        torch.manual_seed(42)

        model = MultiModalNCF(disabled_features=disabled, **TRAIN_KWARGS)
        start = time.time()
        model.fit(train_data)
        train_time = time.time() - start

        metrics = evaluate_model(model, test_pairs, train_by_user, k_values=(5, 10, 20))
        metrics["train_time"] = round(train_time, 2)
        metrics["disabled_features"] = sorted(disabled)
        results[label] = metrics

        print(f"{label}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(metrics.items())
            if k.startswith(("HR", "NDCG", "Recall"))
        ))

    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved → {out_path}")


if __name__ == "__main__":
    main()
