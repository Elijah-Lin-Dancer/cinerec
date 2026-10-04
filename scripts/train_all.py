"""
One-click script to train all models and run evaluation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.runner import run_evaluation

if __name__ == "__main__":
    print("=" * 60)
    print("CineRec — Training All Models & Running Evaluation")
    print("=" * 60)
    run_evaluation()
    print("\nDone! Check data/processed/eval_results.json for results.")
