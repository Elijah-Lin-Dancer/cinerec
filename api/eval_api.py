"""Evaluation results endpoints.

These endpoints only ever return *measured* results. When an artefact is missing
they say so explicitly (``available: false``) instead of inventing plausible
numbers — a fabricated metric is indistinguishable from a real one once it is
rendered, so the honest thing is to report the absence.
"""
import os
import json
from fastapi import APIRouter

router = APIRouter()

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _load(filename):
    """Return the parsed JSON artefact, or a structured 'unavailable' marker."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return {"available": False, "note": f"{filename} not found — run the training/evaluation pipeline."}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload.setdefault("available", True)
    return payload


@router.get("/results")
async def get_eval_results():
    """Offline evaluation results for all models (measured on MovieLens 100K)."""
    return _load("eval_results.json")


@router.get("/ablation")
async def get_ablation_results():
    """Ablation study results for MultiModalNCF (measured, not estimated)."""
    return _load("ablation_results.json")
