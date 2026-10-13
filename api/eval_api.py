"""Evaluation results endpoints.

These endpoints only ever return *measured* results. When an artefact is missing
they say so explicitly (``available: false``) instead of inventing plausible
numbers — a fabricated metric is indistinguishable from a real one once it is
rendered, so the honest thing is to report the absence.

Multiseed artefacts (``*_multiseed.json``, mean ± std over seeds 42–46) are
preferred; the single-seed files remain as a fallback so the endpoints keep
working on older checkouts. The cross-dataset endpoint pairs the MovieLens
artefacts with the committed Amazon Video_Games ones (RQ5, ADR 0005).
"""
import os
import json
from fastapi import APIRouter

from config import PROCESSED_DIR, SECOND_PROCESSED_DIR  # processed dirs (config is the single source)

router = APIRouter()


def _load(filename, directory=None):
    """Return the parsed JSON artefact, or a structured 'unavailable' marker.

    A missing *or* unreadable/corrupt artefact is reported as unavailable rather
    than surfacing a 500 — the endpoint's job is to be honest about what exists.
    """
    path = os.path.join(directory or PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return {"available": False, "note": f"{filename} not found — run the training/evaluation pipeline."}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"available": False, "note": f"{filename} could not be read ({e})."}
    if isinstance(payload, dict):
        payload.setdefault("available", True)
    return payload


def _load_multiseed(name, directory=None):
    """``name_multiseed.json`` preferred, ``name.json`` as the single-seed fallback.

    Adds a ``multiseed: bool`` marker so clients can tell the mean ± std payload
    from the flat one without guessing from the value shapes.
    """
    ms = _load(f"{name}_multiseed.json", directory)
    if ms.get("available"):
        ms["multiseed"] = True
        return ms
    flat = _load(f"{name}.json", directory)
    flat["multiseed"] = False
    return flat


@router.get("/results")
async def get_eval_results():
    """Offline evaluation results for all models (measured on MovieLens 100K)."""
    return _load_multiseed("eval")


@router.get("/ablation")
async def get_ablation_results():
    """Ablation study results for MultiModalNCF (measured, not estimated)."""
    return _load_multiseed("ablation")


@router.get("/coldstart")
async def get_coldstart_results():
    """Cold-start study: content-aware ranking of items with no collaborative history."""
    return _load_multiseed("coldstart")


@router.get("/cross")
async def get_cross_dataset_results():
    """RQ5 cross-dataset evidence: the same protocol re-run on Amazon Video_Games.

    Returns both domains' multiseed artefacts side by side. ``amazon.available``
    is false on checkouts that predate the second-dataset study.
    """
    return {
        "available": True,
        "movielens": {
            "eval": _load_multiseed("eval"),
            "ablation": _load_multiseed("ablation"),
            "coldstart": _load_multiseed("coldstart"),
        },
        "amazon": {
            "eval": _load_multiseed("eval", SECOND_PROCESSED_DIR),
            "ablation": _load_multiseed("ablation", SECOND_PROCESSED_DIR),
            "coldstart": _load_multiseed("coldstart", SECOND_PROCESSED_DIR),
        },
    }
