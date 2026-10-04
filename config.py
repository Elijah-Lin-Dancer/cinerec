"""Central runtime configuration.

One codebase, two deployment modes selected by the ``APP_MODE`` environment
variable (see ``docs/adr/0001-dual-mode-deployment.md``):

- ``full``  — the ML stack (torch) is installed and trained artefacts are present;
  recommendations come from live model inference.
- ``lite``  — no torch; recommendations are served from a precomputed
  ``recs_cache.json`` so the app can run on a small, always-on free tier.

Everything that differs between the two modes is expressed here, so the rest of
the code reads a flag instead of re-deriving it from the environment.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DB_PATH = os.path.join(BASE_DIR, "db", "cinerec.db")

_RAW_MODE = os.environ.get("APP_MODE", "full").strip().lower()
APP_MODE = _RAW_MODE if _RAW_MODE in ("full", "lite") else "full"

#: When True the recommendation service answers from the precomputed cache.
USE_PRECOMPUTED = APP_MODE == "lite"

#: Depth of the precomputed cache (must be >= the largest ``top_k`` we serve).
RECS_CACHE_TOP_K = 50
RECS_CACHE_PATH = os.path.join(PROCESSED_DIR, "recs_cache.json")


def _int_env(name, default):
    """Read a positive int from the environment, falling back on bad input."""
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


#: LRU size for full-mode inference results (keyed per algorithm × user × exclusions).
MODEL_RECS_CACHE_SIZE = _int_env("RECS_LRU_SIZE", 512)
