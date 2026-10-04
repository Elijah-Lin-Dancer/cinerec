"""
Precompute the recommendation cache used by ``APP_MODE=lite`` deployments.

``lite`` runs without torch: instead of live inference it replays a cache of the
top-N items for every user the models were trained on. This script fills that
cache for every algorithm in the ladder and writes it to
``data/processed/recs_cache.json``.

Users outside the training set (e.g. accounts registered on the live demo) are
not cached; the API answers them with a clearly-labelled popularity fallback.

Usage:
    python scripts/precompute.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RECS_CACHE_PATH, RECS_CACHE_TOP_K
from db.database import get_connection
from models.registry import ALGORITHMS, load_model, AlgorithmUnavailable


def user_history(conn):
    """Return {user_id: {rated item ids}} for every user in the database."""
    history = {}
    for row in conn.execute("SELECT user_id, movie_id FROM ratings"):
        history.setdefault(int(row["user_id"]), set()).add(int(row["movie_id"]))
    return history


def main():
    conn = get_connection()
    history = user_history(conn)
    conn.close()
    users = sorted(history)
    print(f"Precomputing top-{RECS_CACHE_TOP_K} for {len(users)} users × {len(ALGORITHMS)} algorithms")

    cache = {}
    for name in ALGORITHMS:
        try:
            model = load_model(name)
        except AlgorithmUnavailable as e:
            print(f"  ! {name}: {e}")
            continue

        print(f"  {name}: loading model...")
        started = time.time()
        per_user = {}
        for user_id in users:
            recs = model.recommend(
                user_id, top_k=RECS_CACHE_TOP_K, exclude_items=history[user_id]
            )
            per_user[str(user_id)] = [[int(i), round(float(s), 4)] for i, s in recs]
        cache[name] = per_user
        print(f"  {name}: {len(per_user)} users in {time.time() - started:.1f}s")

    with open(RECS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    size_mb = os.path.getsize(RECS_CACHE_PATH) / 1e6
    print(f"\nWrote {RECS_CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
