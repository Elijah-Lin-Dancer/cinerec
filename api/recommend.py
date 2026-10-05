"""Recommendation endpoints.

Two runtime modes (see ``config.APP_MODE``):
- ``full``: models are loaded from disk and run live inference.
- ``lite``: no torch; results come from a precomputed ``recs_cache.json``.

In both modes the endpoint never fabricates results. Unknown algorithms,
unavailable models and empty result sets are reported explicitly, and
popularity fallbacks (for users outside the training set) are labelled as such.
"""
import os
import json
import logging
from functools import lru_cache
import numpy as np
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from db.database import get_connection
from api.auth import resolve_user
from config import USE_PRECOMPUTED, RECS_CACHE_PATH, MODEL_RECS_CACHE_SIZE
from models.registry import ALGORITHMS, AlgorithmUnavailable, load_model

router = APIRouter()

#: Candidate pool pulled from a model per cached call. Comfortably above the
#: largest servable ``top_k`` so exclusion + slicing still fills a full page.
_MODEL_CANDIDATE_POOL = 100

NOTE_EMPTY = "No recommendations are available for this user right now."
NOTE_FALLBACK = (
    "This account is outside the model's training set, so popularity-based "
    "recommendations are shown instead of personalized ones."
)

# Lazy-loaded model cache (full mode) and precomputed cache (lite mode)
_models_cache = {}
_recs_cache = None
_explainer = None


def _get_model(name):
    """Return a cached model, loading it on first use."""
    if name in _models_cache:
        return _models_cache[name]
    model = load_model(name)
    if model is not None:
        _models_cache[name] = model
    return model


@lru_cache(maxsize=MODEL_RECS_CACHE_SIZE)
def _cached_model_recs(algorithm, user_id, exclude_key):
    """Full-mode inference behind an LRU cache.

    Keyed by ``(algorithm, user, excluded items)``: a user rating a new film
    changes the exclusion set and so naturally invalidates their entry, while a
    repeated request for the same view is served from memory. Returns a tuple so
    the cached value is immutable.
    """
    model = _get_model(algorithm)
    recs = model.recommend(user_id, top_k=_MODEL_CANDIDATE_POOL, exclude_items=set(exclude_key))
    return tuple((int(i), float(s)) for i, s in recs)


def inference_cache_info():
    """Expose LRU statistics for the ``/api/metrics`` endpoint."""
    info = _cached_model_recs.cache_info()
    return {
        "currsize": info.currsize,
        "maxsize": info.maxsize,
        "hits": info.hits,
        "misses": info.misses,
    }


def _get_recs_cache():
    """Load (once) the precomputed recommendation cache used by lite mode."""
    global _recs_cache
    if _recs_cache is None:
        if os.path.exists(RECS_CACHE_PATH):
            with open(RECS_CACHE_PATH, encoding="utf-8") as f:
                _recs_cache = json.load(f)
        else:
            _recs_cache = {}
    return _recs_cache


def get_explainer():
    """Lazy-load the recommender explainer."""
    global _explainer
    if _explainer is None:
        from models.explain import RecommenderExplainer
        _explainer = RecommenderExplainer()
        _explainer.load_data()
        conn = get_connection()
        rows = conn.execute("SELECT user_id, movie_id, rating FROM ratings").fetchall()
        conn.close()
        train_data = {
            'user_id': [r['user_id'] for r in rows],
            'item_id': [r['movie_id'] for r in rows],
            'rating': [r['rating'] for r in rows],
        }
        _explainer.load_user_ratings(train_data)
    return _explainer


def _fetch_movies(conn, item_ids):
    """Fetch the requested movie rows in a single query, keyed by id.

    Replaces one ``SELECT`` per recommendation (an N+1 that scaled with
    ``top_k``) with a single ``IN`` query. Order is preserved by the caller
    through the returned mapping. The placeholder count derives from ``len``,
    never from user text, so the statement stays parameterised.
    """
    ids = [int(i) for i in dict.fromkeys(item_ids)]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, title, genres, poster_url, release_year "
        f"FROM movies WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(r["id"]): r for r in rows}


def _popular_items(conn, exclude, top_k):
    """Popularity fallback for users outside the model's training set.

    Returns ``(movie_id, average_rating)`` pairs for well-rated, frequently
    rated movies. Scores are average ratings — deliberately *not* dressed up as
    personalized model scores.
    """
    rows = conn.execute(
        """
        SELECT movie_id, AVG(rating) AS avg_rating, COUNT(*) AS n
        FROM ratings
        GROUP BY movie_id
        HAVING n >= 20
        ORDER BY avg_rating DESC, n DESC
        LIMIT ?
        """,
        (top_k + len(exclude),),
    ).fetchall()
    out = []
    for r in rows:
        if r["movie_id"] in exclude:
            continue
        out.append((int(r["movie_id"]), float(r["avg_rating"])))
        if len(out) >= top_k:
            break
    return out


def _recs_from_cache(conn, algorithm, user_id, exclude, top_k):
    """Serve one user's recommendations from the precomputed cache (lite mode).

    Returns ``(recs, is_fallback)``. Raises 503 when the algorithm was never
    precomputed, so the caller gets a clear message instead of an empty page.
    """
    cache = _get_recs_cache()
    if algorithm not in cache:
        available = ", ".join(k for k in cache if not k.startswith("_")) or "none"
        raise HTTPException(
            503,
            f"'{algorithm}' is not available in this precomputed deployment. "
            f"Available algorithms: {available}.",
        )

    cached = cache[algorithm].get(str(user_id))
    if cached is not None:
        recs = [(int(i), float(s)) for i, s in cached if int(i) not in exclude][:top_k]
        if recs:
            return recs, False

    recs = _popular_items(conn, exclude, top_k)
    return recs, bool(recs)


@router.get("")
async def get_recommendations(
    user_id: int = Query(None, description="User ID (must match the session token)"),
    algorithm: str = Query("SVD", description="Algorithm name"),
    top_k: int = Query(10, ge=1, le=50),
    current_user: int = Depends(resolve_user),
):
    """Get personalized movie recommendations.

    Requires a session token; a caller may only read their own recommendations.
    """
    if algorithm not in ALGORITHMS:
        raise HTTPException(400, f"Unknown algorithm '{algorithm}'")
    if user_id is not None and user_id != current_user:
        raise HTTPException(403, "You may only request your own recommendations")
    user_id = current_user

    conn = get_connection()
    try:
        rated_rows = conn.execute(
            "SELECT movie_id FROM ratings WHERE user_id = ?", (user_id,)
        ).fetchall()
        exclude = {r["movie_id"] for r in rated_rows}

        if USE_PRECOMPUTED:
            recs, is_fallback = _recs_from_cache(conn, algorithm, user_id, exclude, top_k)
        else:
            try:
                recs = list(_cached_model_recs(algorithm, user_id, tuple(sorted(exclude))))[:top_k]
            except AlgorithmUnavailable as e:
                raise HTTPException(503, str(e))
            except Exception:
                logging.exception("Model inference failed")
                raise HTTPException(500, f"Inference failed for algorithm '{algorithm}'")

            # Cold-start / out-of-range users: fall back to popularity, clearly labelled.
            is_fallback = False
            if not recs:
                recs = _popular_items(conn, exclude, top_k)
                is_fallback = bool(recs)

        page = recs[:top_k]
        movies_by_id = _fetch_movies(conn, [item_id for item_id, _ in page])

        recommendations = []
        for item_id, score in page:
            movie = movies_by_id.get(int(item_id))
            if movie:
                recommendations.append({
                    "item_id": int(movie["id"]),
                    "title": str(movie["title"]),
                    "genres": str(movie["genres"] or ""),
                    "poster_url": str(movie["poster_url"] or ""),
                    "release_year": int(movie["release_year"]) if movie["release_year"] else None,
                    "score": round(float(score), 4),
                    "algorithm": str(algorithm)
                })
    finally:
        conn.close()

    # Explanations are only attached to genuine, personalized model output —
    # never to popularity fallbacks (that would imply personalization).
    if recommendations and not is_fallback:
        try:
            explainer = get_explainer()
            for rec in recommendations:
                reasons = explainer.explain(user_id, rec["item_id"])
                rec["reasons"] = reasons.get("reasons", [])
        except Exception:
            logging.exception("Explainer failed")
            for rec in recommendations:
                rec["reasons"] = []

    if not recommendations:
        note = NOTE_EMPTY
    elif is_fallback:
        note = NOTE_FALLBACK
    else:
        note = ""

    def _json_default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    return JSONResponse(content=json.loads(json.dumps({
        "user_id": user_id,
        "algorithm": algorithm,
        "recommendations": recommendations,
        "total": len(recommendations),
        "fallback": is_fallback,
        "note": note,
    }, default=_json_default)))


@router.get("/{movie_id}/explain")
async def explain_recommendation(
    movie_id: int,
    user_id: int = Query(None, description="User ID (must match the session token)"),
    current_user: int = Depends(resolve_user),
):
    """Get explanation for why a movie was recommended."""
    if user_id is not None and user_id != current_user:
        raise HTTPException(403, "You may only request your own explanations")
    try:
        explainer = get_explainer()
        return explainer.explain(current_user, movie_id)
    except Exception:
        logging.exception("Explain recommendation failed")
        raise HTTPException(500, "Internal server error")
