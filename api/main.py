"""
CineRec FastAPI Application — Main entry point.
Serves the REST API and static frontend files.
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="CineRec — Multi-Modal Movie Recommendation System",
    description="A 6-level algorithm recommendation system with explainability",
    version="1.0.0"
)


@app.get("/api/health", tags=["Health"])
async def health():
    """Liveness probe used by Docker/healthchecks and uptime monitors."""
    return {"status": "ok"}


@app.get("/api/metrics", tags=["Metrics"])
async def metrics():
    """Process-local throughput / latency snapshot plus inference-cache stats.

    Counters reset on restart and are per-replica; see ``api/metrics.py``.
    """
    from api.metrics import registry
    snapshot = registry.snapshot()
    try:
        from api.recommend import inference_cache_info
        snapshot["inference_cache"] = inference_cache_info()
    except Exception:  # pragma: no cover - defensive, cache info is optional
        snapshot["inference_cache"] = None
    return snapshot


# Request timing/observability (must be registered before the app starts).
from api.metrics import metrics_middleware  # noqa: E402

app.middleware("http")(metrics_middleware)

# CORS origins are configurable so the same image works locally and behind a
# real host; the localhost defaults keep `make serve` working out of the box.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://localhost:3000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from api.auth import router as auth_router
from api.movies import router as movies_router
from api.recommend import router as recommend_router
from api.eval_api import router as eval_router

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(movies_router, prefix="/api/movies", tags=["Movies"])
app.include_router(recommend_router, prefix="/api/recommend", tags=["Recommendations"])
app.include_router(eval_router, prefix="/api/eval", tags=["Evaluation"])

# One-time DB initialisations (idempotent)
from db.database import init_db, seed_if_empty
init_db()
seed_if_empty()

# Serve frontend static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


def _safe_frontend_path(path: str):
    """Resolve a requested path inside FRONTEND_DIR; return None if it escapes the root."""
    candidate = os.path.abspath(os.path.join(FRONTEND_DIR, path))
    if candidate == FRONTEND_DIR or not candidate.startswith(FRONTEND_DIR + os.sep):
        return None
    return candidate

# Serve static assets (css, js, images, fonts) — sub-mounted so API routes work
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="frontend-assets")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="frontend-css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="frontend-js")


# Serve the SPA entry point. Registered *before* the catch-all: otherwise
# `/{path:path}` matches "/" with an empty path, which the traversal guard
# rejects (empty path resolves to FRONTEND_DIR itself) and the root 404s.
@app.get("/")
async def serve_index():
    """Serve the SPA entry point."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Catch-all: serve index.html or a specific frontend file for any non-API route
@app.get("/{path:path}")
async def serve_frontend(path: str):
    """Serve frontend files; fallback to index.html for SPA routing.

    A path that escapes ``FRONTEND_DIR`` (e.g. ``../.env``) is rejected with 404
    instead of being silently rewritten to the SPA shell.
    """
    file_path = _safe_frontend_path(path)
    if file_path is None:
        raise HTTPException(404, "Not found")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
