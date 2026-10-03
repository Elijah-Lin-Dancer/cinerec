"""
CineRec FastAPI Application — Main entry point.
Serves the REST API and static frontend files.
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

async def get_optional_user(user_id: int = Query(None)):
    return user_id

app = FastAPI(
    title="CineRec — Multi-Modal Movie Recommendation System",
    description="A 5-level algorithm recommendation system with explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000"],
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
from db.database import init_db
init_db()

# Serve frontend static files (built-in path traversal protection via StaticFiles)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Serve static assets (css, js, images, fonts) — sub-mounted so API routes work
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="frontend-assets")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="frontend-css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="frontend-js")


# Catch-all: serve index.html or a specific frontend file for any non-API route
@app.get("/{path:path}")
async def serve_frontend(path: str):
    """Serve frontend files; fallback to index.html for SPA routing."""
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/")
async def serve_index():
    """Serve the SPA entry point."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
