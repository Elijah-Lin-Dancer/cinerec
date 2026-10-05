"""Movie browsing and rating endpoints."""
import time

from fastapi import APIRouter, Query, HTTPException, Depends
from db.database import get_connection, DBConnection
from api.auth import resolve_user

router = APIRouter()


def _row_to_dict(row):
    """Convert sqlite3.Row to JSON-safe dict."""
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, (np.integer,)):
            d[key] = int(val)
        elif isinstance(val, (np.floating,)):
            d[key] = float(val)
        else:
            d[key] = val
    return d

import numpy as np


@router.get("")
async def list_movies(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: str = Query("", description="Search by title"),
    genre: str = Query("", description="Filter by genre"),
    year_from: int = Query(0, ge=0, description="Earliest release year (0 = no lower bound)"),
    year_to: int = Query(0, ge=0, description="Latest release year (0 = no upper bound)"),
    sort: str = Query("id", description="Sort field: id, title, year")
):
    with DBConnection() as conn:
        offset = (page - 1) * per_page

        query = "SELECT * FROM movies WHERE 1=1"
        params = []

        if search:
            query += " AND title LIKE ?"
            params.append(f"%{search}%")
        if genre:
            query += " AND genres LIKE ?"
            params.append(f"%{genre}%")
        if year_from:
            query += " AND release_year >= ?"
            params.append(year_from)
        if year_to:
            query += " AND release_year <= ?"
            params.append(year_to)

        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]

        # Sort and paginate
        valid_sorts = {"id": "id", "title": "title", "year": "release_year"}
        sort_col = valid_sorts.get(sort, "id")
        query += f" ORDER BY {sort_col} LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        movies = conn.execute(query, params).fetchall()

    return {
        "movies": [_row_to_dict(m) for m in movies],
        "total": int(total),
        "page": int(page),
        "per_page": int(per_page),
        "pages": int((total + per_page - 1) // per_page)
    }


@router.get("/genres")
async def list_genres():
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT genres FROM movies WHERE genres IS NOT NULL AND genres != ''").fetchall()
    conn.close()
    all_genres = set()
    for r in rows:
        if r["genres"]:
            for g in r["genres"].split("|"):
                g = g.strip()
                if g:
                    all_genres.add(g)
    return {"genres": sorted(all_genres)}


@router.get("/{movie_id}")
async def get_movie(movie_id: int):
    with DBConnection() as conn:
        movie = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
        if not movie:
            raise HTTPException(404, "Movie not found")

        # Get average rating
        rating_row = conn.execute(
            "SELECT AVG(rating) as avg_rating, COUNT(*) as count FROM ratings WHERE movie_id = ?",
            (movie_id,)
        ).fetchone()

    result = _row_to_dict(movie)
    result["avg_rating"] = round(float(rating_row["avg_rating"]), 2) if rating_row["avg_rating"] else None
    result["rating_count"] = int(rating_row["count"])
    return result


@router.post("/{movie_id}/rate")
async def rate_movie(
    movie_id: int,
    user_id: int = Query(None, description="User ID (must match the session token)"),
    rating: float = Query(..., ge=1, le=5),
    current_user: int = Depends(resolve_user),
):
    """Submit or update a movie rating for the authenticated user."""
    if user_id is not None and user_id != current_user:
        raise HTTPException(403, "You may only rate as yourself")
    user_id = current_user
    conn = get_connection()
    try:
        # Reject ratings for movies that do not exist rather than silently
        # storing a dangling foreign key.
        if not conn.execute("SELECT 1 FROM movies WHERE id = ?", (movie_id,)).fetchone():
            raise HTTPException(404, "Movie not found")

        # Store the timestamp as an integer epoch, matching the schema comment
        # and the seed import (previously a TEXT datetime was written here,
        # giving the column two incompatible encodings).
        conn.execute(
            """
            INSERT INTO ratings (user_id, movie_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating, timestamp = excluded.timestamp
            """,
            (user_id, movie_id, rating, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()
    return {"message": f"Rating {rating} saved for movie {movie_id}", "user_id": user_id}
