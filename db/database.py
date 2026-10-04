import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "cinerec.db")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RATINGS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "ratings.csv")
MOVIES_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "movies_enriched.json")

def get_connection():
    """Get a SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create the schema if missing (idempotent, safe on a fresh clone)."""
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY, title TEXT, genres TEXT,
            overview TEXT DEFAULT '', poster_url TEXT DEFAULT '',
            tmdb_id TEXT DEFAULT '', release_year INTEGER
        );
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER, movie_id INTEGER, rating REAL,
            timestamp INTEGER, PRIMARY KEY (user_id, movie_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ratings_user_movie
            ON ratings(user_id, movie_id);
        """
    )
    conn.commit()
    conn.close()


def seed_if_empty():
    """Populate the database from the committed seed files when it is empty.

    Platforms deploy by pulling the repository, so a fresh instance starts with
    an empty database. Without this the demo would have no movies, no ratings and
    no demo users to log in as. The operation is idempotent: it returns
    immediately once ratings exist.
    """
    conn = get_connection()
    try:
        if conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]:
            return
        if not os.path.exists(RATINGS_CSV):
            print("[seed] ratings seed not found — skipping (database stays empty)")
            return

        import pandas as pd

        df = pd.read_csv(RATINGS_CSV)
        conn.executemany(
            "INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)",
            [(int(u), f"user_{int(u)}", None) for u in df["user_id"].drop_duplicates()],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO movies (id, title) VALUES (?, ?)",
            [(int(m), f"Movie {int(m)}") for m in df["item_id"].drop_duplicates()],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            [(int(r.user_id), int(r.item_id), float(r.rating), int(r.timestamp)) for r in df.itertuples()],
        )

        if os.path.exists(MOVIES_JSON):
            with open(MOVIES_JSON, encoding="utf-8") as f:
                movies = json.load(f)
            conn.executemany(
                """UPDATE movies SET title=?, genres=?, overview=?, poster_url=?,
                   tmdb_id=?, release_year=? WHERE id=?""",
                [
                    (m.get("title", ""), m.get("genres", ""), m.get("overview", ""),
                     m.get("poster_url", ""), str(m.get("tmdb_id", "") or ""),
                     int(m["release_year"]) if m.get("release_year") else None,
                     int(m["id"]))
                    for m in movies
                ],
            )

        conn.commit()
        print(f"[seed] imported {len(df)} ratings, {df['user_id'].nunique()} users "
              f"and {df['item_id'].nunique()} movies from the seed files")
    finally:
        conn.close()


class DBConnection:
    """Context manager for database connections with auto commit/rollback."""

    def __enter__(self):
        self.conn = get_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()
        return False
