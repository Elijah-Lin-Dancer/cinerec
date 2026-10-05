"""CLI helper to build the SQLite database from the CSV/JSON sources.

Deliberately consistent with the runtime path in :mod:`db.database`: it reuses
the shared schema and connection factory (Row factory, WAL, foreign keys) and
always releases the connection, rather than opening a second, differently
configured connection.
"""
import json
import os

import pandas as pd

from db.database import get_connection, init_db as create_schema


def init_db(ratings_csv=None, movies_csv=None):
    """Create the schema and optionally import data (idempotent)."""
    create_schema()
    conn = get_connection()
    try:
        if ratings_csv and os.path.exists(ratings_csv):
            print("Importing ratings data...")
            df = pd.read_csv(ratings_csv)

            # Insert unique movies
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id, title) VALUES (?, ?)",
                [(int(mid), f"Movie {int(mid)}") for mid in df["item_id"].drop_duplicates()],
            )
            # Insert unique users (no password hash: these are ratings-only seeds,
            # mirroring db.database.seed_if_empty so the two paths agree).
            conn.executemany(
                "INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                [(int(uid), f"user_{int(uid)}", None) for uid in df["user_id"].drop_duplicates()],
            )
            # Insert ratings in batch
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
                [
                    (int(r.user_id), int(r.item_id), float(r.rating), int(r.timestamp))
                    for r in df.itertuples()
                ],
            )
            print(f"Imported {len(df)} ratings.")

        if movies_csv and os.path.exists(movies_csv):
            print("Updating movie metadata...")
            with open(movies_csv, encoding="utf-8") as f:
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
            print(f"Updated {len(movies)} movies.")

        conn.commit()
    finally:
        conn.close()
    print("Database initialized.")


if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ratings.csv")
    movies_json = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "movies_enriched.json")
    if os.path.exists(csv):
        init_db(csv, movies_json if os.path.exists(movies_json) else None)
    else:
        print("Ratings CSV not found. Run data/download.py first.")
