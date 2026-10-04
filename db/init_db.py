import sqlite3
import os
import pandas as pd
from db.database import DB_PATH, init_db as create_schema

def init_db(ratings_csv=None, movies_csv=None):
    """Initialize database with tables and optionally import data."""
    create_schema()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if ratings_csv and os.path.exists(ratings_csv):
        print("Importing ratings data...")
        df = pd.read_csv(ratings_csv)
        # Insert unique movies
        movies = df[["item_id"]].drop_duplicates()
        for mid in movies["item_id"]:
            c.execute("INSERT OR IGNORE INTO movies (id, title) VALUES (?, ?)",
                      (int(mid), f"Movie {int(mid)}"))
        # Insert unique users
        users = df[["user_id"]].drop_duplicates()
        for uid in users["user_id"]:
            c.execute("INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                      (int(uid), f"user_{int(uid)}", "guest"))
        # Insert ratings in batch
        conn.executemany(
            "INSERT OR IGNORE INTO ratings VALUES (?,?,?,?)",
            [(int(r["user_id"]), int(r["item_id"]), float(r["rating"]), int(r["timestamp"]))
             for _, r in df.iterrows()]
        )
        print(f"Imported {len(df)} ratings.")

    if movies_csv and os.path.exists(movies_csv):
        print("Updating movie metadata...")
        mdf = pd.read_json(movies_csv)
        for _, row in mdf.iterrows():
            c.execute("""UPDATE movies SET title=?, genres=?, overview=?,
                        poster_url=?, tmdb_id=?, release_year=? WHERE id=?""",
                     (row.get("title", ""), row.get("genres", ""),
                      row.get("overview", ""), row.get("poster_url", ""),
                      row.get("tmdb_id", ""),
                      int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
                      int(row["id"])))
        print(f"Updated {len(mdf)} movies.")

    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ratings.csv")
    movies_json = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "movies_enriched.json")
    if os.path.exists(csv):
        init_db(csv, movies_json if os.path.exists(movies_json) else None)
    else:
        print("Ratings CSV not found. Run data/download.py first.")
