"""Shared pytest setup: run the API against an isolated temporary database.

The API resolves its SQLite path from ``db.database.DB_PATH`` at call time, so
we point it at a throwaway file *before* importing any application module.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db.database as _database  # noqa: E402

_TMP_DIR = tempfile.mkdtemp(prefix="cinerec-test-")
_database.DB_PATH = os.path.join(_TMP_DIR, "test.db")
# Keep the app's startup seeding out of the tests: point the seed sources at a
# nonexistent path so `seed_if_empty` is a no-op and our tiny fixture is used.
_database.RATINGS_CSV = os.path.join(_TMP_DIR, "no-ratings.csv")
_database.MOVIES_JSON = os.path.join(_TMP_DIR, "no-movies.json")
_database.init_db()


def _seed():
    """Insert a tiny deterministic dataset so API tests have data to serve."""
    conn = _database.get_connection()
    conn.executemany(
        "INSERT OR IGNORE INTO movies (id, title, genres, poster_url, release_year) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Alpha", "Action|Sci-Fi", "", 1999),
            (2, "Beta", "Comedy", "", 2001),
            (3, "Gamma", "Drama", "", 2003),
            (4, "Delta", "Action", "", 2005),
            (5, "Epsilon", "Romance", "", 2007),
        ],
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (1, 'alice', 'x')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (2, 'bob', 'x')"
    )
    conn.commit()
    conn.close()


_seed()
