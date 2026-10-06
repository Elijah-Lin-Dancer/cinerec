"""Shared pytest setup: run the API against an isolated temporary database.

The API resolves its SQLite path from ``db.database.DB_PATH`` at call time, so
we point it at a throwaway file *before* importing any application module.
"""
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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


def _free_port():
    """Reserve an ephemeral port so parallel runs do not collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_healthy(url, timeout=180):
    """Poll ``/api/health`` until the server is up (seeding can take a moment)."""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:  # not up yet
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server at {url} never became healthy: {last_error}")


@pytest.fixture(scope="session")
def server_url():
    """Base URL of the app under test, for the browser (``e2e``) tests.

    Reuses ``CINEREC_E2E_BASE_URL`` when set; otherwise starts a ``lite`` uvicorn
    instance on a free port for the session. ``lite`` is the free-tier deployment
    shape and needs no torch. Only requested by ``e2e`` tests, so the default unit
    run never pays for it.
    """
    existing = os.environ.get("CINEREC_E2E_BASE_URL")
    if existing:
        base = existing.rstrip("/")
        _wait_until_healthy(base)
        yield base
        return

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "APP_MODE": "lite", "CINEREC_SECRET": "e2e-secret"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=_PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_until_healthy(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
