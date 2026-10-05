"""Authentication endpoints and stateless demo session tokens.

Security model (intentionally lightweight, and honestly documented):
- Passwords are stored as PBKDF2-HMAC-SHA256 with a per-user random salt.
- Login returns an HMAC-signed token binding the caller to a user id; the token
  is verified on state-changing / user-scoped endpoints so a caller cannot read
  or write another user's data by guessing an id.
- This is *not* a full auth system (no revocation, single shared secret) — it is
  a stateless demo session with a signed expiry, adequate for a portfolio deployment.
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from db.database import get_connection

router = APIRouter()

#: HMAC key for session tokens. There is deliberately **no** hard-coded fallback:
#: if the operator forgets to set ``CINEREC_SECRET`` we generate an ephemeral
#: random key, so a publicly-known default can never be used to forge tokens.
#: (A side effect is that sessions stop validating on restart — acceptable, and
#: far safer, for a demo deployment.)
_SECRET = os.environ.get("CINEREC_SECRET") or secrets.token_hex(32)

#: Signed token lifetime; tokens older than this are rejected (default 7 days).
_TOKEN_TTL_SECONDS = max(60, int(os.environ.get("CINEREC_TOKEN_TTL", 7 * 24 * 3600)))

_PBKDF2_ITERATIONS = 200_000
_LEGACY_SHA256_HEX_LEN = 64


def _hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a fresh random salt."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify against a stored hash; also accepts legacy unsalted sha256 hashes."""
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt, expected = stored.split("$")
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest.hex(), expected)
    # Legacy unsalted sha256 (accounts created before the migration).
    if len(stored) == _LEGACY_SHA256_HEX_LEN:
        return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)
    return False


def _is_legacy_hash(stored: str) -> bool:
    return bool(stored) and not stored.startswith("pbkdf2_sha256$") and len(stored) == _LEGACY_SHA256_HEX_LEN


def make_token(user_id: int) -> str:
    """Create a stateless demo token binding the caller to a user id.

    The signed payload is ``<user_id>.<issued_at_epoch>``; the signature covers
    the timestamp too, so it cannot be edited to extend a token's life.
    """
    payload = f"{int(user_id)}.{int(time.time())}"
    sig = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str):
    """Return the user id encoded in ``token``, or ``None`` when it is invalid/expired."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    raw_id, raw_ts, sig = parts
    if not raw_id.isdigit() or not raw_ts.isdigit():
        return None
    payload = f"{raw_id}.{raw_ts}"
    expected = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    if time.time() - int(raw_ts) > _TOKEN_TTL_SECONDS:
        return None
    return int(raw_id)


def resolve_user(
    authorization: str = Header(default=None),
    x_user_token: str = Header(default=None),
) -> int:
    """FastAPI dependency: resolve the authenticated demo user from a token header."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_user_token:
        token = x_user_token.strip()
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(401, "Missing or invalid session token")
    return user_id


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(req: LoginRequest):
    conn = get_connection()
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?", (req.username,)
    ).fetchone()

    # Uniform error message: never reveal whether the username exists.
    if not user or not _verify_password(req.password, user["password_hash"]):
        conn.close()
        raise HTTPException(401, "Invalid username or password / 用户名或密码错误")

    # Transparently upgrade legacy unsalted hashes on successful login.
    if _is_legacy_hash(user["password_hash"]):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hash_password(req.password), user["id"]),
        )
        conn.commit()
    conn.close()

    return {
        "user_id": user["id"],
        "username": user["username"],
        "token": make_token(user["id"]),
        "message": "Login success / 登录成功",
    }


@router.post("/register")
async def register(req: RegisterRequest):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (req.username, _hash_password(req.password)),
        )
        conn.commit()
        user = conn.execute(
            "SELECT id, username FROM users WHERE username = ?", (req.username,)
        ).fetchone()
        user_id, username = user["id"], user["username"]
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username already exists / 用户名已存在")
    finally:
        conn.close()

    return {
        "user_id": user_id,
        "username": username,
        "token": make_token(user_id),
        "message": "Registration success / 注册成功",
    }


@router.get("/guest")
async def guest_login():
    """Assign a random existing demo user and issue a session token for it."""
    conn = get_connection()
    user = conn.execute("SELECT id, username FROM users ORDER BY RANDOM() LIMIT 1").fetchone()
    conn.close()
    if not user:
        raise HTTPException(404, "No users in database")
    return {
        "user_id": user["id"],
        "username": user["username"],
        "token": make_token(user["id"]),
        "message": "Guest login / 游客登录",
    }
