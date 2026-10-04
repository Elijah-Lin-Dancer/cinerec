"""Tests for password hashing and demo session tokens."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.auth import _hash_password, _verify_password, make_token, verify_token  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)


def test_password_hash_is_salted_and_verifies():
    hashed = _hash_password("correct horse")
    assert hashed.startswith("pbkdf2_sha256$")
    assert "correct horse" not in hashed
    assert _verify_password("correct horse", hashed)
    assert not _verify_password("wrong", hashed)


def test_two_hashes_of_same_password_differ():
    assert _hash_password("same") != _hash_password("same")


def test_legacy_unsalted_sha256_still_verifies():
    legacy = hashlib.sha256(b"secret").hexdigest()
    assert _verify_password("secret", legacy)
    assert not _verify_password("nope", legacy)


def test_token_roundtrip_and_tampering():
    assert verify_token(make_token(42)) == 42
    assert verify_token("42.deadbeef") is None
    assert verify_token("not-a-token") is None
    assert verify_token(None) is None


def test_register_login_and_token_issue():
    reg = client.post("/api/auth/register", json={"username": "carol", "password": "pw12345"})
    assert reg.status_code == 200
    user_id = reg.json()["user_id"]
    assert verify_token(reg.json()["token"]) == user_id

    login = client.post("/api/auth/login", json={"username": "carol", "password": "pw12345"})
    assert login.status_code == 200
    assert verify_token(login.json()["token"]) == user_id


def test_login_error_is_uniform_for_unknown_user():
    resp = client.post("/api/auth/login", json={"username": "ghost", "password": "x"})
    assert resp.status_code == 401
    assert "not found" not in resp.text.lower()


def test_duplicate_registration_rejected():
    client.post("/api/auth/register", json={"username": "dave", "password": "pw12345"})
    dup = client.post("/api/auth/register", json={"username": "dave", "password": "pw12345"})
    assert dup.status_code == 400
