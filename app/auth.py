"""Username/password accounts: bcrypt for password storage, a signed JWT as
the mobile app's session token (stateless - no server-side session table to
manage/expire), and a small in-memory rate limiter on login attempts since
this endpoint is the one place in the app that takes a secret from the
client and is worth minimally protecting against brute force.

JWT_SECRET should be set in backend/.env for real use - if it's missing this
generates a random one at import time so local dev still works, but every
restart invalidates all existing sessions (logged out) when it does.
"""

import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .db import User, get_db

JWT_SECRET = os.environ.get("JWT_SECRET") or uuid.uuid4().hex
JWT_ALGORITHM = "HS256"
TOKEN_TTL_DAYS = 30

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,20}$")
_MIN_PASSWORD_LENGTH = 8

# username -> list of failed-attempt timestamps in the current window.
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_RATE_LIMIT_WINDOW_SECONDS = 15 * 60
_RATE_LIMIT_MAX_ATTEMPTS = 10


def normalize_username(username: str) -> str:
    return username.strip().lower()


def validate_username(username: str) -> str:
    normalized = normalize_username(username)
    if not _USERNAME_RE.match(normalized):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-20 characters: letters, numbers, and underscores only.",
        )
    return normalized


def validate_password(password: str) -> None:
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def check_rate_limit(username: str) -> None:
    now = time.time()
    attempts = [t for t in _LOGIN_ATTEMPTS.get(username, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[username] = attempts
    if len(attempts) >= _RATE_LIMIT_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again in 15 minutes.")


def record_failed_attempt(username: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(username, []).append(time.time())


def clear_attempts(username: str) -> None:
    _LOGIN_ATTEMPTS.pop(username, None)


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_TTL_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    user = db.get(User, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
