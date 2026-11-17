"""Password hashing and session-token helpers."""

import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import get_settings

JWT_ALGORITHM = "HS256"
TOKEN_TYPE = "access"

_hasher = PasswordHasher()  # argon2id with the library's current recommended parameters


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


@lru_cache
def _dummy_hash() -> str:
    return _hasher.hash("not-a-real-password")


def burn_password_check(password: str) -> None:
    """Spend the same time as a real verification, so unknown emails can't be told apart by timing."""
    verify_password(_dummy_hash(), password)


def create_access_token(user_id: uuid.UUID) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "typ": TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> uuid.UUID | None:
    """Return the user id for a valid, unexpired access token, otherwise None."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],  # pinned: never trust the algorithm named in the token
            options={"require": ["exp", "iat", "sub"]},
        )
        if claims.get("typ") != TOKEN_TYPE:
            return None
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError, TypeError):
        return None
