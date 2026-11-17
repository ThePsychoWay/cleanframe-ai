import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import create_access_token
from app.config import ConfigError, get_settings
from app.main import create_app
from app.models import User
from tests.conftest import ALLOWED_ORIGIN, GOOD_PASSWORD, TEST_SECRET, register


def _session_cookie_name() -> str:
    return get_settings().session_cookie_name


# --- registration -----------------------------------------------------------------------------


def test_register_creates_user_and_sets_httponly_cookie(client: TestClient) -> None:
    response = register(client)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "ada@example.com"
    assert body["name"] == "Ada Lovelace"
    assert body["role"] == "user"
    assert "password" not in body and "password_hash" not in body

    cookie = response.headers["set-cookie"].lower()
    assert f"{_session_cookie_name()}=" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie

    assert client.get("/api/auth/me").json()["email"] == "ada@example.com"


def test_register_normalises_email_and_blocks_case_variant_duplicates(client: TestClient) -> None:
    first = register(client, email="  Ada@Example.COM ")
    assert first.status_code == 201
    assert first.json()["email"] == "ada@example.com"

    duplicate = register(client, email="ADA@example.com")
    assert duplicate.status_code == 409
    assert "already exists" in duplicate.json()["detail"]


def test_password_is_stored_as_argon2id_hash(client: TestClient, db: Session) -> None:
    register(client)
    user = db.scalar(select(User).where(User.email == "ada@example.com"))
    assert user is not None
    assert user.password_hash.startswith("$argon2id$")
    assert GOOD_PASSWORD not in user.password_hash


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "ada@example.com", "name": "Ada", "password": "short"},
        {"email": "ada@example.com", "name": "Ada", "password": "x" * 129},
        {"email": "not-an-email", "name": "Ada", "password": GOOD_PASSWORD},
        {"email": "ada@example.com", "name": "", "password": GOOD_PASSWORD},
        {"email": "ada@example.com", "name": "   ", "password": GOOD_PASSWORD},
        {"email": "ada@example.com", "name": "A" * 121, "password": GOOD_PASSWORD},
        {"email": "ada@example.com", "password": GOOD_PASSWORD},
    ],
)
def test_register_rejects_invalid_input(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/api/auth/register", json=payload).status_code == 422


# --- login / logout ---------------------------------------------------------------------------


def test_login_succeeds_with_correct_credentials_in_any_email_case(client: TestClient) -> None:
    register(client)
    client.cookies.clear()

    response = client.post("/api/auth/login", json={"email": "ADA@Example.com", "password": GOOD_PASSWORD})
    assert response.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_wrong_password_and_unknown_email_give_identical_errors(client: TestClient) -> None:
    register(client)
    client.cookies.clear()

    wrong_password = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "wrong-password"})
    unknown_email = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong-password"})

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert "set-cookie" not in wrong_password.headers


def test_logout_clears_the_session(client: TestClient) -> None:
    register(client)
    assert client.get("/api/auth/me").status_code == 200

    response = client.post("/api/auth/logout")
    assert response.status_code == 204
    assert client.get("/api/auth/me").status_code == 401


# --- protected endpoint behaviour -------------------------------------------------------------


def test_me_requires_a_session(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401


def test_inactive_user_cannot_log_in_or_keep_using_a_session(client: TestClient, db: Session) -> None:
    register(client)
    user = db.scalar(select(User).where(User.email == "ada@example.com"))
    assert user is not None
    user.is_active = False
    db.commit()

    assert client.get("/api/auth/me").status_code == 401
    client.cookies.clear()
    login = client.post("/api/auth/login", json={"email": "ada@example.com", "password": GOOD_PASSWORD})
    assert login.status_code == 401


# --- token attacks ----------------------------------------------------------------------------


def _me_with_token(client: TestClient, token: str) -> int:
    client.cookies.clear()
    client.cookies.set(_session_cookie_name(), token)
    return client.get("/api/auth/me").status_code


def _claims(user_id: str, **overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    claims: dict[str, object] = {"sub": user_id, "typ": "access", "iat": now, "exp": now + timedelta(minutes=5)}
    claims.update(overrides)
    return claims


def _register_and_get_id(client: TestClient) -> str:
    return register(client).json()["id"]


def test_valid_token_is_accepted_baseline(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    token = create_access_token(uuid.UUID(user_id))
    assert _me_with_token(client, token) == 200


def test_tampered_token_is_rejected(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    token = create_access_token(uuid.UUID(user_id))
    flipped = token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1]
    assert _me_with_token(client, flipped) == 401


def test_expired_token_is_rejected(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    expired = jwt.encode(_claims(user_id, exp=datetime.now(UTC) - timedelta(minutes=1)), TEST_SECRET, algorithm="HS256")
    assert _me_with_token(client, expired) == 401


def test_token_signed_with_another_secret_is_rejected(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    forged = jwt.encode(_claims(user_id), "a-different-secret-" + "y" * 40, algorithm="HS256")
    assert _me_with_token(client, forged) == 401


def test_alg_none_token_is_rejected(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    unsigned = jwt.encode(_claims(user_id), key=None, algorithm="none")
    assert _me_with_token(client, unsigned) == 401


def test_token_without_expiry_or_with_wrong_type_is_rejected(client: TestClient) -> None:
    user_id = _register_and_get_id(client)
    no_exp = jwt.encode({"sub": user_id, "typ": "access", "iat": datetime.now(UTC)}, TEST_SECRET, algorithm="HS256")
    wrong_type = jwt.encode(_claims(user_id, typ="refresh"), TEST_SECRET, algorithm="HS256")
    assert _me_with_token(client, no_exp) == 401
    assert _me_with_token(client, wrong_type) == 401


def test_token_for_a_user_that_does_not_exist_is_rejected(client: TestClient) -> None:
    register(client)
    ghost = create_access_token(uuid.uuid4())
    assert _me_with_token(client, ghost) == 401


def test_garbage_cookie_is_rejected(client: TestClient) -> None:
    assert _me_with_token(client, "not-a-jwt") == 401


# --- origin guard and CORS --------------------------------------------------------------------


def test_cross_origin_writes_from_unlisted_origins_are_blocked(client: TestClient) -> None:
    payload = {"email": "ada@example.com", "password": GOOD_PASSWORD}
    blocked = client.post("/api/auth/login", json=payload, headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403

    allowed = client.post("/api/auth/login", json=payload, headers={"Origin": ALLOWED_ORIGIN})
    assert allowed.status_code == 401  # reached the handler; credentials are just unknown

    no_origin = client.post("/api/auth/login", json=payload)
    assert no_origin.status_code == 401


def test_cors_preflight_allows_only_listed_origins_with_credentials(client: TestClient) -> None:
    ok = client.options(
        "/api/auth/login",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert ok.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert ok.headers["access-control-allow-credentials"] == "true"

    bad = client.options(
        "/api/auth/login",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


# --- configuration ----------------------------------------------------------------------------


def test_cookie_is_secure_outside_dev_mode(env: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    from app.db import get_engine
    from app.models import Base

    env.setenv("DEV_MODE", "false")
    env.setenv("AI_PROVIDER", "local")  # the development engine is refused in production
    env.setenv("RATE_LIMIT_ENABLED", "true")  # ...and so is running without rate limiting (no Redis here: it fails open)
    get_settings.cache_clear()
    app = create_app()
    Base.metadata.create_all(get_engine())
    with TestClient(app, base_url="https://testserver") as secure_client:
        response = register(secure_client)
    assert "secure" in response.headers["set-cookie"].lower()


@pytest.mark.parametrize("secret", ["", "too-short"])
def test_startup_fails_fast_on_weak_or_missing_jwt_secret(env: pytest.MonkeyPatch, secret: str) -> None:
    env.setenv("JWT_SECRET", secret)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="JWT_SECRET"):
        with TestClient(create_app()):
            pass


def test_startup_fails_fast_without_database_url(env: pytest.MonkeyPatch) -> None:
    env.setenv("DATABASE_URL", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        with TestClient(create_app()):
            pass
