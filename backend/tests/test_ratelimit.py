"""Rate limiting: windows, the sign-in rules, per-person limits, the flood guard, client addresses and Redis."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import ratelimit
from app.config import ConfigError, Settings, get_settings
from app.ratelimit import (
    RULES,
    MemoryBackend,
    RateLimiter,
    RedisBackend,
    Rule,
    TooManyRequests,
    client_ip,
    email_key,
    get_limiter,
    wait_text,
)
from tests.conftest import ALLOWED_ORIGIN, GOOD_PASSWORD, register


@pytest.fixture
def limiting(env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    env.setenv("RATE_LIMIT_ENABLED", "true")
    get_settings.cache_clear()
    get_limiter.cache_clear()
    return env


def login(client: TestClient, email: str, password: str):  # noqa: ANN202
    return client.post("/api/auth/login", json={"email": email, "password": password})


# --- counting --------------------------------------------------------------------------------------


def test_a_window_counts_then_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    backend = MemoryBackend()
    assert [backend.hit("k", 60).count for _ in range(3)] == [1, 2, 3]
    assert backend.peek("k").count == 3 and backend.peek("k").seconds_left == 60
    now[0] += 59
    assert backend.hit("k", 60).count == 4
    now[0] += 2
    assert backend.hit("k", 60).count == 1, "a new window starts after the old one ends"
    backend.reset("k")
    assert backend.peek("k") == ratelimit.Reading(0, 0)


def test_keys_are_counted_separately() -> None:
    backend = MemoryBackend()
    backend.hit("a", 60)
    backend.hit("a", 60)
    assert backend.peek("a").count == 2 and backend.peek("b").count == 0


def test_the_limiter_refuses_after_the_limit_and_says_when_to_come_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(RULES, "export_link", Rule(2, 120, "Slow down. Wait {wait}."))
    limiter = RateLimiter(MemoryBackend())
    limiter.enforce("export_link", "me")
    limiter.enforce("export_link", "me")
    with pytest.raises(TooManyRequests) as caught:
        limiter.enforce("export_link", "me")
    assert caught.value.status_code == 429
    assert caught.value.detail == "Slow down. Wait 2 minutes."
    assert 1 <= int(caught.value.headers["Retry-After"]) <= 120
    limiter.enforce("export_link", "someone else")  # not affected


def test_a_disabled_limiter_never_refuses() -> None:
    limiter = RateLimiter(MemoryBackend(), enabled=False)
    for _ in range(10_000):
        limiter.enforce("login_account", "x")
    limiter.require_clear("login_account", "x")


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(1, "1 seconds"), (45, "45 seconds"), (89, "89 seconds"), (120, "2 minutes"), (301, "6 minutes"), (7200, "2 hours")],
)
def test_waiting_times_read_naturally(seconds: int, text: str) -> None:
    assert wait_text(seconds) == text


# --- who is asking ---------------------------------------------------------------------------------


def request_with(peer: str, forwarded: str | None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234), "method": "GET", "path": "/"})


def test_the_forwarded_header_is_ignored_unless_proxies_are_trusted() -> None:
    assert client_ip(request_with("10.0.0.5", "1.2.3.4"), trusted_proxies=0) == "10.0.0.5"


def test_behind_one_proxy_the_address_the_proxy_saw_is_used_and_spoofed_entries_are_not() -> None:
    # The client wrote "6.6.6.6"; the proxy appended the address it actually saw.
    assert client_ip(request_with("10.0.0.5", "6.6.6.6, 203.0.113.9"), trusted_proxies=1) == "203.0.113.9"
    assert client_ip(request_with("10.0.0.5", "6.6.6.6, 203.0.113.9, 10.0.0.2"), trusted_proxies=2) == "203.0.113.9"


@pytest.mark.parametrize("header", [None, "", "not-an-ip", "1.2.3.4"])
def test_a_missing_short_or_invalid_header_falls_back_to_the_connection(header: str | None) -> None:
    expected = "1.2.3.4" if header == "1.2.3.4" else "10.0.0.5"
    result = client_ip(request_with("10.0.0.5", header), trusted_proxies=1)
    assert result == expected
    assert client_ip(request_with("10.0.0.5", "a, b"), trusted_proxies=3) == "10.0.0.5"


def test_emails_are_hashed_and_case_insensitive() -> None:
    assert email_key("Ada@Example.com ") == email_key("ada@example.com")
    assert "ada" not in email_key("ada@example.com")


# --- signing in ------------------------------------------------------------------------------------


def test_repeated_wrong_passwords_lock_one_account_for_a_while_even_for_the_right_password(
    limiting: pytest.MonkeyPatch, client: TestClient
) -> None:
    register(client)
    register(client, email="grace@example.com", name="Grace Hopper")
    attacker = TestClient(client.app)
    for _ in range(RULES["login_account"].limit):
        assert login(attacker, "ada@example.com", "wrong password!").status_code == 401

    blocked = login(attacker, "ada@example.com", GOOD_PASSWORD)
    assert blocked.status_code == 429
    assert "Too many failed sign-in attempts" in blocked.json()["detail"] and "minutes" in blocked.json()["detail"]
    assert int(blocked.headers["retry-after"]) > 0
    assert login(attacker, "grace@example.com", GOOD_PASSWORD).status_code == 200, "other accounts are not affected"


def test_an_unknown_email_is_limited_the_same_way_so_lockout_reveals_nothing(
    limiting: pytest.MonkeyPatch, client: TestClient
) -> None:
    for _ in range(RULES["login_account"].limit):
        assert login(client, "nobody@example.com", "wrong password!").status_code == 401
    assert login(client, "nobody@example.com", "wrong password!").status_code == 429


def test_a_successful_sign_in_clears_the_failure_count(limiting: pytest.MonkeyPatch, client: TestClient) -> None:
    register(client)
    other = TestClient(client.app)
    for _ in range(RULES["login_account"].limit - 1):
        assert login(other, "ada@example.com", "wrong password!").status_code == 401
    assert login(other, "ada@example.com", GOOD_PASSWORD).status_code == 200
    for _ in range(RULES["login_account"].limit - 1):
        assert login(other, "ada@example.com", "wrong password!").status_code == 401, "the counter started again from zero"


def test_one_address_cannot_try_many_accounts_forever(limiting: pytest.MonkeyPatch, client: TestClient) -> None:
    limit = RULES["login_ip"].limit
    for index in range(limit):
        assert login(client, f"user{index}@example.com", "wrong password!").status_code == 401
    refused = login(client, "another@example.com", "wrong password!")
    assert refused.status_code == 429 and "your network" in refused.json()["detail"]


def test_account_creation_is_limited_per_address(limiting: pytest.MonkeyPatch, client: TestClient) -> None:
    limit = RULES["register_ip"].limit
    for index in range(limit):
        assert register(TestClient(client.app), email=f"new{index}@example.com").status_code == 201
    refused = register(TestClient(client.app), email="one-too-many@example.com")
    assert refused.status_code == 429 and "accounts were created" in refused.json()["detail"]


def test_a_spoofed_forwarded_header_does_not_dodge_the_limit_when_no_proxy_is_trusted(
    limiting: pytest.MonkeyPatch, client: TestClient
) -> None:
    limit = RULES["login_ip"].limit
    for index in range(limit):
        spoof = {"X-Forwarded-For": f"9.9.9.{index}"}
        assert client.post("/api/auth/login", json={"email": f"u{index}@example.com", "password": "wrong password!"}, headers=spoof).status_code == 401
    assert client.post("/api/auth/login", json={"email": "x@example.com", "password": "wrong password!"}, headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 429


def test_with_a_trusted_proxy_each_real_client_has_its_own_count(limiting: pytest.MonkeyPatch, client: TestClient) -> None:
    limiting.setenv("TRUSTED_PROXY_COUNT", "1")
    get_settings.cache_clear()
    get_limiter.cache_clear()
    limit = RULES["login_ip"].limit
    busy = {"X-Forwarded-For": "6.6.6.6, 203.0.113.1"}
    for index in range(limit):
        assert client.post("/api/auth/login", json={"email": f"u{index}@example.com", "password": "wrong password!"}, headers=busy).status_code == 401
    assert client.post("/api/auth/login", json={"email": "x@example.com", "password": "wrong password!"}, headers=busy).status_code == 429
    calm = {"X-Forwarded-For": "6.6.6.6, 203.0.113.2"}
    assert client.post("/api/auth/login", json={"email": "x@example.com", "password": "wrong password!"}, headers=calm).status_code == 401


# --- per person and per address --------------------------------------------------------------------


def test_project_creation_is_limited_per_person(
    limiting: pytest.MonkeyPatch, user_client: TestClient, other_client: TestClient
) -> None:
    limiting.setitem(RULES, "project_create", Rule(2, 3600, "Slow down. Wait {wait}."))
    body = {"filename": "a.mp4", "size_bytes": 6 * 1024 * 1024}
    assert user_client.post("/api/projects", json=body).status_code == 201
    assert user_client.post("/api/projects", json={**body, "filename": "b.mp4"}).status_code == 201
    refused = user_client.post("/api/projects", json={**body, "filename": "c.mp4"})
    assert refused.status_code == 429 and refused.json()["detail"].startswith("Slow down. Wait ")
    assert other_client.post("/api/projects", json=body).status_code == 201, "another person's allowance is separate"


def test_signed_out_callers_get_401_not_a_rate_limit_answer(limiting: pytest.MonkeyPatch, client: TestClient) -> None:
    limiting.setitem(RULES, "project_create", Rule(1, 3600, "Slow down."))
    for _ in range(3):
        assert client.post("/api/projects", json={"filename": "a.mp4", "size_bytes": 6 * 1024 * 1024}).status_code == 401


def test_the_flood_guard_stops_a_burst_but_not_the_health_check_or_preflights(
    limiting: pytest.MonkeyPatch, client: TestClient
) -> None:
    limiting.setitem(RULES, "api_ip", Rule(5, 60, "Too many requests. Wait {wait}."))
    headers = {"Origin": ALLOWED_ORIGIN}
    codes = [client.get("/api/auth/me", headers=headers).status_code for _ in range(5)]
    assert codes == [401] * 5
    refused = client.get("/api/auth/me", headers=headers)
    assert refused.status_code == 429 and "Retry-After" in refused.headers
    assert refused.headers["access-control-allow-origin"] == ALLOWED_ORIGIN, "the browser must be able to read the 429"
    assert client.get("/api/health").status_code == 200
    preflight = client.options("/api/projects", headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 200


# --- production settings ---------------------------------------------------------------------------


def test_production_refuses_in_process_counting(env: pytest.MonkeyPatch) -> None:
    env.setenv("DEV_MODE", "false")
    env.setenv("AI_PROVIDER", "api")
    env.setenv("RATE_LIMIT_ENABLED", "true")
    env.setenv("RATE_LIMIT_STORAGE", "memory")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="RATE_LIMIT_STORAGE=memory"):
        get_settings().validate_runtime()


def test_production_refuses_to_run_without_rate_limiting(env: pytest.MonkeyPatch) -> None:
    env.setenv("DEV_MODE", "false")
    env.setenv("AI_PROVIDER", "api")
    env.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="RATE_LIMIT_ENABLED=false"):
        get_settings().validate_runtime()
    env.setenv("RATE_LIMIT_ENABLED", "true")
    get_settings.cache_clear()
    get_settings().validate_runtime()  # auto + enabled is fine


def test_auto_means_redis_in_production_and_memory_in_development(env: pytest.MonkeyPatch) -> None:
    assert isinstance(get_limiter()._backend, MemoryBackend)  # noqa: SLF001
    env.setenv("DEV_MODE", "false")
    get_settings.cache_clear()
    get_limiter.cache_clear()
    assert isinstance(get_limiter()._backend, RedisBackend)  # noqa: SLF001
    assert Settings().rate_limit_storage == "auto"


# --- Redis -----------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def redis_url(tmp_path: Path) -> Iterator[str]:
    server = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server isn't installed")
    port = free_port()
    process = subprocess.Popen(
        [server, "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_redis_counts_share_state_across_connections_and_expire(redis_url: str) -> None:
    first, second = RedisBackend(redis_url), RedisBackend(redis_url)
    assert first.hit("k", 1).count == 1
    assert second.hit("k", 1).count == 2, "a second API process sees the first one's requests"
    assert 0 < second.peek("k").seconds_left <= 1
    time.sleep(1.3)
    assert first.peek("k").count == 0 and first.hit("k", 1).count == 1
    first.reset("k")
    assert second.peek("k").count == 0


def test_a_redis_key_that_lost_its_expiry_is_given_one_again(redis_url: str) -> None:
    backend = RedisBackend(redis_url)
    backend._redis.set("stuck", 99)  # noqa: SLF001  (no TTL: what a crash between INCR and EXPIRE would leave)
    reading = backend.hit("stuck", 30)
    assert reading.count == 100 and 0 < reading.seconds_left <= 30


def test_the_limiter_works_end_to_end_on_redis(redis_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(RULES, "export_link", Rule(3, 60, "Wait {wait}."))
    limiter = RateLimiter(RedisBackend(redis_url))
    for _ in range(3):
        limiter.enforce("export_link", "me")
    with pytest.raises(TooManyRequests):
        limiter.enforce("export_link", "me")


def test_when_redis_is_down_requests_pass_and_the_problem_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    limiter = RateLimiter(RedisBackend(f"redis://127.0.0.1:{free_port()}/0"))
    with caplog.at_level(logging.ERROR, logger="app.ratelimit"):
        for _ in range(5):
            limiter.enforce("login_ip", "1.2.3.4")
            limiter.require_clear("login_account", "x")
            limiter.record("login_account", "x")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "letting requests through" in errors[0].getMessage()
