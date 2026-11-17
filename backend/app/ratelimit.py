"""Rate limiting.

Fixed windows counted per key. Two backends: Redis (shared by every API process, used in production) and an in-process
dictionary (development and tests only, because each process would count on its own).

The limits are guards against abuse, not quotas: they sit well above what the app itself does (the editor polls tracking
once a second, for example). If Redis can't be reached the limiter lets the request through and logs it, because
refusing every request when a cache is down turns a small problem into an outage; Redis is also the job queue, so a
Redis outage already stops processing.

Keys never contain raw personal data: IP addresses are used as they come (they are the point), emails are hashed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import math
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, HTTPException, Request, status

from app.auth.deps import CurrentUser
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    limit: int
    window_seconds: int
    message: str  # `{wait}` is replaced with "30 seconds" / "5 minutes"


_SLOW_DOWN = "You're doing that too quickly. Please wait {wait} and try again."

RULES: dict[str, Rule] = {
    # Every call to the API from one address: a flood guard.
    "api_ip": Rule(900, 60, "Too many requests from your network. Please wait {wait} and try again."),
    # Signing in: every attempt from an address, and the failed attempts against one account.
    "login_ip": Rule(30, 900, "Too many sign-in attempts from your network. Please wait {wait} and try again."),
    "login_account": Rule(8, 900, "Too many failed sign-in attempts for this account. Please wait {wait} and try again."),
    "register_ip": Rule(10, 3600, "Too many accounts were created from your network. Please wait {wait} and try again."),
    # Per signed-in person.
    "project_create": Rule(30, 3600, _SLOW_DOWN),
    "upload_urls": Rule(600, 3600, _SLOW_DOWN),
    "mask_save": Rule(600, 3600, _SLOW_DOWN),
    "tracking_start": Rule(60, 3600, _SLOW_DOWN),
    "render_start": Rule(30, 3600, _SLOW_DOWN),
    "export_link": Rule(60, 3600, _SLOW_DOWN),
}


@dataclass(frozen=True)
class Reading:
    count: int
    seconds_left: int  # until the window resets


class Backend(Protocol):
    def hit(self, key: str, window: int) -> Reading: ...
    def peek(self, key: str) -> Reading: ...
    def reset(self, key: str) -> None: ...


class MemoryBackend:
    """Counts in this process only. Development and tests."""

    def __init__(self) -> None:
        self._counts: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, window: int) -> Reading:
        now = time.monotonic()
        with self._lock:
            if len(self._counts) > 20_000:
                self._counts = {k: v for k, v in self._counts.items() if v[1] > now}
            count, expires = self._counts.get(key, (0, 0.0))
            if expires <= now:
                count, expires = 0, now + window
            count += 1
            self._counts[key] = (count, expires)
            return Reading(count, max(1, math.ceil(expires - now)))

    def peek(self, key: str) -> Reading:
        now = time.monotonic()
        with self._lock:
            count, expires = self._counts.get(key, (0, 0.0))
            return Reading(count, max(1, math.ceil(expires - now))) if expires > now else Reading(0, 0)

    def reset(self, key: str) -> None:
        with self._lock:
            self._counts.pop(key, None)


class RedisBackend:
    def __init__(self, url: str) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)

    def hit(self, key: str, window: int) -> Reading:
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()
        if ttl is None or ttl < 0:  # a new key, or one that lost its expiry: (re)start the window
            self._redis.expire(key, window)
            ttl = window
        return Reading(int(count), int(ttl))

    def peek(self, key: str) -> Reading:
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.ttl(key)
        value, ttl = pipe.execute()
        return Reading(int(value or 0), max(0, int(ttl or 0)))

    def reset(self, key: str) -> None:
        self._redis.delete(key)


def wait_text(seconds: int) -> str:
    if seconds < 90:
        return f"{max(1, seconds)} seconds"
    minutes = math.ceil(seconds / 60)
    if minutes < 90:
        return f"{minutes} minutes"
    return f"{math.ceil(minutes / 60)} hours"


class TooManyRequests(HTTPException):
    def __init__(self, rule: str, seconds: int) -> None:
        text = RULES[rule].message.format(wait=wait_text(seconds))
        super().__init__(status.HTTP_429_TOO_MANY_REQUESTS, text, headers={"Retry-After": str(max(1, seconds))})
        self.rule = rule


class RateLimiter:
    def __init__(self, backend: Backend, *, enabled: bool = True) -> None:
        self._backend = backend
        self._enabled = enabled
        self._last_backend_warning = 0.0

    def _safe(self, action: str, rule: str, call):  # noqa: ANN001, ANN202
        try:
            return call()
        except Exception:
            now = time.monotonic()
            if now - self._last_backend_warning > 60:  # one line a minute, not one per request
                self._last_backend_warning = now
                logger.error("rate limiter backend failed; letting requests through", extra={"rule": rule, "action": action}, exc_info=True)
            return None

    def enforce(self, rule: str, key: str) -> None:
        """Count one request against `key`; raise 429 when the rule's limit is passed."""
        if not self._enabled:
            return
        spec = RULES[rule]
        reading = self._safe("hit", rule, lambda: self._backend.hit(f"rl:{rule}:{key}", spec.window_seconds))
        if reading is not None and reading.count > spec.limit:
            logger.warning("rate limited", extra={"rule": rule, "retry_after": reading.seconds_left})
            raise TooManyRequests(rule, reading.seconds_left)

    def require_clear(self, rule: str, key: str) -> None:
        """Raise 429 if `key` has already used up the rule, without counting this call."""
        if not self._enabled:
            return
        spec = RULES[rule]
        reading = self._safe("peek", rule, lambda: self._backend.peek(f"rl:{rule}:{key}"))
        if reading is not None and reading.count >= spec.limit:
            logger.warning("rate limited", extra={"rule": rule, "retry_after": reading.seconds_left})
            raise TooManyRequests(rule, reading.seconds_left)

    def record(self, rule: str, key: str) -> None:
        """Count one event (a failed sign-in) against `key` without refusing anything."""
        if self._enabled:
            self._safe("hit", rule, lambda: self._backend.hit(f"rl:{rule}:{key}", RULES[rule].window_seconds))

    def clear(self, rule: str, key: str) -> None:
        if self._enabled:
            self._safe("reset", rule, lambda: self._backend.reset(f"rl:{rule}:{key}"))


@lru_cache
def get_limiter() -> RateLimiter:
    settings = get_settings()
    use_redis = settings.rate_limit_storage == "redis" or (settings.rate_limit_storage == "auto" and not settings.dev_mode)
    backend: Backend = RedisBackend(settings.redis_url) if use_redis else MemoryBackend()
    return RateLimiter(backend, enabled=settings.rate_limit_enabled)


Limiter = Annotated[RateLimiter, Depends(get_limiter)]


def client_ip(request: Request, trusted_proxies: int | None = None) -> str:
    """The caller's address. Behind `n` trusted proxies the real client is the n-th entry from the right of
    X-Forwarded-For (everything to its left was written by the client and can't be trusted). With none configured the
    header is ignored entirely, so it can't be used to dodge a limit."""
    n = get_settings().trusted_proxy_count if trusted_proxies is None else trusted_proxies
    peer = request.client.host if request.client else "unknown"
    if n <= 0:
        return peer
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(parts) < n:
        return peer
    candidate = parts[-n]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer


def email_key(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:32]


def per_user(rule: str):  # noqa: ANN201
    """Route dependency: count the request against the signed-in person (401 comes first for anonymous callers)."""

    def dependency(user: CurrentUser, limiter: Limiter) -> None:
        limiter.enforce(rule, str(user.id))

    return Depends(dependency)


