"""ASGI middleware: the request body cap, and the per-request context (ids, access log, security headers, error answers).

The first part caps request body size.

Every API body is small JSON; video bytes go straight to object storage and never reach the API. So a
tight global cap costs nothing legitimate and stops a client from making the server buffer something
huge. It checks the declared Content-Length first, and counts the bytes of streamed (chunked) bodies as
they arrive, because a client can lie about or omit the header. It runs before the body is parsed.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_settings
from app.logging_config import clear_context, start_context

TOO_LARGE = "This request is too large."


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int, exempt_prefixes: tuple[str, ...] = ()) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt_prefixes = exempt_prefixes  # routes that enforce their own, larger limit (upload parts in local mode)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path", "").startswith(self.exempt_prefixes):
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse({"detail": "Invalid request."}, status_code=400)(scope, receive, send)
                return
            if int(declared) > self.max_bytes:
                await JSONResponse({"detail": TOO_LARGE}, status_code=413)(scope, receive, send)
                return

        received = 0
        started = False  # the app has begun its own response
        refused = False  # we have already answered 413

        async def limited_receive() -> Message:
            nonlocal received, refused
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    if not started and not refused:
                        # Answer now. Frameworks between here and the route may turn an exception raised
                        # while reading the body into their own error (FastAPI makes it a 400), so the
                        # response can't depend on how the exception is handled.
                        refused = True
                        await JSONResponse({"detail": TOO_LARGE}, status_code=413)(scope, receive, send)
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if refused:
                return  # the 413 is already out; drop whatever the app says afterwards
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not refused and not started:
                await JSONResponse({"detail": TOO_LARGE}, status_code=413)(scope, receive, send)
            elif started and not refused:
                raise  # too late to change the response; the connection is dropped instead


# --- request context: ids, access log, security headers, and a readable answer for unexpected errors ---------------

logger = logging.getLogger("app.request")

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
DATABASE_DOWN = "We can't reach our database right now. Please try again in a moment."
UNEXPECTED = "Something went wrong on our side. Please try again in a moment. If it keeps happening, quote reference {request_id}."
QUIET_PATHS = frozenset({"/api/health"})

# Sent on every API response. They cost nothing and the API only ever returns JSON, an event stream or an empty body.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",  # responses are personal; only applied where the route didn't choose otherwise
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


def friendly_error(error: Exception, request_id: str) -> tuple[int, str]:
    """What to tell the person about an error nobody planned for. Never the exception text itself."""
    if isinstance(error, PoolTimeout | OperationalError):
        return 503, DATABASE_DOWN
    return 500, UNEXPECTED.format(request_id=request_id)


class RequestContextMiddleware:
    """Gives each request an id (returned as X-Request-ID and attached to every log line it causes), writes one access
    line per request, adds the security headers, and answers an unexpected error with a message a person can read
    instead of a bare 500. It sits inside CORS, so even that answer carries the headers the browser needs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if REQUEST_ID_PATTERN.match(incoming) else uuid.uuid4().hex[:16]
        start_context(request_id=request_id)
        started = time.perf_counter()
        status_code = 0
        response_started = False
        hsts = not get_settings().dev_mode

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code, response_started = message["status"], True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    if name not in headers:
                        headers[name] = value
                if hsts and "Strict-Transport-Security" not in headers:
                    headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except ClientDisconnect:
            status_code = 499  # the caller went away; nobody to answer
        except Exception as error:
            logger.exception("unhandled error")
            if response_started:
                raise
            status, message = friendly_error(error, request_id)
            await JSONResponse({"detail": message, "request_id": request_id}, status_code=status)(scope, receive, send_with_headers)
        finally:
            self._log(scope, status_code, started)
            clear_context()

    @staticmethod
    def _log(scope: Scope, status_code: int, started: float) -> None:
        path = str(scope.get("path", ""))
        route = getattr(scope.get("route"), "path", None)
        level = logging.ERROR if status_code >= 500 else logging.DEBUG if path in QUIET_PATHS else logging.INFO
        logger.log(
            level,
            "request",
            extra={
                "method": scope.get("method"),
                "path": path[:200],
                "route": route,
                "status": status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
