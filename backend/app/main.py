"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth as auth_api
from app.api import editor as editor_api
from app.api import engine as engine_api
from app.api import projects as projects_api
from app.api import render as render_api
from app.api import exports as exports_api
from app.api import result as result_api
from app.api import tracking as tracking_api
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.ratelimit import TooManyRequests, client_ip, get_limiter
from app.storage.s3 import StorageError, get_storage

logger = logging.getLogger(__name__)

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _prepare_local(settings: "Settings") -> None:
    """Local mode: make the storage folder and bring the SQLite database up to date, so the first run just works."""
    from pathlib import Path

    from alembic.config import Config

    from alembic import command

    get_storage().ensure_bucket()
    root = Path(__file__).resolve().parent.parent  # the backend folder, where alembic.ini and alembic/ live
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(config, "head")
    logger.info("local mode ready", extra={"data_dir": str(Path(settings.local_data_dir).resolve())})


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    allowed_origins = set(settings.cors_origin_list)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        current = get_settings()
        current.validate_runtime()
        if current.local_mode:
            _prepare_local(current)
        if current.dev_mode and current.storage_endpoint and current.storage_backend == "s3":
            # Local MinIO starts empty. Production buckets are provisioned, never auto-created.
            try:
                get_storage().ensure_bucket()
                from app.storage.provision import provision

                provision()
            except Exception:
                logger.warning("storage isn't reachable yet; uploads will fail until it is", exc_info=True)
        if current.task_backend == "thread":
            from app.workers import local

            local.recover_interrupted()
            local.start_sweeper()
        if current.ai_keep_model_loaded:
            import threading

            from app.ai.registry import warm_up

            threading.Thread(target=warm_up, args=(current,), name="cf-warm-up", daemon=True).start()
        yield
        if current.task_backend == "thread":
            from app.workers import local

            local.shutdown()

    # The interactive API docs describe every route; they're a development tool, not something to publish.
    docs = {} if settings.dev_mode else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="CleanFrame AI API", version="0.2.0", lifespan=lifespan, **docs)

    @app.middleware("http")
    async def origin_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Defence in depth on top of SameSite cookies: browsers send Origin on cross-site writes."""
        if request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and origin.rstrip("/") not in allowed_origins:
                return JSONResponse({"detail": "This origin isn't allowed to call the API."}, status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def flood_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """One generous limit per address on every API call, before any work is done for it."""
        if request.url.path.startswith("/api/") and request.url.path != "/api/health" and request.method != "OPTIONS":
            try:
                get_limiter().enforce("api_ip", client_ip(request))
            except TooManyRequests as refused:
                return JSONResponse({"detail": refused.detail}, status_code=refused.status_code, headers=refused.headers)
        return await call_next(request)

    # Added before CORS so that CORS wraps it and a 413 still carries the headers the browser needs.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes, exempt_prefixes=("/api/local-files/",))
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        expose_headers=["ETag"],  # the browser reads each uploaded part's ETag (local mode sends parts to this API)
    )

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "dev_mode": settings.dev_mode, "ai_provider": settings.ai_provider}

    @app.exception_handler(StorageError)
    async def storage_error_handler(_: Request, error: StorageError) -> JSONResponse:
        logger.error("storage error: %s", error, exc_info=error)
        return JSONResponse({"detail": f"{error} Please try again in a moment."}, status_code=502)

    app.include_router(auth_api.router)
    app.include_router(projects_api.router)
    app.include_router(editor_api.router)
    app.include_router(tracking_api.router)
    app.include_router(render_api.router)
    app.include_router(result_api.router)
    app.include_router(exports_api.router)
    app.include_router(engine_api.router)
    if settings.storage_backend == "filesystem":
        from app.api import local_files as local_files_api

        app.include_router(local_files_api.router)
    return app


app = create_app()
