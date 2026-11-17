# ruff: noqa: F811  (pytest fixtures imported from test_render_api are re-declared as test arguments)
"""Structured logs, request ids, secret scrubbing, security headers and readable answers to unexpected errors."""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app import logging_config as lc
from app.config import get_settings
from app.middleware import friendly_error
from app.workers.celery_app import _bind_task, _unbind_task
from tests.conftest import ALLOWED_ORIGIN, GOOD_PASSWORD, register
from tests.test_render_api import marked, render, source_video  # noqa: F401  (fixtures)

SIGNED = "https://storage.example.com/bucket/outputs/u/p/j.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA123%2F2026&X-Amz-Signature=deadbeef1234"


class Collector(logging.Handler):
    """Formats each record to JSON at the moment it is emitted, as a real handler does (the ids come from context)."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[dict[str, Any]] = []
        self.raw: list[str] = []
        self.setFormatter(lc.JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        text = self.format(record)
        self.raw.append(text)
        self.lines.append(json.loads(text))

    def named(self, message: str) -> list[dict[str, Any]]:
        return [line for line in self.lines if line["message"] == message]


@pytest.fixture
def logs() -> Iterator[Collector]:
    collector = Collector()
    root = logging.getLogger()
    previous = root.level
    root.addHandler(collector)
    root.setLevel(logging.DEBUG)
    lc.clear_context()
    yield collector
    root.removeHandler(collector)
    root.setLevel(previous)
    lc.clear_context()


def record(message: str, **extra: Any) -> logging.LogRecord:
    made = logging.LogRecord("app.test", logging.INFO, __file__, 1, message, (), None)
    made.__dict__.update(extra)
    return made


# --- the format ------------------------------------------------------------------------------------


def test_each_line_is_one_json_object_with_time_level_logger_message_and_the_call_sites_fields() -> None:
    line = json.loads(lc.JsonFormatter().format(record("render finished", job_id="j1", duration_ms=1200, frames=30)))
    assert line["message"] == "render finished" and line["level"] == "INFO" and line["logger"] == "app.test"
    assert line["job_id"] == "j1" and line["duration_ms"] == 1200 and line["frames"] == 30
    assert line["time"].endswith("+00:00")


def test_ids_bound_for_the_request_or_job_appear_on_every_line_and_a_call_site_can_override_them() -> None:
    lc.start_context(request_id="r1", user_id="u1")
    lc.bind_context(project_id="p1", job_id=None)
    line = json.loads(lc.JsonFormatter().format(record("hello", job_id="explicit")))
    assert (line["request_id"], line["user_id"], line["project_id"], line["job_id"]) == ("r1", "u1", "p1", "explicit")
    lc.clear_context()
    assert "request_id" not in json.loads(lc.JsonFormatter().format(record("after")))


def test_ids_bound_in_a_worker_thread_are_seen_by_the_request(logs: Collector) -> None:
    """FastAPI runs sync routes in threads on a copy of the context; the ids must still reach the access line."""
    lc.start_context(request_id="r2")
    context = contextvars.copy_context()
    worker = threading.Thread(target=lambda: context.run(lc.bind_context, user_id="u9"))
    worker.start()
    worker.join()
    assert lc.current_context() == {"request_id": "r2", "user_id": "u9"}


def test_text_format_is_readable_and_scrubbed_too() -> None:
    text = lc.TextFormatter().format(record(f"fetched {SIGNED}", project_id="p1", password="hunter2"))
    assert "fetched" in text and "project_id=p1" in text and "hunter2" not in text and "deadbeef" not in text


def test_configuring_twice_leaves_one_handler_of_ours() -> None:
    root = logging.getLogger()
    lc.configure_logging(level="INFO", json_output=True)
    lc.configure_logging(level="DEBUG", json_output=False)
    ours = [h for h in root.handlers if getattr(h, "_cleanframe_handler", False)]
    assert len(ours) == 1 and isinstance(ours[0].formatter, lc.TextFormatter)
    assert root.level == logging.DEBUG
    lc.configure_logging(level="nonsense", json_output=True)
    assert root.level == logging.INFO
    root.removeHandler(next(h for h in root.handlers if getattr(h, "_cleanframe_handler", False)))


# --- secrets and personal data -----------------------------------------------------------------------


def test_signed_link_credentials_never_reach_a_log_line() -> None:
    out = lc.JsonFormatter().format(record(f"download {SIGNED} done", link=SIGNED, note="Signature=abc123 X-Amz-Security-Token=tok"))
    for secret in ("deadbeef1234", "AKIA123", "abc123", "tok", "AWS4-HMAC"):
        assert secret not in out
    assert "https://storage.example.com/bucket/outputs/u/p/j.mp4?[query removed]" in out


def test_secret_looking_fields_are_replaced_and_upload_names_are_dropped() -> None:
    line = json.loads(
        lc.JsonFormatter().format(
            record("x", password="hunter2", Authorization="Bearer abc", session_cookie="cf=1", api_key="k", jwt_secret="s", filename="holiday.mp4", original_filename="holiday.mp4", project_id="p")
        )
    )
    for key in ("password", "Authorization", "session_cookie", "api_key", "jwt_secret"):
        assert line[key] == "[redacted]"
    assert "filename" not in line and "original_filename" not in line and line["project_id"] == "p"


def test_exceptions_are_logged_with_a_traceback_but_without_the_links_inside_them() -> None:
    try:
        raise RuntimeError(f"failed to fetch {SIGNED}")
    except RuntimeError:
        import sys

        made = record("render failed")
        made.exc_info = sys.exc_info()
    line = json.loads(lc.JsonFormatter().format(made))
    assert "RuntimeError" in line["exception"] and "Traceback" in line["exception"]
    assert "deadbeef1234" not in line["exception"] and "AKIA123" not in line["exception"]


# --- the request wrapper -----------------------------------------------------------------------------


def test_every_response_has_a_request_id_and_the_security_headers(client: TestClient) -> None:
    for response in (client.get("/api/health"), client.get("/api/auth/me"), client.get("/api/nope")):
        assert len(response.headers["x-request-id"]) >= 8
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "strict-transport-security" not in response.headers, "plain http in development"


def test_hsts_is_sent_outside_development(env: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient as Client

    from app.main import create_app

    env.setenv("DEV_MODE", "false")
    env.setenv("AI_PROVIDER", "api")
    env.setenv("RATE_LIMIT_ENABLED", "true")
    get_settings.cache_clear()
    with Client(create_app(), base_url="https://testserver") as secure:
        assert "max-age=" in secure.get("/api/health").headers["strict-transport-security"]


def test_a_routes_own_cache_header_is_kept(client: TestClient) -> None:
    router = APIRouter()

    @router.get("/api/_test/stream")
    def stream():  # noqa: ANN202
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("x", headers={"Cache-Control": "no-cache"})

    client.app.include_router(router)
    assert client.get("/api/_test/stream").headers["cache-control"] == "no-cache"


def test_a_sensible_inbound_request_id_is_kept_and_a_hostile_one_is_replaced(client: TestClient) -> None:
    assert client.get("/api/health", headers={"X-Request-ID": "trace-abc_123.x"}).headers["x-request-id"] == "trace-abc_123.x"
    for bad in ("short", "x" * 200, "has space in it", "semi;colon;id"):
        assert client.get("/api/health", headers={"X-Request-ID": bad}).headers["x-request-id"] != bad


def test_one_access_line_per_request_with_route_status_duration_user_and_no_query_or_cookie(
    user_client: TestClient, logs: Collector
) -> None:
    response = user_client.get("/api/projects?secret=hunter2&page=2", headers={"X-Request-ID": "req-12345678"})
    assert response.status_code == 200
    line = next(entry for entry in logs.named("request") if entry["request_id"] == "req-12345678")
    assert (line["method"], line["path"], line["route"], line["status"]) == ("GET", "/api/projects", "/api/projects", 200)
    assert line["duration_ms"] >= 0 and line["user_id"]
    text = json.dumps(line)
    assert "hunter2" not in text and "cf_session" not in text and "page=2" not in text


def test_health_checks_are_not_logged_at_info_level(client: TestClient, logs: Collector) -> None:
    client.get("/api/health")
    health = [line for line in logs.named("request") if line["path"] == "/api/health"]
    assert health and all(line["level"] == "DEBUG" for line in health)


# --- unexpected errors -------------------------------------------------------------------------------


def add_route(client: TestClient, path: str, exception: Exception) -> None:
    router = APIRouter()

    @router.get(path)
    def boom() -> None:
        raise exception

    client.app.include_router(router)


def test_an_unexpected_error_gets_a_readable_answer_with_a_reference_and_no_internals(client: TestClient, logs: Collector) -> None:
    add_route(client, "/api/_test/boom", RuntimeError("internal detail: /srv/secret/path password=hunter2"))
    response = client.get("/api/_test/boom", headers={"Origin": ALLOWED_ORIGIN})
    body = response.json()

    assert response.status_code == 500
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["detail"].startswith("Something went wrong on our side.") and body["request_id"] in body["detail"]
    assert "internal detail" not in response.text and "hunter2" not in response.text
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN, "the browser must be able to read it"

    error = next(line for line in logs.lines if line["message"] == "unhandled error")
    assert error["level"] == "ERROR" and error["request_id"] == body["request_id"]
    assert "RuntimeError" in error["exception"] and "Traceback" in error["exception"]
    access = next(line for line in logs.named("request") if line["request_id"] == body["request_id"])
    assert access["status"] == 500 and access["level"] == "ERROR"


def test_a_database_outage_is_explained_as_one(client: TestClient) -> None:
    add_route(client, "/api/_test/db", OperationalError("SELECT 1", {}, Exception("connection refused")))
    response = client.get("/api/_test/db")
    assert response.status_code == 503 and "database" in response.json()["detail"]
    assert "connection refused" not in response.text and "SELECT" not in response.text


def test_friendly_error_never_echoes_the_exception() -> None:
    status, message = friendly_error(ValueError("token=abc"), "ref123456")
    assert status == 500 and "abc" not in message and "ref123456" in message


# --- the events the spec asks for ----------------------------------------------------------------------


def test_the_whole_workflow_leaves_the_required_events_with_ids_and_durations_and_nothing_private(
    user_client: TestClient, logs: Collector, source_video, marked: dict[str, Any]  # noqa: ANN001
) -> None:
    pid = marked["id"]
    assert user_client.post(f"/api/projects/{pid}/track").status_code == 202
    assert user_client.get(f"/api/projects/{pid}/tracking").status_code == 200
    assert render(user_client, marked).status_code == 202
    assert user_client.post(f"/api/projects/{pid}/export").status_code == 200

    def one(message: str) -> dict[str, Any]:
        found = logs.named(message)
        assert found, f"missing log event: {message!r}; have {sorted({line['message'] for line in logs.lines})}"
        return found[-1]

    for event in ("upload started", "upload completed", "analysis started", "tracking started", "render started"):
        line = one(event)
        assert line.get("user_id") and line.get("project_id"), event
    for event in ("analysis completed", "tracking finished", "render finished"):
        line = one(event)
        assert isinstance(line["duration_ms"], int) and line["duration_ms"] >= 0, event
        assert line.get("project_id"), event
    finished = one("render finished")
    assert finished["job_id"] and finished["user_id"] and finished["project_id"] == pid
    export = one("export link issued")
    assert export["job_id"] == finished["job_id"] and export["user_id"] == finished["user_id"] and export["project_id"] == pid

    everything = "\n".join(logs.raw)
    assert "source.mp4" not in everything, "upload file names are not logged"
    for secret in ("X-Amz-Signature", "X-Amz-Credential", "Signature=", "cf_session="):
        assert secret not in everything, secret


def test_sign_up_and_sign_in_are_logged_without_emails_or_passwords(client: TestClient, logs: Collector) -> None:
    created = register(client, email="private.person@example.com")
    client.cookies.clear()
    assert client.post("/api/auth/login", json={"email": "private.person@example.com", "password": "not the password"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "not the password"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "private.person@example.com", "password": GOOD_PASSWORD}).status_code == 200

    assert logs.named("user registered")[-1]["user_id"] == created.json()["id"]
    assert logs.named("login succeeded")[-1]["user_id"] == created.json()["id"]
    assert len(logs.named("login failed")) == 2
    everything = "\n".join(logs.raw)
    for private in ("private.person", "nobody@", "not the password", GOOD_PASSWORD):
        assert private not in everything, private


# --- workers -------------------------------------------------------------------------------------------


def test_a_task_binds_its_id_and_the_next_task_starts_clean(logs: Collector) -> None:
    _bind_task(task_id="task-1", task=type("T", (), {"name": "render_video"})())
    assert lc.current_context() == {"task_id": "task-1", "task": "render_video"}
    lc.bind_context(job_id="j1")
    logging.getLogger("app.test").info("inside")
    assert logs.named("inside")[0]["job_id"] == "j1" and logs.named("inside")[0]["task_id"] == "task-1"
    _unbind_task()
    assert lc.current_context() == {}
