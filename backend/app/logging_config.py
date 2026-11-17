"""Structured logging for the API and the workers.

One JSON object per line (or readable text in development). Every line carries the ids that were bound for the
current request or job (request_id, user_id, project_id, job_id, task_id), plus whatever the call site passed in
`extra=`. Nothing about the video itself is ever logged: no frames, no file names, no content.

Defence in depth against leaks, because a log line is easy to get wrong:
* fields whose name looks like a secret (password, token, cookie, signature ...) are replaced;
* upload file names are dropped;
* in every string, the query part of a URL is removed (signed storage links carry their credentials there) and
  `X-Amz-*` / `Signature=` parameters are redacted wherever they appear, including inside exception messages.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# --- context: ids attached to every line logged while handling one request or job ----------------------------------

# A mutable dict held in a context variable: code that runs in a worker thread (FastAPI runs sync routes there)
# works on a copy of the context, but the copy points at this same dict, so ids bound anywhere are seen everywhere.
_context: ContextVar[dict[str, str] | None] = ContextVar("log_context", default=None)


def start_context(**values: object) -> dict[str, str]:
    fresh: dict[str, str] = {}
    _context.set(fresh)
    bind_context(**values)
    return fresh


def bind_context(**values: object) -> None:
    current = _context.get()
    if current is None:
        current = start_context()
    current.update({key: str(value) for key, value in values.items() if value is not None})


def clear_context() -> None:
    _context.set(None)


def current_context() -> dict[str, str]:
    return dict(_context.get() or {})


# --- scrubbing --------------------------------------------------------------------------------------------------

SENSITIVE_KEY = re.compile(r"pass(word)?|secret|token|authorization|cookie|api[_-]?key|signature|credential", re.IGNORECASE)
NEVER_LOGGED = frozenset({"filename", "original_filename"})  # what the person called their upload
_AMZ_PARAM = re.compile(r"((?:X-Amz-[A-Za-z0-9-]+|Signature|AWSAccessKeyId)=)[^&\s\"'<>]+", re.IGNORECASE)
_URL_QUERY = re.compile(r"(https?://[^\s?\"'<>]+)\?[^\s\"'<>]*")
MAX_VALUE_CHARS = 2000
MAX_TRACEBACK_CHARS = 8000


def scrub(text: str) -> str:
    text = _AMZ_PARAM.sub(r"\1[redacted]", text)
    return _URL_QUERY.sub(r"\1?[query removed]", text)


def _clean_value(key: str, value: Any) -> Any:
    if key in NEVER_LOGGED:
        return None
    if SENSITIVE_KEY.search(key):
        return "[redacted]"
    if value is None or isinstance(value, bool | int | float):
        return value
    return scrub(str(value))[:MAX_VALUE_CHARS]


_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


def fields_of(record: logging.LogRecord) -> dict[str, Any]:
    """Context ids first, then the call site's own fields (which win), all cleaned."""
    fields: dict[str, Any] = dict(current_context())
    for key, value in record.__dict__.items():
        if key not in _STANDARD and not key.startswith("_"):
            fields[key] = value
    cleaned = {key: _clean_value(key, value) for key, value in fields.items()}
    return {key: value for key, value in cleaned.items() if key not in NEVER_LOGGED}


def _exception_text(record: logging.LogRecord) -> str | None:
    if not record.exc_info or record.exc_info[0] is None:
        return None
    return scrub("".join(traceback.format_exception(*record.exc_info)))[-MAX_TRACEBACK_CHARS:]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage())[:MAX_VALUE_CHARS],
        }
        entry.update(fields_of(record))
        exception = _exception_text(record)
        if exception:
            entry["exception"] = exception
        return json.dumps(entry, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Readable lines for a developer's terminal. Same scrubbing as the JSON output."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        extras = " ".join(f"{key}={value}" for key, value in fields_of(record).items() if value is not None)
        line = f"{stamp} {record.levelname:<7} {record.name}: {scrub(record.getMessage())}"
        if extras:
            line += f"  [{extras}]"
        exception = _exception_text(record)
        return f"{line}\n{exception}" if exception else line


_HANDLER_MARK = "_cleanframe_handler"


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Send all logging to stderr in one format. Safe to call more than once (it replaces its own handler only)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
    root.setLevel(level.upper() if level.upper() in logging.getLevelNamesMapping() else "INFO")
    # The app writes its own access line (with a request id, without query strings); uvicorn's would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
