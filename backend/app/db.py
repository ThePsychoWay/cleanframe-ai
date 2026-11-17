"""Database engine and session dependency."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import ConfigError, get_settings


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if not url:
        raise ConfigError("DATABASE_URL is not set.")
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        # SQLite ignores foreign keys unless asked. Turn them on so development and tests behave
        # like PostgreSQL, including ON DELETE CASCADE.
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=15000")  # another thread writing: wait for it instead of failing
            if ":memory:" not in url:
                cursor.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer (local mode: API plus job threads)
            cursor.close()

    return engine


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    with get_sessionmaker()() as session:
        yield session
