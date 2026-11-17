import os
from pathlib import Path

import pytest

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from app.models import Base

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _config(url: str) -> Config:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


POSTGRES_URL = os.environ.get("TEST_DATABASE_URL", "")


@pytest.mark.parametrize("backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(not POSTGRES_URL, reason="set TEST_DATABASE_URL to a PostgreSQL address"))])
def test_migrations_match_models_and_downgrade_cleanly(tmp_path: Path, backend: str) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'migrate.db'}" if backend == "sqlite" else POSTGRES_URL
    if backend == "postgres":
        _empty(url)
    cfg = _config(url)

    command.upgrade(cfg, "head")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            drift = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        assert drift == [], f"models and migrations disagree: {drift}"

        command.downgrade(cfg, "base")
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    finally:
        engine.dispose()


def _empty(url: str) -> None:
    """Start the PostgreSQL test database from nothing, so the migrations are what builds the schema."""
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()
