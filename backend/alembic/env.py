"""Alembic environment. The database URL is read from the app settings (DATABASE_URL)."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from app.config import get_settings
from app.models import Base

config = context.config
if config.config_file_name is not None:
    # Don't silence the application's loggers when migrations run inside a process that already configured logging.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    # Tests and tooling may set sqlalchemy.url explicitly; otherwise use the app settings.
    explicit = config.get_main_option("sqlalchemy.url")
    url = explicit or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
