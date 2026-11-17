"""automatic detection: what was found, and where a mask came from

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("automation", sa.JSON().with_variant(sa.dialects.postgresql.JSONB(), "postgresql"), nullable=True))
    op.add_column("masks", sa.Column("source", sa.String(length=10), nullable=False, server_default="manual"))


def downgrade() -> None:
    op.drop_column("masks", "source")
    op.drop_column("projects", "automation")
