"""mask tracking state and results

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-03
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("masks", sa.Column("tracking_progress", sa.Integer(), server_default="0", nullable=False))
    op.add_column("masks", sa.Column("tracking_total", sa.Integer(), server_default="0", nullable=False))
    op.add_column("masks", sa.Column("tracking_error", sa.String(length=500), nullable=True))
    op.add_column("masks", sa.Column("tracking_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "masks", sa.Column("tracking_data", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True)
    )


def downgrade() -> None:
    with op.batch_alter_table("masks") as batch:
        batch.drop_column("tracking_data")
        batch.drop_column("tracking_started_at")
        batch.drop_column("tracking_error")
        batch.drop_column("tracking_total")
        batch.drop_column("tracking_progress")
