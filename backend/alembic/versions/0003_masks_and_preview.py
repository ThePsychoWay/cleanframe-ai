"""masks and editor preview media

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("proxy_file_key", sa.String(length=512), nullable=True))
    op.add_column("projects", sa.Column("thumbnail_file_key", sa.String(length=512), nullable=True))
    op.add_column("projects", sa.Column("proxy_width", sa.Integer(), nullable=True))
    op.add_column("projects", sa.Column("proxy_height", sa.Integer(), nullable=True))

    op.create_table(
        "masks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("mask_data", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("frame_start", sa.Integer(), nullable=False),
        sa.Column("frame_end", sa.Integer(), nullable=False),
        sa.Column("tracking_status", sa.String(length=20), server_default="none", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], name="fk_masks_project_id_projects", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_masks"),
    )
    op.create_index("ix_masks_project_id", "masks", ["project_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_masks_project_id", table_name="masks")
    op.drop_table("masks")
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("proxy_height")
        batch.drop_column("proxy_width")
        batch.drop_column("thumbnail_file_key")
        batch.drop_column("proxy_file_key")
