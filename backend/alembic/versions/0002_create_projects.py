"""create projects

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("original_file_key", sa.String(length=512), nullable=False),
        sa.Column("output_file_key", sa.String(length=512), nullable=True),
        sa.Column("upload_id", sa.String(length=255), nullable=True),
        sa.Column("container", sa.String(length=40), nullable=True),
        sa.Column("codec", sa.String(length=40), nullable=True),
        sa.Column("audio_codec", sa.String(length=40), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("bitrate", sa.BigInteger(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_projects_user_id_users", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_projects"),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])
    op.create_index("ix_projects_status", "projects", ["status"])


def downgrade() -> None:
    op.drop_index("ix_projects_status", table_name="projects")
    op.drop_index("ix_projects_user_id", table_name="projects")
    op.drop_table("projects")
