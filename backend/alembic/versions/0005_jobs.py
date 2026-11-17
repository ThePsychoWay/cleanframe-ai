"""background jobs

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("queue", sa.String(length=20), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("frames_done", sa.Integer(), nullable=False),
        sa.Column("frames_total", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("engine", sa.String(length=20), nullable=False),
        sa.Column("development_only", sa.Boolean(), nullable=False),
        sa.Column("tracking_used", sa.Boolean(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("output_file_key", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconstruct_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], name="fk_jobs_project_id_projects", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_jobs_user_id_users", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
    )
    op.create_index("ix_jobs_project_id", "jobs", ["project_id"])
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_user_id", table_name="jobs")
    op.drop_index("ix_jobs_project_id", table_name="jobs")
    op.drop_table("jobs")
