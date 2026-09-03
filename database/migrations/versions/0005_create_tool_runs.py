"""create tool_runs table

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-03

Part 10.2 Phase J. Purely additive: a new table, so no existing row is read or
rewritten. The historical task_log table is deliberately left in place with its
254 legacy rows -- it was written only by the legacy router and is retained as
history rather than migrated.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("turn_id", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("tool_version", sa.Text(), nullable=True),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("risk_level", sa.Text(), nullable=False),
        sa.Column("requires_approval", sa.Boolean(), nullable=False),
        sa.Column("approval_id", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("user_visible_summary", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('success', 'failed', 'blocked', 'requires_approval', 'unknown')",
            name="ck_tool_runs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tool_runs_id"), "tool_runs", ["id"], unique=False)
    op.create_index(
        op.f("ix_tool_runs_session_id"), "tool_runs", ["session_id", "id"], unique=False
    )
    op.create_index(op.f("ix_tool_runs_turn_id"), "tool_runs", ["turn_id"], unique=False)
    op.create_index(op.f("ix_tool_runs_tool_name"), "tool_runs", ["tool_name"], unique=False)
    op.create_index(op.f("ix_tool_runs_status"), "tool_runs", ["status"], unique=False)
    op.create_index(
        op.f("ix_tool_runs_request_hash"), "tool_runs", ["request_hash"], unique=False
    )
    op.create_index(
        op.f("ix_tool_runs_approval_id"), "tool_runs", ["approval_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_runs_approval_id"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_request_hash"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_status"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_tool_name"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_turn_id"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_session_id"), table_name="tool_runs")
    op.drop_index(op.f("ix_tool_runs_id"), table_name="tool_runs")
    op.drop_table("tool_runs")
