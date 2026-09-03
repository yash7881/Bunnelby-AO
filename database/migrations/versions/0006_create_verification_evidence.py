"""create verification_evidence table

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-03

Part 10.2 Phase J/K. Purely additive. The approvals table is not touched: its
immutable payload snapshots and unique idempotency keys remain the source of
truth for what a write was authorized to do, and this table records only what
was independently OBSERVED afterwards.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "verification_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tool_run_id", sa.Integer(), nullable=True),
        sa.Column("approval_id", sa.Integer(), nullable=True),
        sa.Column("verifier_name", sa.Text(), nullable=False),
        sa.Column("expected_json", sa.Text(), nullable=True),
        sa.Column("observed_json", sa.Text(), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "verdict IN ('verified', 'failed', 'uncertain', 'skipped')",
            name="ck_verification_evidence_verdict",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_verification_evidence_id"), "verification_evidence", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_verification_evidence_tool_run_id"),
        "verification_evidence",
        ["tool_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verification_evidence_approval_id"),
        "verification_evidence",
        ["approval_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verification_evidence_verifier_name"),
        "verification_evidence",
        ["verifier_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verification_evidence_verdict"),
        "verification_evidence",
        ["verdict"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verification_evidence_created_at"),
        "verification_evidence",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    for index in (
        "ix_verification_evidence_created_at",
        "ix_verification_evidence_verdict",
        "ix_verification_evidence_verifier_name",
        "ix_verification_evidence_approval_id",
        "ix_verification_evidence_tool_run_id",
        "ix_verification_evidence_id",
    ):
        op.drop_index(op.f(index), table_name="verification_evidence")
    op.drop_table("verification_evidence")
