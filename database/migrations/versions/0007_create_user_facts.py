"""create user_facts table

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-04

Personal Facts Memory V1. Purely additive. Stores explicit, user-stated
personal facts (name, father, mother, other family relations) so they
survive app restarts and new sessions, independent of the messages table's
session-scoped conversational memory. One row per relation key.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("source_session_id", sa.Text(), nullable=True),
        sa.Column("source_turn_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_user_facts_id"), "user_facts", ["id"], unique=False)
    op.create_index(
        op.f("ix_user_facts_relation"), "user_facts", ["relation"], unique=True
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_user_facts_relation"), table_name="user_facts")
    op.drop_index(op.f("ix_user_facts_id"), table_name="user_facts")
    op.drop_table("user_facts")
