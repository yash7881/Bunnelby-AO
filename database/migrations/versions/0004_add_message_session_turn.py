"""add session_id and turn_id to messages

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-03

Part 10.2 Phase D. Before this revision the messages table had no
conversational boundary at all, so memory retrieval scanned the last N rows
globally and a fresh app or voice-runtime launch inherited the previous
session's final exchange as its "current active topic".

Both columns are added nullable so SQLite performs a metadata-only ALTER
rather than rewriting the table, then historical rows are backfilled with a
single sentinel session. Nothing about the approvals table is touched.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Every pre-Part-10.2 message belongs to one archival session. Retrieval scopes
# to the active session, so this keeps old history reachable by explicit
# long-term recall later without it ever becoming the current active topic.
LEGACY_SESSION_ID = "legacy-pre-10.2"


def upgrade() -> None:
    op.add_column("messages", sa.Column("session_id", sa.Text(), nullable=True))
    op.add_column("messages", sa.Column("turn_id", sa.Text(), nullable=True))

    messages = sa.table(
        "messages",
        sa.column("id", sa.Integer),
        sa.column("session_id", sa.Text),
        sa.column("turn_id", sa.Text),
    )
    # Backfill history: one archival session, and a per-row turn id so the
    # user/assistant pairing logic has a stable key even for legacy rows.
    op.execute(
        messages.update()
        .where(messages.c.session_id.is_(None))
        .values(session_id=LEGACY_SESSION_ID)
    )
    op.execute(
        sa.text(
            "UPDATE messages SET turn_id = 'legacy-turn-' || id WHERE turn_id IS NULL"
        )
    )

    op.create_index(
        op.f("ix_messages_session_id"), "messages", ["session_id", "id"], unique=False
    )
    op.create_index(op.f("ix_messages_turn_id"), "messages", ["turn_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_messages_turn_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_session_id"), table_name="messages")
    op.drop_column("messages", "turn_id")
    op.drop_column("messages", "session_id")
