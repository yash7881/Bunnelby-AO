from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Part 10.2 Phase D. Nullable so the migration is metadata-only and so any
    # pre-10.2 row (backfilled to the legacy sentinel session) stays valid.
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    turn_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class TaskLog(Base):
    __tablename__ = "task_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class ToolRun(Base):
    """Durable audit record for one capability invocation (Part 10.2 Phase J).

    Before this table, `task_log` was written only by the legacy router (dead
    since brain-first dispatch), so reads, answers and approval proposals had no
    durable record at all and NFR-007 was satisfied only for approved writes.

    request_json holds SANITIZED typed arguments: bodies and raw user text are
    reduced to a length plus a fingerprint by ToolRequest.audit_arguments(), so
    an audit row never becomes a copy of an email or a secret.
    """

    __tablename__ = "tool_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('success', 'failed', 'blocked', 'requires_approval', 'unknown')",
            name="ck_tool_runs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    turn_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    tool_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    risk_level: Mapped[str] = mapped_column(Text, nullable=False)
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_visible_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class VerificationEvidence(Base):
    """Verifier verdict plus the evidence it rested on (Part 10.2 Phase J/K).

    A provider's own success response is captured in `observed_json`, but a
    verdict of 'verified' requires an independent read-back comparison. The
    'uncertain' verdict exists so Bunnelby can say "I attempted X but could not
    verify Y" rather than converting an attempt into a success claim.
    """

    __tablename__ = "verification_evidence"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('verified', 'failed', 'uncertain', 'skipped')",
            name="ck_verification_evidence_verdict",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    tool_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    verifier_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    expected_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )


class Approval(Base):
    """Durable human decision plus immutable execution snapshot for controlled actions."""

    __tablename__ = "approvals"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_approvals_status",
        ),
        CheckConstraint(
            "execution_state IN ('not_started', 'executing', 'completed', 'failed', 'unknown')",
            name="ck_approvals_execution_state",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    task_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    preview_content: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending", index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    execution_state: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="not_started",
    )
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_error: Mapped[str | None] = mapped_column(Text, nullable=True)
