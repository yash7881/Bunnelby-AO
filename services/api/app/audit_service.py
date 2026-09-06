from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Final, Literal, Mapping

from .database import SessionLocal
from .models import ToolRun, VerificationEvidence

logger = logging.getLogger(__name__)

# Part 10.2 Phase J: durable execution audit.
#
# Every registered capability attempt gets a tool_runs row -- reads and
# approval-required proposals included. Before this, `task_log` was written only
# by the legacy router (dead since brain-first dispatch), so the only durable
# record of anything was the approvals table, and that covers approved writes
# only.
#
# Auditing must never break a turn: a failure to record is logged and swallowed.
# An assistant that refuses to answer because its audit log is unavailable is
# worse than one that answers and reports a logging problem.

ToolRunStatus = Literal["success", "failed", "blocked", "requires_approval", "unknown"]
Verdict = Literal["verified", "failed", "uncertain", "skipped"]

MAX_SUMMARY_CHARS: Final[int] = 400
MAX_JSON_CHARS: Final[int] = 4000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump(payload: object) -> str:
    """Serialize sanitized audit data, bounded so a row cannot grow unbounded."""
    try:
        text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        text = json.dumps({"unserializable": type(payload).__name__})
    if len(text) > MAX_JSON_CHARS:
        return json.dumps({"truncated": True, "chars": len(text), "head": text[:MAX_JSON_CHARS]})
    return text


def _clip(text: str | None, limit: int = MAX_SUMMARY_CHARS) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(str(text).split())
    return cleaned[: limit - 1] + "…" if len(cleaned) > limit else cleaned


def record_tool_run(
    *,
    tool_name: str,
    tool_version: str | None,
    request_arguments: Mapping[str, Any],
    request_hash: str,
    risk_level: str,
    requires_approval: bool,
    status: ToolRunStatus,
    session_id: str | None = None,
    turn_id: str | None = None,
    approval_id: int | None = None,
    error_code: str | None = None,
    user_visible_summary: str | None = None,
    started_at: datetime | None = None,
) -> int | None:
    """Persist one capability invocation. Returns the row id, or None on failure.

    `request_arguments` must already be sanitized -- callers pass
    ToolRequest.audit_arguments(), which strips raw user text and reduces bodies
    to a length plus a fingerprint.
    """
    try:
        with SessionLocal() as db:
            row = ToolRun(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                tool_version=tool_version,
                request_json=_dump(dict(request_arguments)),
                request_hash=request_hash,
                risk_level=risk_level,
                requires_approval=bool(requires_approval),
                approval_id=approval_id,
                started_at=started_at or _now(),
                finished_at=_now(),
                status=status,
                error_code=error_code,
                user_visible_summary=_clip(user_visible_summary),
            )
            db.add(row)
            db.commit()
            return int(row.id)
    except Exception:
        # Never let auditing break a user turn.
        logger.warning("Could not record tool_run for %s", tool_name, exc_info=True)
        return None


def record_verification(
    *,
    verifier_name: str,
    verdict: Verdict,
    tool_run_id: int | None = None,
    approval_id: int | None = None,
    expected: Mapping[str, Any] | None = None,
    observed: Mapping[str, Any] | None = None,
    evidence_text: str | None = None,
) -> int | None:
    """Persist one verifier verdict plus the evidence it rested on."""
    try:
        with SessionLocal() as db:
            row = VerificationEvidence(
                tool_run_id=tool_run_id,
                approval_id=approval_id,
                verifier_name=verifier_name,
                expected_json=_dump(dict(expected)) if expected is not None else None,
                observed_json=_dump(dict(observed)) if observed is not None else None,
                verdict=verdict,
                evidence_text=_clip(evidence_text, 1000),
                created_at=_now(),
            )
            db.add(row)
            db.commit()
            return int(row.id)
    except Exception:
        logger.warning(
            "Could not record verification evidence for %s", verifier_name, exc_info=True
        )
        return None


def status_for_result(action_type: str | None) -> ToolRunStatus:
    """Map an OrchestratorResult action_type onto an audit status."""
    value = (action_type or "").strip()
    if value == "approval_required":
        return "requires_approval"
    if value == "error":
        return "failed"
    if value == "clarification_required":
        return "blocked"
    if not value:
        return "unknown"
    return "success"


__all__ = [
    "ToolRunStatus",
    "Verdict",
    "record_tool_run",
    "record_verification",
    "status_for_result",
]
