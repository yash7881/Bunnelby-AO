from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Mapping

from . import audit_service, calendar_service, gmail_service
from .tool_requests import (
    CalendarReadRequest,
    CrossToolReadRequest,
    GmailReadRequest,
    ToolRequest,
)

logger = logging.getLogger(__name__)

# Part 10.2 Phase K: deterministic verification.
#
# The governing rule (PRD 28): a provider's own success response is not by
# itself proof that external state changed. Where a read-back is practical, the
# verifier performs one and compares it against the EXPECTED values taken from
# the approval's immutable payload snapshot -- never from the raw user text and
# never from the model.
#
# 'uncertain' is a first-class verdict so Bunnelby can say "I attempted X but
# could not verify Y" instead of converting an attempt into a success claim.
# Nothing here retries: a failed or uncertain verdict is reported, not repaired.

Verdict = Literal["verified", "failed", "uncertain", "skipped"]

MAX_EVIDENCE_CHARS: Final[int] = 800


@dataclass(frozen=True)
class VerificationResult:
    verifier_name: str
    verdict: Verdict
    expected: Mapping[str, Any] = field(default_factory=dict)
    observed: Mapping[str, Any] = field(default_factory=dict)
    evidence_text: str = ""

    @property
    def verified(self) -> bool:
        return self.verdict == "verified"


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_address(value: object) -> str:
    """Reduce a header address to a comparable bare mailbox."""
    text = str(value or "").strip().lower()
    if "<" in text and ">" in text:
        text = text[text.rfind("<") + 1 : text.rfind(">")]
    return text.strip()


def _normalize_subject(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


# --------------------------------------------------------------------------- #
# Read verifiers: confirm an observation is real, bounded and sourced
# --------------------------------------------------------------------------- #


def verify_gmail_read(request: GmailReadRequest, result: Any) -> VerificationResult:
    """Confirm a Gmail read produced a bounded, attributable observation."""
    reply = str(getattr(result, "reply", "") or "")
    metadata = dict(getattr(result, "spoken_metadata", {}) or {})
    count = metadata.get("email_count")
    expected = {
        "read_kind": request.read_kind,
        "limit": request.limit,
        "unread_only": request.unread_only,
    }
    observed = {"email_count": count, "reply_chars": len(reply)}

    if getattr(result, "action_type", None) == "error":
        return VerificationResult(
            "GmailReadVerifier", "failed", expected, observed, "the read reported an error"
        )
    if count is None:
        return VerificationResult(
            "GmailReadVerifier",
            "uncertain",
            expected,
            observed,
            "no email_count was reported, so the observation cannot be bounded",
        )
    if isinstance(count, int) and count > request.limit:
        return VerificationResult(
            "GmailReadVerifier",
            "failed",
            expected,
            observed,
            f"returned {count} messages for a limit of {request.limit}",
        )
    if metadata.get("unread_only") != request.unread_only:
        return VerificationResult(
            "GmailReadVerifier",
            "failed",
            expected,
            observed,
            "the executed read kind does not match the requested read kind",
        )
    return VerificationResult(
        "GmailReadVerifier",
        "verified",
        expected,
        observed,
        f"{count} message(s) within the requested bound, matching read_kind",
    )


def verify_calendar_read(request: CalendarReadRequest, result: Any) -> VerificationResult:
    """Confirm a Calendar read stayed a read and produced an answer."""
    action_type = getattr(result, "action_type", None)
    expected = {"mode": request.mode, "class": "read"}
    observed = {"action_type": action_type, "has_approval": bool(getattr(result, "approval", None))}

    if observed["has_approval"]:
        # Structurally unreachable after Phase G; verified here anyway because
        # this is precisely the invariant that used to be violable.
        return VerificationResult(
            "CalendarReadVerifier",
            "failed",
            expected,
            observed,
            "a calendar read produced an approval proposal",
        )
    if action_type == "error":
        return VerificationResult(
            "CalendarReadVerifier", "failed", expected, observed, "the read reported an error"
        )
    if action_type in {"clarification_required"}:
        return VerificationResult(
            "CalendarReadVerifier",
            "uncertain",
            expected,
            observed,
            "the read could not resolve a date and asked for clarification",
        )
    return VerificationResult(
        "CalendarReadVerifier",
        "verified",
        expected,
        observed,
        "calendar read returned a read-class result with no proposal",
    )


def verify_cross_tool_read(request: CrossToolReadRequest, result: Any) -> VerificationResult:
    """Confirm every requested source actually contributed, and nothing was written."""
    metadata = dict(getattr(result, "spoken_metadata", {}) or {})
    expected = {"sources": list(request.sources), "class": "read"}
    observed = {
        "steps_total": metadata.get("steps_total"),
        "steps_succeeded": metadata.get("steps_succeeded"),
        "steps_failed": metadata.get("steps_failed"),
        "has_approval": bool(getattr(result, "approval", None)),
    }

    if observed["has_approval"]:
        return VerificationResult(
            "CrossToolReadVerifier",
            "failed",
            expected,
            observed,
            "a cross-tool read produced an approval proposal",
        )
    succeeded = observed["steps_succeeded"]
    if not isinstance(succeeded, int) or succeeded <= 0:
        return VerificationResult(
            "CrossToolReadVerifier",
            "uncertain",
            expected,
            observed,
            "no source step reported success, so the combined answer is unproven",
        )
    if isinstance(observed["steps_failed"], int) and observed["steps_failed"] > 0:
        return VerificationResult(
            "CrossToolReadVerifier",
            "uncertain",
            expected,
            observed,
            f"{observed['steps_failed']} source step(s) failed; the answer is partial",
        )
    return VerificationResult(
        "CrossToolReadVerifier",
        "verified",
        expected,
        observed,
        f"{succeeded} source step(s) succeeded with no failures",
    )


READ_VERIFIERS: Final[Mapping[str, Any]] = {
    "gmail_read": verify_gmail_read,
    "calendar_read": verify_calendar_read,
    "cross_tool_read": verify_cross_tool_read,
}


def verify_read(request: ToolRequest, result: Any) -> VerificationResult | None:
    """Run the read verifier for a request, if one is registered."""
    verifier = READ_VERIFIERS.get(request.tool_name)
    if verifier is None:
        return None
    try:
        return verifier(request, result)
    except Exception as exc:
        logger.warning("Read verifier for %s raised: %s", request.tool_name, exc, exc_info=True)
        return VerificationResult(
            f"{request.tool_name}_verifier",
            "uncertain",
            {},
            {},
            f"verifier raised {type(exc).__name__}",
        )


# --------------------------------------------------------------------------- #
# External read-back for approved writes
# --------------------------------------------------------------------------- #


def _gmail_sent_headers(message_id: str) -> dict[str, str]:
    """Read one sent message back from Gmail and return its key headers."""
    service = gmail_service._gmail_service()
    message = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata",
             metadataHeaders=["To", "Subject", "Message-Id"])
        .execute()
    )
    headers = {
        str(h.get("name", "")).lower(): str(h.get("value", ""))
        for h in (message.get("payload", {}).get("headers") or [])
    }
    return {
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "message_id": str(message.get("id", "")),
        "thread_id": str(message.get("threadId", "")),
        "label_ids": ",".join(message.get("labelIds") or []),
        "snippet": str(message.get("snippet", ""))[:200],
    }


def verify_gmail_send(
    payload: Mapping[str, Any], completion: Mapping[str, Any]
) -> VerificationResult:
    """Read the sent message back and compare it against the approval snapshot.

    `payload` is the approval's immutable payload snapshot; `completion` is what
    the send reported. Expected values come from the snapshot only.
    """
    expected = {
        "recipient": _normalize_address(payload.get("to")),
        "subject": _normalize_subject(payload.get("subject")),
        "body_fingerprint": _fingerprint(str(payload.get("body", ""))),
    }
    message_id = str(completion.get("gmail_message_id", "")).strip()
    if not message_id:
        return VerificationResult(
            "GmailSendVerifier",
            "uncertain",
            expected,
            {},
            "Gmail returned no message id, so the send cannot be read back",
        )

    try:
        observed_headers = _gmail_sent_headers(message_id)
    except Exception as exc:
        logger.warning("GmailSendVerifier read-back failed: %s", exc, exc_info=True)
        return VerificationResult(
            "GmailSendVerifier",
            "uncertain",
            expected,
            {"message_id": message_id},
            f"read-back failed ({type(exc).__name__}); the send is unverified, not failed",
        )

    observed = {
        "recipient": _normalize_address(observed_headers.get("to")),
        "subject": _normalize_subject(observed_headers.get("subject")),
        "message_id": observed_headers.get("message_id", ""),
        "label_ids": observed_headers.get("label_ids", ""),
    }

    mismatches = [
        field
        for field in ("recipient", "subject")
        if expected[field] and expected[field] != observed[field]
    ]
    if mismatches:
        return VerificationResult(
            "GmailSendVerifier",
            "failed",
            expected,
            observed,
            f"read-back mismatch on {', '.join(mismatches)}",
        )
    return VerificationResult(
        "GmailSendVerifier",
        "verified",
        expected,
        observed,
        f"message {observed['message_id']} read back with matching recipient and subject",
    )


def _calendar_event(calendar_id: str, event_id: str) -> dict[str, Any]:
    service = calendar_service._calendar_service()
    return (
        service.events().get(calendarId=calendar_id or "primary", eventId=event_id).execute()
    )


def _event_instant(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("dateTime") or value.get("date") or "")
    return str(value or "")


def verify_calendar_create(
    payload: Mapping[str, Any], completion: Mapping[str, Any]
) -> VerificationResult:
    """Read the created event back and compare it against the approval snapshot."""
    expected = {
        "title": _normalize_subject(payload.get("title")),
        "start": str(payload.get("start", "")),
        "end": str(payload.get("end", "")),
        "attendees": sorted(
            _normalize_address(a) for a in (payload.get("attendees") or [])
        ),
    }
    event_id = str(completion.get("calendar_event_id", "")).strip()
    if not event_id:
        return VerificationResult(
            "CalendarCreateVerifier",
            "uncertain",
            expected,
            {},
            "Calendar returned no event id, so the creation cannot be read back",
        )

    try:
        event = _calendar_event(str(payload.get("calendar_id") or "primary"), event_id)
    except Exception as exc:
        logger.warning("CalendarCreateVerifier read-back failed: %s", exc, exc_info=True)
        return VerificationResult(
            "CalendarCreateVerifier",
            "uncertain",
            expected,
            {"event_id": event_id},
            f"read-back failed ({type(exc).__name__}); the event is unverified, not failed",
        )

    observed = {
        "title": _normalize_subject(event.get("summary")),
        "start": _event_instant(event.get("start")),
        "end": _event_instant(event.get("end")),
        "attendees": sorted(
            _normalize_address(a.get("email")) for a in (event.get("attendees") or [])
        ),
        "event_id": str(event.get("id", "")),
        "status": str(event.get("status", "")),
    }

    if observed["status"] == "cancelled":
        return VerificationResult(
            "CalendarCreateVerifier",
            "failed",
            expected,
            observed,
            "the event exists but is cancelled",
        )

    mismatches: list[str] = []
    if expected["title"] and expected["title"] != observed["title"]:
        mismatches.append("title")
    for field in ("start", "end"):
        # Compare to minute precision: providers normalize offsets and seconds.
        if expected[field][:16] and expected[field][:16] != observed[field][:16]:
            mismatches.append(field)
    if expected["attendees"] and expected["attendees"] != observed["attendees"]:
        mismatches.append("attendees")

    if mismatches:
        return VerificationResult(
            "CalendarCreateVerifier",
            "failed",
            expected,
            observed,
            f"read-back mismatch on {', '.join(mismatches)}",
        )
    return VerificationResult(
        "CalendarCreateVerifier",
        "verified",
        expected,
        observed,
        f"event {observed['event_id']} read back with matching title and window",
    )


WRITE_VERIFIERS: Final[Mapping[str, Any]] = {
    "gmail_compose": verify_gmail_send,
    "gmail_reply": verify_gmail_send,
    "calendar_event": verify_calendar_create,
}


def verify_approved_execution(approval: Any, outcome: str) -> VerificationResult | None:
    """Verify one just-executed approved write, and record the evidence.

    Called after approval execution completes. It never mutates the approval and
    never retries: the approval engine's snapshot, idempotency key, CAS claim and
    failed-vs-unknown semantics are untouched by design.
    """
    if outcome not in {"sent", "created"}:
        # Nothing external is known to have happened, so there is nothing to
        # read back. already_sent/already_created were verified on first execution.
        return None

    task_type = str(getattr(approval, "task_type", "") or "")
    verifier = WRITE_VERIFIERS.get(task_type)
    if verifier is None:
        return None

    try:
        payload = json.loads(getattr(approval, "payload_json", "") or "{}")
        completion = json.loads(getattr(approval, "execution_result", "") or "{}")
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not decode approval %s for verification", getattr(approval, "id", "?"))
        return None

    try:
        result = verifier(payload, completion)
    except Exception as exc:
        logger.warning("Write verifier for %s raised: %s", task_type, exc, exc_info=True)
        result = VerificationResult(
            WRITE_VERIFIERS[task_type].__name__, "uncertain", {}, {},
            f"verifier raised {type(exc).__name__}",
        )

    audit_service.record_verification(
        verifier_name=result.verifier_name,
        verdict=result.verdict,
        approval_id=getattr(approval, "id", None),
        expected=result.expected,
        observed=result.observed,
        evidence_text=result.evidence_text[:MAX_EVIDENCE_CHARS],
    )
    logger.info(
        "Verification %s approval=%s verdict=%s: %s",
        result.verifier_name,
        getattr(approval, "id", "?"),
        result.verdict,
        result.evidence_text,
    )
    return result


def uncertainty_message(result: VerificationResult | None) -> str | None:
    """User-facing wording for an unverified external action (PRD 28 trust rule)."""
    if result is None or result.verdict == "verified":
        return None
    if result.verdict == "failed":
        return (
            "The action was reported as done, but reading it back did not match what "
            "you approved. Please check before relying on it."
        )
    return (
        "I attempted the action, but I could not verify it afterwards. I will not "
        "retry automatically."
    )


__all__ = [
    "READ_VERIFIERS",
    "WRITE_VERIFIERS",
    "VerificationResult",
    "Verdict",
    "uncertainty_message",
    "verify_approved_execution",
    "verify_calendar_create",
    "verify_calendar_read",
    "verify_cross_tool_read",
    "verify_gmail_read",
    "verify_gmail_send",
    "verify_read",
]
