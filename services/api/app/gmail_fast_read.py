from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from .gmail_service import (
    DEFAULT_UNREAD_MAX_RESULTS,
    GmailServiceError,
    _execute,
    _gmail_service,
    _headers,
    _message_to_email,
)

logger = logging.getLogger(__name__)

MAX_FAST_GMAIL_ITEMS = 10
MAX_FAST_GMAIL_SCAN_ITEMS = 50
_TODAY_RE = re.compile(r"\btoday(?:['’]s)?\b", re.IGNORECASE)
_UNREAD_RE = re.compile(r"\bunread\b", re.IGNORECASE)


def _message_payload_to_email(message_id: str, message: Mapping[str, object]) -> dict[str, str]:
    payload = message.get("payload")
    headers = _headers(dict(payload) if isinstance(payload, Mapping) else {})
    internal_date = str(message.get("internalDate", "")).strip()
    timestamp = headers.get("date", "")
    if internal_date.isdigit():
        try:
            timestamp = datetime.fromtimestamp(
                int(internal_date) / 1000,
                tz=timezone.utc,
            ).isoformat()
        except (OSError, OverflowError, ValueError):
            pass

    return {
        "id": message_id,
        "thread_id": str(message.get("threadId", "")).strip(),
        "sender": headers.get("from", "Unknown sender"),
        "subject": headers.get("subject", "(no subject)"),
        "snippet": str(message.get("snippet", "")).strip(),
        "timestamp": timestamp,
    }


def _batch_message_metadata(service: Any, message_ids: list[str]) -> list[dict[str, str]]:
    """Fetch Gmail metadata in one HTTP batch, with safe sequential fallback.

    Gmail's messages.list endpoint returns IDs only. The older path then performed one
    network round trip per message. This helper keeps exactly the same metadata fields but
    groups those read-only GET requests into a single Google API batch. If batch transport
    is unavailable or one subrequest fails, the proven sequential helper is used for the
    affected messages so correctness is preserved rather than traded for speed.
    """
    if not message_ids:
        return []

    responses: dict[str, Mapping[str, object]] = {}
    failed_ids: set[str] = set()

    def callback(request_id: str, response: object, exception: BaseException | None) -> None:
        if exception is not None or not isinstance(response, Mapping):
            failed_ids.add(request_id)
            return
        responses[request_id] = response

    try:
        batch = service.new_batch_http_request()
        for message_id in message_ids:
            request = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
            )
            batch.add(request, callback=callback, request_id=message_id)
        batch.execute()
    except Exception as exc:
        logger.warning("Gmail metadata batch unavailable; using sequential reads: %s", exc)
        return [_message_to_email(service, message_id) for message_id in message_ids]

    emails: list[dict[str, str]] = []
    for message_id in message_ids:
        response = responses.get(message_id)
        if response is not None and message_id not in failed_ids:
            emails.append(_message_payload_to_email(message_id, response))
            continue
        # Preserve the old failure semantics for any failed subrequest: retry through the
        # established helper, which translates Gmail HTTP/auth/rate-limit errors correctly.
        emails.append(_message_to_email(service, message_id))
    return emails


def _fetch_fast_messages(*, max_results: int, unread_only: bool) -> list[dict[str, str]]:
    service = _gmail_service()
    label_ids = ["INBOX"]
    if unread_only:
        label_ids.append("UNREAD")

    result = _execute(
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=label_ids,
            maxResults=max(1, min(max_results, 100)),
        )
    )
    refs = result.get("messages", []) or []
    message_ids = [
        str(ref.get("id", "")).strip()
        for ref in refs
        if isinstance(ref, Mapping) and str(ref.get("id", "")).strip()
    ]
    return _batch_message_metadata(service, message_ids)


def _parse_message_local_date(timestamp: str) -> object | None:
    raw = timestamp.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone().date()


def gmail_read_fast_executor(
    instruction: str,
    _context: Mapping[str, object],
) -> Mapping[str, object]:
    """Cross-tool Gmail read with the same visible contract as the established executor."""
    unread_only = bool(_UNREAD_RE.search(instruction))
    today_only = bool(_TODAY_RE.search(instruction))

    if unread_only:
        emails = _fetch_fast_messages(
            max_results=DEFAULT_UNREAD_MAX_RESULTS,
            unread_only=True,
        )
    else:
        emails = _fetch_fast_messages(
            max_results=MAX_FAST_GMAIL_SCAN_ITEMS if today_only else MAX_FAST_GMAIL_ITEMS,
            unread_only=False,
        )

    if today_only:
        today = datetime.now().astimezone().date()
        emails = [
            item
            for item in emails
            if _parse_message_local_date(str(item.get("timestamp", ""))) == today
        ]

    safe_items = [
        {
            "sender": str(item.get("sender", "Unknown sender")),
            "subject": str(item.get("subject", "(no subject)")),
            "timestamp": str(item.get("timestamp", "")),
            "snippet": str(item.get("snippet", ""))[:500],
        }
        for item in emails[:MAX_FAST_GMAIL_ITEMS]
    ]
    return {
        "unread_only": unread_only,
        "today_only": today_only,
        "count": len(safe_items),
        "emails": safe_items,
    }
