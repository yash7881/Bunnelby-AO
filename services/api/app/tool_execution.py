from __future__ import annotations

import logging
import re

from . import gmail_service, message_dispatch
from .acknowledgments import detect_spoken_language
from .approval_service import approval_public_dict, create_gmail_reply_approval
from .calendar_service import (
    CalendarAuthorizationError,
    CalendarConfigurationError,
    CalendarParseError,
    CalendarRateLimitError,
    CalendarServiceError,
    _is_open_slot_request,
)
from .orchestrator import OrchestratorResult
from .tool_requests import (
    CalendarCreateRequest,
    CalendarReadRequest,
    CrossToolReadRequest,
    GeneralAnswerRequest,
    GmailComposeRequest,
    GmailReadRequest,
    GmailReplyRequest,
    FileSearchRequest,
)

logger = logging.getLogger(__name__)

# Part 10.2 Phase G: typed, class-scoped execution.
#
# Each executor below serves exactly ONE capability class. The class is decided
# once by the Brain and carried in the request TYPE; an executor may only refine
# fields inside that class.
#
# Structurally there is no path from a *_read executor to an approval builder,
# and none from a *_create / *_compose / *_reply executor to an inbox summary or
# a free/busy answer. That is what closes the four audited split-brain defects:
#
#   before: tool_executor:106-112  calendar_read AND calendar_create both called
#           message_dispatch._calendar_result(user_message) -- identical call,
#           and parse_calendar_request then re-chose the action from wording.
#   before: tool_executor:81/92    gmail_read AND gmail_reply both called
#           orchestrator.gmail_handler(user_message) -- identical call, and
#           gmail_handler re-chose read-vs-reply from wording.
#
# Collaborators are reached through their owning module (message_dispatch.X,
# gmail_service.X) rather than being re-imported here, so the established test
# seams keep working and there is a single definition of each helper.

# An availability question ("am I free tomorrow at 5?") is free/busy, not an
# agenda listing. Used only to pick a READ submode; it can never select create.
_AVAILABILITY_RE = re.compile(
    r"\b(?:free|busy|available|availability)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #


def execute_general_answer(request: GeneralAnswerRequest) -> OrchestratorResult:
    reply = request.reply or request.raw_message
    action_type = "clarification_required" if request.is_clarification else "general_answer"
    return OrchestratorResult(
        reply=reply,
        action_type=action_type,
        memory_content=f"{reply}\nRoute: brain\nWhy: typed general_answer",
        spoken_reply=request.spoken_reply or None,
    )


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


def _calendar_read_mode(request: CalendarReadRequest) -> str:
    """Pick a READ submode. Every branch returns a read; create is unreachable."""
    text = request.raw_message

    # A Brain-supplied non-default submode is authoritative.
    if request.mode in ("free_busy", "open_slots"):
        return request.mode

    if message_dispatch.is_agenda_request(text):
        return "agenda"
    if _is_open_slot_request(text):
        return "open_slots"
    if _AVAILABILITY_RE.search(text):
        return "free_busy"
    return "agenda"


def execute_calendar_read(request: CalendarReadRequest) -> OrchestratorResult:
    """Read the calendar. Cannot mint a create approval by any path."""
    language = detect_spoken_language(request.raw_message)
    normalized = message_dispatch._normalize_calendar_time_language(request.raw_message)
    mode = _calendar_read_mode(request)

    try:
        if mode == "agenda":
            start, end, _zone = message_dispatch.agenda_range(normalized)
            events = message_dispatch.list_events((start, end))
            return message_dispatch._result(
                message_dispatch.format_agenda_response(start, events),
                spoken_reply=message_dispatch.calendar_agenda_briefing(
                    start, events, language
                ),
                action_type="calendar_read",
                route="calendar",
                reason="typed calendar_read (mode=agenda)",
            )

        # force_action pins the class inside the deterministic parser, so create
        # vocabulary inside an availability question cannot promote itself.
        parsed = message_dispatch.parse_calendar_request(
            normalized,
            force_action=mode,
            fallback_duration_minutes=request.duration_minutes,
        )

        if mode == "open_slots":
            slots = message_dispatch.find_open_slots(
                parsed.start.date(), parsed.duration_minutes, parsed.work_hours
            )
            return message_dispatch._result(
                message_dispatch.format_open_slots_response(parsed, slots),
                spoken_reply=message_dispatch.calendar_open_slots_briefing(
                    parsed, slots, language
                ),
                action_type="calendar_read",
                route="calendar",
                reason="typed calendar_read (mode=open_slots)",
            )

        busy = message_dispatch.check_free_busy((parsed.start, parsed.end))
        return message_dispatch._result(
            message_dispatch.format_free_busy_response(parsed, busy),
            spoken_reply=message_dispatch.calendar_free_busy_briefing(
                parsed, busy, language
            ),
            action_type="calendar_read",
            route="calendar",
            reason="typed calendar_read (mode=free_busy)",
        )
    except CalendarParseError as exc:
        return message_dispatch._result(
            str(exc),
            spoken_reply=(
                "शेड्यूल देखने के लिए तारीख थोड़ी और स्पष्ट बताइए।"
                if language == "hi"
                else "I need a clearer date before I can read your schedule."
            ),
            action_type="error",
            route="calendar",
            reason="calendar read date requires clarification",
        )
    except CalendarConfigurationError as exc:
        return message_dispatch._result(
            f"Bunnelby Calendar setup is incomplete: {exc}",
            spoken_reply="Calendar setup is incomplete. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar configuration error",
        )
    except CalendarAuthorizationError as exc:
        return message_dispatch._result(
            f"Bunnelby could not authorize Google Calendar: {exc}",
            spoken_reply="Google Calendar authorization is required. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar authorization error",
        )
    except CalendarRateLimitError:
        return message_dispatch._result(
            "Google Calendar API rate limit reached. Please retry in a moment.",
            spoken_reply="Google Calendar is temporarily rate-limited. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar rate limit",
        )
    except CalendarServiceError as exc:
        logger.warning("Typed calendar read failed: %s", exc)
        return message_dispatch._result(
            f"Bunnelby could not read your Google Calendar: {exc}",
            spoken_reply="I couldn't read your Google Calendar right now.",
            action_type="error",
            route="calendar",
            reason="calendar service error",
        )


def execute_calendar_create(request: CalendarCreateRequest) -> OrchestratorResult:
    """Propose a calendar event. Never decays into a free/busy read."""
    language = detect_spoken_language(request.raw_message)
    normalized = message_dispatch._normalize_calendar_time_language(request.raw_message)

    try:
        parsed = message_dispatch.parse_calendar_request(
            normalized,
            force_action="create_event",
            fallback_title=request.title,
            fallback_duration_minutes=request.duration_minutes,
        )
    except CalendarParseError as exc:
        # Fail closed to clarification. Before Phase G, absent create vocabulary
        # silently produced an availability answer instead of asking.
        return message_dispatch._result(
            str(exc),
            spoken_reply=(
                "इवेंट बनाने के लिए सही समय बताइए।"
                if language == "hi"
                else "I need the exact start time before I can prepare that event."
            ),
            action_type="clarification_required",
            route="calendar",
            reason="calendar create requires an exact time or title",
        )
    except CalendarConfigurationError as exc:
        return message_dispatch._result(
            f"Bunnelby Calendar setup is incomplete: {exc}",
            spoken_reply="Calendar setup is incomplete. Nothing was created.",
            action_type="error",
            route="calendar",
            reason="calendar configuration error",
        )
    except CalendarAuthorizationError as exc:
        return message_dispatch._result(
            f"Bunnelby could not authorize Google Calendar: {exc}",
            spoken_reply="Google Calendar authorization is required. Nothing was created.",
            action_type="error",
            route="calendar",
            reason="calendar authorization error",
        )

    try:
        busy = message_dispatch.check_free_busy((parsed.start, parsed.end))
        if busy:
            return message_dispatch._result(
                "That requested time overlaps an existing busy period. Nothing was created. "
                "Choose another time or ask me to find an open slot.",
                spoken_reply=(
                    "उस समय आपका कैलेंडर व्यस्त है। कुछ भी नहीं बनाया गया।"
                    if language == "hi"
                    else "That time overlaps something already on your calendar. Nothing was created."
                ),
                action_type="calendar_read",
                route="calendar",
                reason="calendar conflict check before create",
            )

        proposal = message_dispatch.calendar_event_proposal(parsed)
        approval = message_dispatch.create_calendar_event_approval(
            proposal, spoken_language=language
        )
        assumption = (
            " I used a 60-minute duration because you did not specify one; review the "
            "end time before approving."
            if parsed.assumed_duration
            else ""
        )
        return message_dispatch._result(
            "I've prepared the calendar event. Review the exact title, time, timezone, "
            "and attendees before I create anything." + assumption,
            spoken_reply=(
                "मैंने कैलेंडर इवेंट तैयार कर दिया है। बनाने से पहले आपकी मंज़ूरी चाहिए।"
                if language == "hi"
                else "I've prepared the calendar event. Review it before I create anything."
            ),
            action_type="approval_required",
            route="calendar",
            reason="calendar event requires explicit approval",
            approval=message_dispatch.approval_public_dict(approval),
        )
    except CalendarRateLimitError:
        return message_dispatch._result(
            "Google Calendar API rate limit reached. Please retry in a moment.",
            spoken_reply="Google Calendar is temporarily rate-limited. Nothing was created.",
            action_type="error",
            route="calendar",
            reason="calendar rate limit",
        )
    except CalendarServiceError as exc:
        logger.warning("Typed calendar create failed: %s", exc)
        return message_dispatch._result(
            f"Bunnelby could not prepare the Calendar event: {exc}",
            spoken_reply="I couldn't prepare that calendar event. Nothing was created.",
            action_type="error",
            route="calendar",
            reason="calendar service error",
        )


# --------------------------------------------------------------------------- #
# Gmail
# --------------------------------------------------------------------------- #


def execute_gmail_read(request: GmailReadRequest) -> OrchestratorResult:
    """Read the inbox using the typed request. Never builds an approval."""
    try:
        if request.unread_only:
            emails = gmail_service.get_unread_emails()
            empty_message = "You have no unread inbox emails."
        else:
            emails = gmail_service.get_recent_emails(max_results=request.limit)
            empty_message = "No recent inbox emails were found."

        if not emails:
            return OrchestratorResult(
                reply=empty_message,
                action_type="gmail_summary",
                memory_content=f"{empty_message}\nRoute: gmail\nWhy: typed gmail_read",
                spoken_metadata={"email_count": 0, "unread_only": request.unread_only},
            )

        reply = gmail_service.summarize_with_graceful_fallback(emails)
        return OrchestratorResult(
            reply=reply,
            action_type="gmail_summary",
            memory_content=f"{reply}\nRoute: gmail\nWhy: typed gmail_read",
            spoken_metadata={
                "email_count": len(emails),
                "unread_only": request.unread_only,
            },
        )
    except gmail_service.GmailConfigurationError as exc:
        return message_dispatch._result(
            f"Bunnelby Gmail setup is incomplete: {exc}",
            spoken_reply="Gmail setup is incomplete. No messages were changed.",
            action_type="error",
            route="gmail",
            reason="gmail configuration error",
        )
    except gmail_service.GmailAuthorizationError as exc:
        return message_dispatch._result(
            f"Bunnelby could not authorize Gmail: {exc}",
            spoken_reply="Gmail authorization failed. No messages were changed.",
            action_type="error",
            route="gmail",
            reason="gmail authorization error",
        )
    except gmail_service.GmailRateLimitError:
        return message_dispatch._result(
            "Gmail API rate limit reached. Please retry in a moment.",
            spoken_reply="Gmail is temporarily rate-limited. Nothing was changed.",
            action_type="error",
            route="gmail",
            reason="gmail rate limit",
        )
    except gmail_service.GmailSummaryError as exc:
        logger.warning("Typed Gmail summary failed: %s", exc)
        return message_dispatch._result(
            "Bunnelby read your email, but could not summarize it right now. Please retry.",
            spoken_reply="I read the inbox, but the summary failed. Your email is unaffected.",
            action_type="error",
            route="gmail",
            reason="gmail summary error",
        )
    except gmail_service.GmailServiceError as exc:
        logger.warning("Typed Gmail read failed: %s", exc)
        return message_dispatch._result(
            f"Bunnelby could not read Gmail right now: {exc}",
            spoken_reply="I couldn't read Gmail. No messages were changed.",
            action_type="error",
            route="gmail",
            reason="gmail service error",
        )


def execute_gmail_reply(request: GmailReplyRequest) -> OrchestratorResult:
    """Propose a reply to an existing thread. Never returns an inbox summary."""
    language = detect_spoken_language(request.raw_message)
    try:
        draft = gmail_service.draft_reply_from_request(request.raw_message)
        approval = create_gmail_reply_approval(draft, spoken_language=language)
        target = (
            draft.get("recipient_display") or draft.get("to") or "the selected recipient"
        )
        return message_dispatch._result(
            f"I drafted a reply to {target}. Review the exact recipient, subject, and "
            "message below. Nothing will be sent until you explicitly approve it.",
            spoken_reply=(
                "मैंने जवाब का ड्राफ्ट तैयार कर दिया है। भेजने से पहले आपकी मंज़ूरी चाहिए।"
                if language == "hi"
                else "I've drafted the reply. Review it before I send anything."
            ),
            action_type="approval_required",
            route="gmail",
            reason="typed gmail_reply requires explicit approval",
            approval=approval_public_dict(approval),
        )
    except gmail_service.GmailTargetResolutionError as exc:
        return message_dispatch._result(
            str(exc),
            spoken_reply="I need a clearer sender or subject before I can prepare the reply.",
            action_type="clarification_required",
            route="gmail",
            reason="gmail reply target requires clarification",
        )
    except gmail_service.GmailDraftError as exc:
        return message_dispatch._result(
            f"I couldn't create the Gmail reply draft: {exc}",
            spoken_reply="I couldn't prepare that reply. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail reply draft error",
        )
    except gmail_service.GmailConfigurationError as exc:
        return message_dispatch._result(
            f"Bunnelby Gmail setup is incomplete: {exc}",
            spoken_reply="Gmail setup is incomplete. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail configuration error",
        )
    except gmail_service.GmailAuthorizationError as exc:
        return message_dispatch._result(
            f"Bunnelby could not authorize Gmail: {exc}",
            spoken_reply="Gmail authorization is required. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail authorization error",
        )
    except gmail_service.GmailRateLimitError:
        return message_dispatch._result(
            "Gmail API rate limit reached. Please retry in a moment.",
            spoken_reply="Gmail is temporarily rate-limited. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail rate limit",
        )
    except gmail_service.GmailServiceError as exc:
        logger.warning("Typed Gmail reply failed: %s", exc)
        return message_dispatch._result(
            f"Bunnelby could not prepare the Gmail reply: {exc}",
            spoken_reply="I couldn't prepare that Gmail reply. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail reply service error",
        )


def execute_gmail_compose(request: GmailComposeRequest) -> OrchestratorResult:
    """Propose a brand-new email from the validated typed request."""
    language = detect_spoken_language(request.raw_message)
    try:
        # Spoken-address normalization is preserved: it repairs STT-dictated
        # addresses before deterministic recipient resolution runs.
        instruction = gmail_service.normalize_spoken_email(request.raw_message)
        draft = message_dispatch.draft_new_email_from_request(instruction)
        resolved = str(draft.get("to") or "")
        if request.recipient and resolved and request.recipient != resolved:
            logger.info(
                "Typed compose recipient hint differs from deterministic resolution; "
                "resolution is authoritative for the approval snapshot."
            )
        approval = message_dispatch.create_gmail_compose_approval(
            draft, spoken_language=language
        )
        return message_dispatch._result(
            f"I drafted a new email to {resolved}. Review the exact recipient, subject, "
            "and message below. Nothing will be sent until you explicitly approve it.",
            spoken_reply=(
                "मैंने नया ईमेल ड्राफ्ट तैयार कर दिया है। भेजने से पहले आपकी मंज़ूरी चाहिए।"
                if language == "hi"
                else "I've drafted the new email. Review it before I send anything."
            ),
            action_type="approval_required",
            route="gmail",
            reason="typed gmail_compose requires explicit approval",
            approval=message_dispatch.approval_public_dict(approval),
        )
    except gmail_service.GmailDraftError as exc:
        detail = str(exc)
        clarification = bool(
            re.search(
                r"(multiple|which one|identify|recipient|exact email|exact address|"
                r"correspondent|matching)",
                detail,
                re.IGNORECASE,
            )
        )
        return message_dispatch._result(
            detail if clarification else f"I couldn't create the new Gmail draft: {detail}",
            spoken_reply=(
                "I need one more detail to identify the correct recipient."
                if clarification
                else "I couldn't prepare that new email. Nothing was sent."
            ),
            action_type="clarification_required" if clarification else "error",
            route="gmail",
            reason=(
                "gmail recipient clarification required"
                if clarification
                else "gmail draft error"
            ),
        )
    except gmail_service.GmailConfigurationError as exc:
        return message_dispatch._result(
            f"Bunnelby Gmail setup is incomplete: {exc}",
            spoken_reply="Gmail setup is incomplete. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail configuration error",
        )
    except gmail_service.GmailAuthorizationError as exc:
        return message_dispatch._result(
            f"Bunnelby could not authorize Gmail: {exc}",
            spoken_reply="Gmail authorization is required. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail authorization error",
        )
    except gmail_service.GmailRateLimitError:
        return message_dispatch._result(
            "Gmail API rate limit reached. Please retry in a moment.",
            spoken_reply="Gmail is temporarily rate-limited. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail rate limit",
        )
    except gmail_service.GmailServiceError as exc:
        logger.warning("Typed Gmail compose failed: %s", exc)
        return message_dispatch._result(
            f"Bunnelby could not prepare the new Gmail message: {exc}",
            spoken_reply="I couldn't prepare that email. Nothing was sent.",
            action_type="error",
            route="gmail",
            reason="gmail compose service error",
        )


# --------------------------------------------------------------------------- #
# Cross-tool
# --------------------------------------------------------------------------- #


def execute_cross_tool_read(request: CrossToolReadRequest) -> OrchestratorResult:
    """Combined Gmail + Calendar read. Read-only by construction."""
    from . import tool_executor

    try:
        result = tool_executor.handle_cross_tool_fast_request(request.raw_message)
    except tool_executor.CrossToolWriteNotSupportedError as exc:
        message = str(exc)
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=message,
            spoken_reply=(
                "Combined tool actions are read-only right now, sir. Request the write "
                "separately so I can preserve the approval check."
            ),
        )
    except Exception as exc:
        logger.exception("Typed cross_tool_read execution failed: %s", exc)
        message = (
            "I couldn't complete the combined Gmail and Calendar request right now. "
            "Neither service was changed."
        )
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=message,
            spoken_reply="I couldn't complete the combined check right now, sir. Nothing was changed.",
        )

    succeeded = sum(step.status == "success" for step in result.steps)
    failed = len(result.steps) - succeeded
    status_note = f"Cross-tool plan: {succeeded} succeeded, {failed} failed."
    spoken_metadata: dict[str, object] = {
        "cross_tool": True,
        "steps_total": len(result.steps),
        "steps_succeeded": succeeded,
        "steps_failed": failed,
        "plan_source": result.plan.source,
        "sources": list(request.sources),
    }
    timings = getattr(result, "timings_ms", None)
    if timings:
        spoken_metadata["latency_ms"] = dict(timings)

    return OrchestratorResult(
        reply=result.reply,
        action_type="task_complete" if succeeded else "error",
        memory_content=f"{result.reply}\n{status_note}",
        spoken_reply=result.spoken_reply,
        spoken_metadata=spoken_metadata,
    )


def execute_file_search(request: FileSearchRequest) -> OrchestratorResult:
    """Render already-local, deterministic retrieval; no model or cloud call."""
    from .local_files.service import default_service

    envelope = default_service().search_request(request)
    if envelope.refinement_missing:
        reply = "That earlier file result set is not available in this session. Please run the search again."
        return OrchestratorResult(
            reply=reply,
            action_type="file_search",
            memory_content=f"{reply}\nRoute: file_search",
            spoken_reply=reply,
            spoken_metadata={
                "result_count": 0,
                "file_ids": [],
                "chunk_ids": [],
                "result_set_id": envelope.result_set_id,
                "refinement_missing": True,
            },
        )
    if not envelope.results:
        reply = "I didn't find any matching files in the current local index."
        spoken = reply
    else:
        lines = [f"Found {len(envelope.results)} matching file(s):"]
        for number, item in enumerate(envelope.results, 1):
            display = f"{item.root_alias.title()}\\{item.relative_path.replace('/', '\\')}"
            provenance = ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in item.provenance.items())
            detail = f" — {item.snippet}" if item.snippet else ""
            lines.append(f"{number}. {display}{' (' + provenance + ')' if provenance else ''}{detail}")
        reply = "\n".join(lines)
        spoken = f"I found {len(envelope.results)} matching file{'s' if len(envelope.results) != 1 else ''}."
    untrusted = "\n\n".join(item.render() for item in envelope.untrusted_snippets)
    return OrchestratorResult(
        reply=reply,
        action_type="file_search",
        memory_content=f"{reply}\n\n{untrusted}\nRoute: file_search",
        spoken_reply=spoken,
        spoken_metadata={
            "result_count": len(envelope.results),
            "file_ids": [item.file_id for item in envelope.results],
            "chunk_ids": [item.chunk_id for item in envelope.results],
            "root_aliases": sorted({item.root_alias for item in envelope.results}),
            "result_set_id": envelope.result_set_id,
            # Part 11.1: per-hit extraction method lets a caller distinguish
            # OCR-derived text from native text, and lets FileSearchVerifier
            # confirm the claim against the indexed chunk.
            "extraction_methods": [
                item.provenance.get("extraction_method") for item in envelope.results
            ],
            "ocr_result_count": sum(
                1
                for item in envelope.results
                if item.provenance.get("extraction_method") == "pymupdf_ocr"
            ),
            "needs_ocr_result_count": sum(1 for item in envelope.results if item.needs_ocr),
            "untrusted_source_ids": [item.source_id for item in envelope.untrusted_snippets],
            "index_freshness": "snapshot",
        },
    )


__all__ = [
    "execute_calendar_create",
    "execute_calendar_read",
    "execute_cross_tool_read",
    "execute_general_answer",
    "execute_gmail_compose",
    "execute_gmail_read",
    "execute_gmail_reply",
    "execute_file_search",
]
