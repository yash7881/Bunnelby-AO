from __future__ import annotations

import logging
import re

from .acknowledgments import detect_spoken_language
from .approval_service import (
    approval_public_dict,
    create_calendar_event_approval,
    create_gmail_compose_approval,
)
from .calendar_agenda_service import (
    agenda_range,
    format_agenda_response,
    is_agenda_request,
    list_events,
)
from .calendar_service import (
    CalendarAuthorizationError,
    CalendarConfigurationError,
    CalendarParseError,
    CalendarRateLimitError,
    CalendarServiceError,
    calendar_event_proposal,
    check_free_busy,
    find_open_slots,
    format_free_busy_response,
    format_open_slots_response,
    parse_calendar_request,
)
from .gmail_service import (
    GmailAuthorizationError,
    GmailConfigurationError,
    GmailDraftError,
    GmailRateLimitError,
    GmailServiceError,
    draft_new_email_from_request,
)
from .orchestrator import OrchestratorResult
from .spoken_briefing import (
    calendar_agenda_briefing,
    calendar_free_busy_briefing,
    calendar_open_slots_briefing,
)

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_REPLY_RE = re.compile(r"\b(?:reply|respond|jawab)\b", re.IGNORECASE)
_COMPOSE_ACTION_RE = re.compile(
    r"\b(?:send|compose|write|draft|create|email|mail|message|bhejo|bhejna|karo|kro|likho|likhna)\b",
    re.IGNORECASE,
)
_CALENDAR_CUE_RE = re.compile(
    r"\b(?:calendar|event|meeting|appointment|schedule|agenda|timetable|time\s+table|book|"
    r"availability|available|free|slot|slots|openings?)\b",
    re.IGNORECASE,
)
_CALENDAR_TIME_RE = re.compile(
    r"\b(?:today|aaj|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"morning|afternoon|evening|tonight|\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{4}-\d{1,2}-\d{1,2})\b|आज",
    re.IGNORECASE,
)
_TODAY_ALIAS_RE = re.compile(r"\baaj\b|आज", re.IGNORECASE)
_IMPLICIT_TODAY_DAYPART_RE = re.compile(
    r"\b(?:this\s+(?:morning|afternoon|evening)|tonight)\b",
    re.IGNORECASE,
)
_EXPLICIT_CALENDAR_DATE_RE = re.compile(
    r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.IGNORECASE,
)


def _normalize_calendar_time_language(user_message: str) -> str:
    """Make deterministic current-day phrases explicit before Calendar parsing.

    Reads such as "this afternoon" and "tonight" have an unambiguous local date: today.
    We normalize only the date token and preserve the daypart for the existing parser. Genuine
    ambiguity still fails closed. Calendar writes remain protected by the exact-clock requirement.
    """
    text = user_message.strip()
    if not text:
        return text

    normalized = _TODAY_ALIAS_RE.sub("today", text)
    if _EXPLICIT_CALENDAR_DATE_RE.search(normalized):
        return normalized
    if _IMPLICIT_TODAY_DAYPART_RE.search(normalized):
        return f"today {normalized}"
    return normalized


def _standalone_email_requested(user_message: str) -> bool:
    """Detect an explicit NEW-email action.

    An exact address is ideal but no longer mandatory here: Bunnelby may
    conservatively resolve a named recipient from the user's Gmail history.
    Reply commands remain on the existing reply-thread path.
    """
    text = user_message.strip()
    if not text:
        return False

    if _REPLY_RE.search(text) or _CALENDAR_CUE_RE.search(text):
        return False

    has_email_word = bool(
        re.search(r"\b(?:email|e-mail|mail|gmail)\b", text, re.IGNORECASE)
    )
    has_compose_action = bool(
        re.search(
            r"\b(?:send|compose|write|draft|bhej(?:o|na)?|bhejo|karo|kro|likho)\b",
            text,
            re.IGNORECASE,
        )
    )

    return bool(has_email_word and has_compose_action)


def _calendar_requested(user_message: str) -> bool:
    text = user_message.strip()
    if not text or not _CALENDAR_CUE_RE.search(text):
        return False
    return bool(
        re.search(
            r"\b(?:calendar|event|meeting|appointment|schedule|agenda|timetable|time\s+table|book|slot|slots|openings?)\b",
            text,
            re.IGNORECASE,
        )
        or (
            re.search(r"\b(?:am\s+i|are\s+we|find|check|show|free|available)\b", text, re.IGNORECASE)
            and _CALENDAR_TIME_RE.search(text)
        )
    )


def _result(
    reply: str,
    *,
    spoken_reply: str,
    action_type: str,
    route: str,
    reason: str,
    approval: dict | None = None,
) -> OrchestratorResult:
    return OrchestratorResult(
        reply=reply,
        action_type=action_type,
        memory_content=f"{reply}\nRoute: {route}\nWhy: {reason}",
        spoken_reply=spoken_reply,
        approval=approval,
    )


def _calendar_agenda_result(user_message: str) -> OrchestratorResult:
    language = detect_spoken_language(user_message)
    try:
        normalized_message = _normalize_calendar_time_language(user_message)
        start, end, _zone = agenda_range(normalized_message)
        events = list_events((start, end))
        reply = format_agenda_response(start, events)
        spoken = calendar_agenda_briefing(start, events, language)
        return _result(
            reply,
            spoken_reply=spoken,
            action_type="calendar_read",
            route="calendar",
            reason="verified Google Calendar agenda lookup",
        )
    except CalendarParseError as exc:
        return _result(
            str(exc),
            spoken_reply=(
                "शेड्यूल देखने के लिए तारीख थोड़ी और स्पष्ट बताइए।"
                if language == "hi"
                else "I need a clearer date before I can read your schedule."
            ),
            action_type="error",
            route="calendar",
            reason="calendar agenda date requires clarification",
        )
    except CalendarConfigurationError as exc:
        return _result(
            f"Bunnelby Calendar setup is incomplete: {exc}",
            spoken_reply="Calendar setup is incomplete. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar configuration error",
        )
    except CalendarAuthorizationError as exc:
        return _result(
            f"Bunnelby could not authorize Google Calendar: {exc}",
            spoken_reply="Google Calendar authorization is required. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar authorization error",
        )
    except CalendarRateLimitError:
        return _result(
            "Google Calendar API rate limit reached. Please retry in a moment.",
            spoken_reply="Google Calendar is temporarily rate-limited. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar rate limit",
        )
    except CalendarServiceError as exc:
        logger.warning("Calendar agenda request failed: %s", exc)
        return _result(
            f"Bunnelby could not read your Google Calendar schedule: {exc}",
            spoken_reply="I couldn't read your Google Calendar schedule right now.",
            action_type="error",
            route="calendar",
            reason="calendar agenda service error",
        )


def _calendar_result(user_message: str) -> OrchestratorResult:
    language = detect_spoken_language(user_message)
    try:
        normalized_message = _normalize_calendar_time_language(user_message)
        request = parse_calendar_request(normalized_message)

        if request.action == "create_event":
            busy = check_free_busy((request.start, request.end))
            if busy:
                reply = (
                    "That requested time overlaps an existing busy period. Nothing was created. "
                    "Choose another time or ask me to find an open slot."
                )
                spoken = (
                    "उस समय आपका कैलेंडर व्यस्त है। कुछ भी नहीं बनाया गया।"
                    if language == "hi"
                    else "That time overlaps something already on your calendar. Nothing was created."
                )
                return _result(reply, spoken_reply=spoken, action_type="calendar_read", route="calendar", reason="calendar conflict check")

            proposal = calendar_event_proposal(request)
            approval = create_calendar_event_approval(proposal, spoken_language=language)
            public = approval_public_dict(approval)
            assumption = (
                " I used a 60-minute duration because you did not specify one; review the end time before approving."
                if request.assumed_duration
                else ""
            )
            return _result(
                "I've prepared the calendar event. Review the exact title, time, timezone, and attendees before I create anything."
                + assumption,
                spoken_reply=(
                    "मैंने कैलेंडर इवेंट तैयार कर दिया है। बनाने से पहले आपकी मंज़ूरी चाहिए।"
                    if language == "hi"
                    else "I've prepared the calendar event. Review it before I create anything."
                ),
                action_type="approval_required",
                route="calendar",
                reason="calendar event requires explicit approval",
                approval=public,
            )

        if request.action == "open_slots":
            slots = find_open_slots(
                request.start.date(),
                request.duration_minutes,
                request.work_hours,
            )
            reply = format_open_slots_response(request, slots)
            spoken = calendar_open_slots_briefing(request, slots, language)
            return _result(reply, spoken_reply=spoken, action_type="calendar_read", route="calendar", reason="calendar open-slot lookup")

        busy = check_free_busy((request.start, request.end))
        reply = format_free_busy_response(request, busy)
        spoken = calendar_free_busy_briefing(request, busy, language)
        return _result(reply, spoken_reply=spoken, action_type="calendar_read", route="calendar", reason="calendar free-busy lookup")

    except CalendarParseError as exc:
        return _result(
            str(exc),
            spoken_reply=(
                "कैलेंडर अनुरोध के लिए तारीख या समय थोड़ा और स्पष्ट बताइए।"
                if language == "hi"
                else "I need a clearer date or time before I can use the calendar safely."
            ),
            action_type="error",
            route="calendar",
            reason="calendar time parsing requires clarification",
        )
    except CalendarConfigurationError as exc:
        return _result(
            f"Bunnelby Calendar setup is incomplete: {exc}",
            spoken_reply="Calendar setup is incomplete. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar configuration error",
        )
    except CalendarAuthorizationError as exc:
        return _result(
            f"Bunnelby could not authorize Google Calendar: {exc}",
            spoken_reply="Google Calendar authorization is required. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar authorization error",
        )
    except CalendarRateLimitError:
        return _result(
            "Google Calendar API rate limit reached. Please retry in a moment.",
            spoken_reply="Google Calendar is temporarily rate-limited. Nothing was changed.",
            action_type="error",
            route="calendar",
            reason="calendar rate limit",
        )
    except CalendarServiceError as exc:
        logger.warning("Calendar request failed: %s", exc)
        return _result(
            f"Bunnelby could not complete the Calendar request: {exc}",
            spoken_reply="I couldn't complete that calendar request. Nothing unsafe was changed.",
            action_type="error",
            route="calendar",
            reason="calendar service error",
        )


def _gmail_compose_result(user_message: str) -> OrchestratorResult:
    language = detect_spoken_language(user_message)
    try:
        draft = draft_new_email_from_request(user_message)
        approval = create_gmail_compose_approval(draft, spoken_language=language)
        public = approval_public_dict(approval)
        return _result(
            f"I drafted a new email to {draft['to']}. Review the exact recipient, subject, and message below. Nothing will be sent until you explicitly approve it.",
            spoken_reply=(
                "मैंने नया ईमेल ड्राफ्ट तैयार कर दिया है। भेजने से पहले आपकी मंज़ूरी चाहिए।"
                if language == "hi"
                else "I've drafted the new email. Review it before I send anything."
            ),
            action_type="approval_required",
            route="gmail",
            reason="explicit standalone email compose request",
            approval=public,
        )
    except GmailDraftError as exc:
        detail = str(exc)
        clarification = bool(
            re.search(
                r"(multiple|which one|identify|recipient|exact email|exact address|correspondent|matching)",
                detail,
                re.IGNORECASE,
            )
        )
        return _result(
            detail if clarification else f"I couldn't create the new Gmail draft: {detail}",
            spoken_reply=(
                "???? ??? ??????? ????? ?? ??? ????? ?? ??????? ??????"
                if language == "hi" and clarification
                else "I need one more detail to identify the correct recipient."
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
    except GmailConfigurationError as exc:
        return _result(f"Bunnelby Gmail setup is incomplete: {exc}", spoken_reply="Gmail setup is incomplete. Nothing was sent.", action_type="error", route="gmail", reason="gmail configuration error")
    except GmailAuthorizationError as exc:
        return _result(f"Bunnelby could not authorize Gmail: {exc}", spoken_reply="Gmail authorization is required. Nothing was sent.", action_type="error", route="gmail", reason="gmail authorization error")
    except GmailRateLimitError:
        return _result("Gmail API rate limit reached. Please retry in a moment.", spoken_reply="Gmail is temporarily rate-limited. Nothing was sent.", action_type="error", route="gmail", reason="gmail rate limit")
    except GmailServiceError as exc:
        logger.warning("Standalone Gmail compose failed: %s", exc)
        return _result(f"Bunnelby could not prepare the new Gmail message: {exc}", spoken_reply="I couldn't prepare that email. Nothing was sent.", action_type="error", route="gmail", reason="gmail service error")


def handle_message_result(user_message: str) -> OrchestratorResult:
    """Brain-first dispatch: a single semantic LLM decision routes every turn.

    A deterministic regex gate no longer runs before the LLM sees the message. Instead
    brain_agent.decide() classifies the turn (answer / clarify / tool), and only a
    mode=="tool" decision reaches deterministic execution via tool_executor, which in turn
    reuses the existing builder functions below (_gmail_compose_result, _calendar_result,
    _calendar_agenda_result) purely as execution helpers.
    """
    from . import tool_executor  # local import avoids a circular import at module load time
    from .brain_agent import decide as brain_decide

    decision = brain_decide(user_message)

    if decision.mode == "tool":
        return tool_executor.execute(decision, user_message)

    action_type = "general_answer" if decision.mode == "answer" else "clarification_required"
    reply = decision.reply
    return OrchestratorResult(
        reply=reply,
        action_type=action_type,
        memory_content=f"{reply}\nRoute: brain\nWhy: {decision.reason or decision.mode}",
        spoken_reply=decision.spoken_reply or None,
    )
