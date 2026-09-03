from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Final, Literal, Mapping

from dotenv import load_dotenv

from .database import PROJECT_ROOT
from .acknowledgments import ActionType, detect_spoken_language
from .approval_service import approval_public_dict, create_gmail_reply_approval
from .gmail_service import (
    GmailAuthorizationError,
    GmailConfigurationError,
    GmailRateLimitError,
    GmailServiceError,
    GmailSummaryError,
    GmailDraftError,
    GmailTargetResolutionError,
    get_recent_emails,
    get_unread_emails,
    summarize_with_graceful_fallback,
    draft_reply_from_request,
)
from .llm_service import (
    LLMConfigurationError,
    LLMUnavailableError,
    generate_fast_text,
    generate_text,
)
from .memory_service import build_memory_context, local_identity_reply
from .persona import (  # noqa: F401  (re-exported for existing call sites/tests)
    AO_CHAT_SYSTEM_INSTRUCTION,
    _COMPLEX_GENERAL_CHAT_PATTERN,
    _general_chat_inference_profile,
    _SIMPLE_GREETING_PATTERN,
    _time_appropriate_greeting,
)

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")

Intent = Literal["gmail", "calendar", "file_search", "terminal", "general_chat"]

ALLOWED_INTENTS: Final[tuple[str, ...]] = (
    "gmail",
    "calendar",
    "file_search",
    "terminal",
    "general_chat",
)

DEFAULT_MODEL: Final[str] = "gemini-3.6-flash"

@dataclass(frozen=True)
class OrchestratorResult:
    reply: str
    action_type: ActionType
    memory_content: str
    spoken_reply: str | None = None
    spoken_metadata: Mapping[str, object] = field(default_factory=dict)
    approval: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HandlerResult:
    reply: str
    spoken_reply: str | None = None
    spoken_metadata: Mapping[str, object] = field(default_factory=dict)
    action_type_override: ActionType | None = None
    approval: Mapping[str, object] | None = None

def _extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise RoutingError("Groq fallback router did not return a JSON object.")
    try:
        payload = json.loads(cleaned[first : last + 1])
    except json.JSONDecodeError as exc:
        raise RoutingError("Groq fallback router returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RoutingError("Groq fallback router returned a non-object JSON value.")
    return payload


def _local_spoken_fallback(reply: str, user_message: str) -> str | None:
    """Extract a bounded useful opening when the model envelope is malformed."""
    cleaned = re.sub(r"```.*?```", " ", reply, flags=re.DOTALL)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    if detect_spoken_language(user_message) == "hi" and not re.search(r"[\u0900-\u097f]", cleaned):
        # Do not send arbitrary Roman Hindi to Rohan. The speech policy will select
        # a safe curated Devanagari fallback instead.
        return None

    sentences = re.split(r"(?<=[.!?।])\s+", cleaned)
    selected: list[str] = []
    word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if selected and word_count + len(sentence_words) > 65:
            break
        selected.append(sentence)
        word_count += len(sentence_words)
        if len(selected) >= 2 or word_count >= 35:
            break

    return " ".join(selected) or None


def _parse_conversational_output(raw_text: str, user_message: str) -> HandlerResult:
    """Parse the common Gemini/Groq envelope without making text chat fragile."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    payload: object | None = None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last > first:
            try:
                payload = json.loads(cleaned[first : last + 1])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        reply = str(payload.get("reply", "")).strip()
        spoken_reply = str(payload.get("spoken_reply", "")).strip()
        if reply:
            if spoken_reply and (
                detect_spoken_language(spoken_reply) != detect_spoken_language(user_message)
            ):
                logger.warning("Model spoken_reply language did not match the current user turn")
                spoken_reply = ""
            return HandlerResult(
                reply=reply,
                spoken_reply=spoken_reply or _local_spoken_fallback(reply, user_message),
            )

    # A malformed envelope must never turn a successful model answer into a chat error.
    return HandlerResult(
        reply=raw_text.strip(),
        spoken_reply=_local_spoken_fallback(raw_text, user_message),
    )


_GMAIL_REPLY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:reply|respond)\b", re.IGNORECASE),
    re.compile(r"\bdraft\s+(?:an?\s+)?reply\b", re.IGNORECASE),
    re.compile(r"\bsend\b.{0,80}\b(?:a\s+)?reply\b", re.IGNORECASE),
)

_GMAIL_UNSUPPORTED_WRITE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:forward|delete|archive)\b", re.IGNORECASE),
    re.compile(r"\bmark\b.+\bas\s+(?:read|unread)\b", re.IGNORECASE),
    re.compile(r"\b(?:add|remove|apply)\b.+\blabel\b", re.IGNORECASE),
    re.compile(r"\b(?:compose|send)\b.{0,80}\b(?:new\s+)?(?:email|mail|message)\b(?!.*\breply\b)", re.IGNORECASE),
)


def _gmail_reply_requested(user_message: str) -> bool:
    return any(pattern.search(user_message) for pattern in _GMAIL_REPLY_PATTERNS)


def _gmail_write_requested(user_message: str) -> bool:
    return _gmail_reply_requested(user_message) or any(
        pattern.search(user_message) for pattern in _GMAIL_UNSUPPORTED_WRITE_PATTERNS
    )


def gmail_handler(user_message: str) -> HandlerResult:
    """Read Gmail or prepare a reply approval. No natural-language path sends email."""
    if _gmail_reply_requested(user_message):
        try:
            draft = draft_reply_from_request(user_message)
            language = detect_spoken_language(user_message)
            approval = create_gmail_reply_approval(draft, spoken_language=language)
            target = draft.get("recipient_display") or draft.get("to") or "the selected recipient"
            return HandlerResult(
                reply=(
                    f"I drafted a reply to {target}. Review the exact recipient, subject, and "
                    "message below. Nothing will be sent until you explicitly approve it."
                ),
                spoken_reply=(
                    "मैंने जवाब का ड्राफ्ट तैयार कर दिया है। भेजने से पहले आपकी मंज़ूरी चाहिए।"
                    if language == "hi"
                    else "I've drafted the reply. Review it before I send anything."
                ),
                action_type_override="approval_required",
                approval=approval_public_dict(approval),
            )
        except GmailTargetResolutionError as exc:
            return HandlerResult(
                reply=str(exc),
                spoken_reply="I need a clearer sender or subject before I can prepare the reply.",
                action_type_override="error",
            )
        except GmailDraftError as exc:
            return HandlerResult(
                reply=f"I couldn't create the Gmail reply draft: {exc}",
                spoken_reply="I couldn't prepare that reply. Nothing was sent.",
                action_type_override="error",
            )
        except GmailConfigurationError as exc:
            return HandlerResult(
                reply=f"Bunnelby Gmail setup is incomplete: {exc}",
                spoken_reply="Gmail setup is incomplete. Nothing was sent. Check the connection settings.",
                action_type_override="error",
            )
        except GmailAuthorizationError as exc:
            return HandlerResult(
                reply=f"Bunnelby could not authorize Gmail: {exc}",
                spoken_reply="Gmail needs authorization before I can prepare that reply. Nothing was sent.",
                action_type_override="error",
            )
        except GmailRateLimitError:
            return HandlerResult(
                reply="Gmail API rate limit reached. Please retry in a moment.",
                spoken_reply="Gmail is temporarily rate-limited. Nothing was sent. Please retry in a moment.",
                action_type_override="error",
            )
        except GmailServiceError as exc:
            logger.warning("Gmail reply drafting failed: %s", exc)
            return HandlerResult(
                reply=f"Bunnelby could not prepare the Gmail reply: {exc}",
                spoken_reply="I couldn't prepare that Gmail reply. Nothing was sent.",
                action_type_override="error",
            )

    if any(pattern.search(user_message) for pattern in _GMAIL_UNSUPPORTED_WRITE_PATTERNS):
        return HandlerResult(
            reply=(
                "This phase supports replies to existing Gmail threads through explicit approval. "
                "New-message compose, forward, delete, archive, label, and read-state changes remain disabled."
            ),
            spoken_reply="That Gmail action is not enabled. Nothing was changed.",
            action_type_override="error",
        )

    try:
        unread_only = bool(re.search(r"\bunread\b", user_message, re.IGNORECASE))
        if unread_only:
            emails = get_unread_emails()
            empty_message = "You have no unread inbox emails."
        else:
            emails = get_recent_emails(max_results=10)
            empty_message = "No recent inbox emails were found."

        if not emails:
            return HandlerResult(
                reply=empty_message,
                spoken_metadata={"email_count": 0, "unread_only": unread_only},
            )

        return HandlerResult(
            reply=summarize_with_graceful_fallback(emails),
            spoken_metadata={"email_count": len(emails), "unread_only": unread_only},
        )
    except GmailConfigurationError as exc:
        return HandlerResult(
            reply=f"Bunnelby Gmail setup is incomplete: {exc}",
            spoken_reply="Gmail setup is incomplete. No messages were changed. Check the connection settings.",
            action_type_override="error",
        )
    except GmailAuthorizationError as exc:
        return HandlerResult(
            reply=f"Bunnelby could not authorize Gmail: {exc}",
            spoken_reply="Gmail authorization failed. No messages were changed. Reconnect Gmail and retry.",
            action_type_override="error",
        )
    except GmailRateLimitError:
        return HandlerResult(
            reply="Gmail API rate limit reached. Please retry in a moment.",
            spoken_reply="Gmail is temporarily rate-limited. Nothing was changed. Please retry in a moment.",
            action_type_override="error",
        )
    except GmailSummaryError as exc:
        logger.warning("Gmail summary generation failed: %s", exc)
        return HandlerResult(
            reply="Bunnelby read your email, but could not summarize it right now. Please retry.",
            spoken_reply="I read the inbox, but the summary failed. Your email is unaffected. Please retry.",
            action_type_override="error",
        )
    except GmailServiceError as exc:
        logger.warning("Gmail request failed: %s", exc)
        return HandlerResult(
            reply=f"Bunnelby could not read Gmail right now: {exc}",
            spoken_reply="I couldn't read Gmail. No messages were changed. Please check the connection and retry.",
            action_type_override="error",
        )


def general_chat_handler(user_message: str) -> HandlerResult:
    """Generate a context-aware Bunnelby response with local memory and cloud failover."""
    if _SIMPLE_GREETING_PATTERN.match(user_message):
        greeting = _time_appropriate_greeting()
        return HandlerResult(reply=greeting, spoken_reply=greeting)

    identity_reply = local_identity_reply(user_message)
    if identity_reply:
        return HandlerResult(reply=identity_reply, spoken_reply=identity_reply)

    local_now = datetime.now().astimezone()
    total_started = time.perf_counter()

    memory_started = time.perf_counter()
    memory_context = build_memory_context(user_message)
    memory_ms = (time.perf_counter() - memory_started) * 1000.0

    inference_profile = _general_chat_inference_profile(user_message)
    spoken_language = detect_spoken_language(user_message)
    spoken_output_directive = (
        "Hindi in natural Devanagari. Do not write Roman Hindi in spoken_reply; "
        "Latin letters are allowed only for unavoidable technical acronyms."
        if spoken_language == "hi"
        else "English. Do not switch spoken_reply into Hindi or Hinglish."
    )
    try:
        generator = (
            generate_fast_text
            if inference_profile == "fast"
            else generate_text
        )

        generation_started = time.perf_counter()

        result = generator(
            system_instruction=AO_CHAT_SYSTEM_INSTRUCTION,
            user_content=(
                f"Current local date/time: {local_now.strftime('%A, %B %d, %Y %I:%M %p %Z')}\n"
                "Use the date/time only when relevant.\n\n"
                f"{memory_context}\n\n"
                "TRUSTED SPOKEN OUTPUT LANGUAGE FOR THIS TURN:\n"
                f"{spoken_output_directive}\n\n"
                "CURRENT TURN POLICY:\n"
                "This is a general conversational turn. Answer the current user message "
                "directly. Do not offer unrelated tools or automation. Previous tool history "
                "must not change the meaning of this request.\n\n"
                "CURRENT USER MESSAGE (answer this now):\n"
                f"{user_message}"
            ),
        )

        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1000.0

        parsed = _parse_conversational_output(
            result.text,
            user_message,
        )

        total_ms = (
            time.perf_counter() - total_started
        ) * 1000.0

        metadata = dict(parsed.spoken_metadata)
        metadata.update(
            {
                "brain_profile": inference_profile,
                "brain_provider": result.provider,
                "brain_model": result.model,
                "memory_context_chars": len(memory_context),
                "latency_ms": {
                    "memory_retrieval": round(memory_ms, 2),
                    "brain_generation": round(generation_ms, 2),
                    "general_chat_total": round(total_ms, 2),
                },
            }
        )

        logger.info(
            "Bunnelby brain profile=%s provider=%s model=%s "
            "memory_chars=%s generation_ms=%.0f total_ms=%.0f",
            inference_profile,
            result.provider,
            result.model,
            len(memory_context),
            generation_ms,
            total_ms,
        )

        return HandlerResult(
            reply=parsed.reply,
            spoken_reply=parsed.spoken_reply,
            spoken_metadata=metadata,
            action_type_override=parsed.action_type_override,
            approval=parsed.approval,
        )

    except LLMConfigurationError:
        return HandlerResult(
            reply=(
                "I can't access a cloud language model yet. Configure Gemini or Groq and "
                "I'll continue normally."
            ),
            spoken_reply=(
                "Cloud language access is not configured. Your local services are unaffected."
            ),
        )
    except LLMUnavailableError:
        return HandlerResult(
            reply=(
                "My cloud AI providers are temporarily unavailable right now. "
                "Please try again shortly."
            ),
            spoken_reply=(
                "The cloud providers are temporarily unavailable. Your local services are still running. Please retry shortly."
            ),
        )
    except Exception as exc:
        logger.exception("AO general chat failed: %s", exc)
        return HandlerResult(
            reply="I couldn't answer that right now. Please try again.",
            spoken_reply="I couldn't answer that request. Nothing else was affected. Please try again.",
        )


_GMAIL_EMPTY_REPLIES: Final[frozenset[str]] = frozenset(
    {"You have no unread inbox emails.", "No recent inbox emails were found."}
)
_GENERAL_ERROR_PREFIXES: Final[tuple[str, ...]] = (
    "I can't access a cloud language model",
    "My cloud AI providers are temporarily unavailable",
    "I couldn't answer that right now",
)

