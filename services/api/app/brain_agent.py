from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal, Mapping

from .acknowledgments import detect_spoken_language
from .llm_service import (
    LLMConfigurationError,
    LLMServiceError,
    LLMUnavailableError,
    generate_fast_text,
    generate_text,
)
from .memory_service import build_memory_context, local_identity_reply
from .personal_facts import build_personal_facts_context, fact_saved_reply, try_save_stated_fact
from .untrusted_content import TRUST_POLICY_CLAUSE
from .persona import (
    AO_CHAT_SYSTEM_INSTRUCTION,
    SIMPLE_GREETING_PATTERN,
    general_chat_inference_profile,
    time_appropriate_greeting,
)

logger = logging.getLogger(__name__)

BrainMode = Literal["answer", "clarify", "tool"]
BrainTool = Literal[
    "gmail_read",
    "gmail_compose",
    "gmail_reply",
    "calendar_read",
    "calendar_create",
    "cross_tool_read",
    "file_search",
]

# Writes must fail closed to clarification on low confidence or missing arguments.
_WRITE_TOOLS: Final[frozenset[str]] = frozenset({"gmail_compose", "calendar_create"})
_MIN_WRITE_CONFIDENCE: Final[float] = 0.6
_ALLOWED_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "gmail_read",
        "gmail_compose",
        "gmail_reply",
        "calendar_read",
        "calendar_create",
        "cross_tool_read",
        "file_search",
    }
)
_REQUIRED_ARGS_FOR_TOOL: Final[Mapping[str, tuple[str, ...]]] = {
    "gmail_compose": ("recipient_hint",),
    "calendar_create": ("title",),
}


ResponsePolicy = Literal["full_ui", "concise_spoken", "both"]


@dataclass(frozen=True)
class BrainDecision:
    """BrainDecisionV2 (Part 10.2 Phase H).

    The original four fields are unchanged so every existing caller and test
    keeps working. The additions carry what the typed pipeline needs:

    requires_approval        the model's own read on impact. ADVISORY ONLY --
                             risk_policy is authoritative and will override a
                             model that under-declares an external write.
    reason_code              short machine-readable identifier for diagnostics
                             and tool_runs, instead of free prose only.
    untrusted_context_used   provenance ids of external content that informed
                             this decision (Phase L).
    response_policy          whether the turn wants screen detail, concise
                             speech, or both.
    """

    mode: BrainMode
    tool: str | None
    confidence: float
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reply: str = ""
    spoken_reply: str = ""
    reason: str = ""
    requires_approval: bool | None = None
    reason_code: str = ""
    untrusted_context_used: tuple[str, ...] = ()
    response_policy: ResponsePolicy = "both"


# Explicit alias for callers that want to name the version they depend on.
BrainDecisionV2 = BrainDecision


BRAIN_SYSTEM_INSTRUCTION: Final[str] = (
    AO_CHAT_SYSTEM_INSTRUCTION
    + "\n\n"
    + """
ADDITIONAL ROUTING RESPONSIBILITY:
You are also Bunnelby's single semantic decision layer for this turn. In addition to the
reply/spoken_reply persona rules above, decide whether this turn is ordinary conversation,
needs clarification, or is an explicit request to use the user's REAL personal Gmail,
Google Calendar, or approved local-file index.

Default to normal conversation ("answer"). The mere presence of words like "email",
"gmail", "calendar", "meeting", or "schedule" is NEVER enough by itself to select a tool.
A question ABOUT Gmail/Calendar/email/meetings (e.g. "what is gmail", "calendar database
kya hota hai", "meeting kaisi thi") is conversation, not a tool request. A casual statement
that merely mentions a meeting or email (e.g. "my meeting with Rahul was terrible") is
conversation, not a tool request.

Only select "tool" when the user unambiguously wants Bunnelby to actually read or act on
their real Gmail or Calendar:
- gmail_read: read/check/summarize the inbox or unread mail.
- gmail_reply: reply to an existing email thread.
- gmail_compose: send/draft a brand-new email to someone.
- calendar_read: check schedule/agenda/availability/free-busy.
- calendar_create: schedule/book/create a new calendar event.
- cross_tool_read: the user explicitly wants BOTH their real Gmail inbox AND their real
  Google Calendar checked together in the same request (e.g. "Check my latest emails and
  what's on my calendar tomorrow", "Read my inbox and check today's calendar"). Use this
  only when both a genuine Gmail read and a genuine Calendar read are clearly requested
  together; a single-tool request must still use gmail_read or calendar_read, not
  cross_tool_read.
- file_search: the user explicitly wants to find/search their actual local files, filenames,
  paths, metadata, or indexed document content. Conceptual questions about file search,
  SQLite FTS5, lexical search, or vector search are ordinary conversation, never this tool.

A conceptual, comparative, or opinion question about Gmail and/or Calendar as products or
concepts (e.g. "Explain the difference between Gmail and Google Calendar", "Compare email
and calendar systems", "Can Gmail and Calendar integrate with each other?", "I use Gmail and
Calendar every day", "My Gmail emails mention calendar meetings") is ordinary conversation
("answer"), NEVER a tool call, no matter how many Gmail/Calendar keywords it contains. Only
select cross_tool_read when the user is unambiguously asking Bunnelby to go look at their
actual personal inbox and actual personal calendar right now.

Voice transcripts may be garbled or use Hindi/Hinglish/STT errors. Interpret semantically,
not by exact phrasing. If the request is a bare ambiguous word ("email", "calendar") or a
garbled/uncertain transcript, use mode="clarify" and ask a brief clarifying question in
reply/spoken_reply.

ANAPHORA AND FOLLOW-UP RESOLUTION (applies to mode=answer as much as mode=tool/clarify):
The memory context below marks a MOST RECENT TURN separately from any EARLIER RECENT
CONVERSATION and RELEVANT OLDER LOCAL MEMORY. When the current user message uses an
ambiguous reference ("it", "that", "this", "they", "iska", "uska") or an implicit follow-up
("explain it more", "give a real life example", "compare that with X", "why is it useful"),
resolve it in this order: the current message itself if self-contained, then the MOST RECENT
TURN (the immediately preceding exchange -- this is the current active topic and the primary
candidate), then EARLIER RECENT CONVERSATION only if the most recent turn offers no plausible
referent, then RELEVANT OLDER LOCAL MEMORY only as a last resort. Do not treat a related
concept the most recent turn merely mentioned in passing (while explaining its single actual
topic) as a second competing topic. Use mode="clarify" for an ambiguous reference only when
the MOST RECENT TURN itself covered multiple distinct, unrelated topics and the current
message gives no way to safely choose one -- never clarify just because the most recent
answer's explanation happened to touch on more than one term.

SET/COLLECTION REFERENCE RESOLUTION (a specific, common case of the anaphora rule above):
When the MOST RECENT TURN presented ONE coherent result SET from a single prior action --
several emails from one Gmail read, several meetings from one Calendar read, several files
from one search, or even several options Bunnelby itself enumerated in one plain-conversation
answer (e.g. "here are three laptop options: X, Y, Z") -- treat that entire set as ONE
resolvable referent, not as multiple distinct topics. A follow-up such as "which one", "which
looks most important", "which needs my attention first", "the first one", "any of them",
"which email", or "which meeting" must be resolved by reasoning over the items already listed
in that MOST RECENT TURN:
- Use mode="answer" (never mode="tool") -- the data needed to answer is already in the
  MOST RECENT TURN above, so no new gmail_read/calendar_read/cross_tool_read call is needed,
  only reasoning over what was already returned.
- Do NOT use mode="clarify" for this case. A single collection of many items returned by one
  prior action is NOT the same kind of ambiguity as multiple unrelated topics -- it is one
  topic (the set) with several members, and picking/ranking/explaining a member is a normal
  reasoning task, not something that requires the user to disambiguate.
- The only time a fresh tool call is appropriate for such a follow-up is when the user
  explicitly asks to refresh, re-check, re-run, or fetch newer/latest data (e.g. "check again
  for new emails", "any new mail since then", "refresh my calendar", "search again") -- that
  is a new data request, not a reasoning-over-existing-data request.
Keep this distinct from genuine multi-topic ambiguity: if the MOST RECENT TURN instead
explained multiple separate, unrelated subjects (e.g. it explained both RAG and FastAPI as
two different concepts, not as items in one returned list), and the follow-up gives no way to
tell which subject it continues, mode="clarify" remains appropriate there -- that is a
different situation from a single enumerated/fetched collection.

For gmail_compose and calendar_create (WRITE actions), uncertainty must ALWAYS fail closed
to mode="clarify" rather than guessing. Never silently choose a write action you are not
confident about. Reads (gmail_read, calendar_read) can tolerate more phrasing slack.

Return ONE valid JSON object with exactly these fields, and nothing else:
{
  "mode": "answer" | "clarify" | "tool",
  "tool": null | "gmail_read" | "gmail_compose" | "gmail_reply" | "calendar_read" | "calendar_create" | "cross_tool_read" | "file_search",
  "confidence": 0.0-1.0,
  "arguments": {"recipient_hint": "...", "subject_hint": "...", "body_hint": "...",
                "title": "...", "start_hint": "...", "end_hint": "...",
                "timezone_hint": "...", "attendees_hint": []},
  "reply": "for answer/clarify: the full on-screen reply; for tool: a brief transitional note or empty string",
  "spoken_reply": "for answer/clarify: the spoken reply; for tool: a brief transitional note or empty string",
  "reason": "short internal-only reason, never shown to the user"
}

Only include argument keys that are actually relevant/known; omit unknown ones rather than
guessing. Never fabricate a recipient email address, event title, or time that was not
stated or clearly implied by the user. When mode is "tool", reply/spoken_reply may be a
short transitional note (e.g. "Let me check that.") because a deterministic execution step
will build the final user-facing reply.
""".strip()
)


def _selectable_tool_names() -> frozenset[str]:
    """Tools the Brain may name, taken from the live registry.

    Falls back to the historical literal set if the registry is somehow empty,
    so a decision can never be silently discarded because of import ordering.
    """
    try:
        from . import tool_executor  # noqa: F401
        from .capability_registry import registry

        names = frozenset(registry().tool_names())
        return names or _ALLOWED_TOOLS
    except Exception:  # pragma: no cover - defensive
        logger.warning("Capability registry unavailable; using the static tool set")
        return _ALLOWED_TOOLS


def brain_system_instruction() -> str:
    """Full routing instruction: persona + prose + trust boundary + live catalog.

    The trust clause is included here (Part 10.2 Phase L) because the live Brain
    prompt previously had NO untrusted-content language at all -- the only two
    such clauses in orchestrator.py belonged to the dead legacy router.
    """
    return "\n\n".join(
        (BRAIN_SYSTEM_INSTRUCTION, TRUST_POLICY_CLAUSE, tool_catalog_section())
    )


def _registered_capabilities() -> tuple[dict[str, Any], ...]:
    """Tool catalog straight from the Capability Registry.

    Imported lazily so brain_agent stays free of an import-time dependency on
    tool_executor (which imports brain_agent for the decision type). Importing
    tool_executor here also guarantees capabilities are registered before the
    catalog is read.
    """
    from . import tool_executor  # noqa: F401  (registers capabilities on import)
    from .capability_registry import registry

    return registry().catalog()


def tool_catalog_section() -> str:
    """Render the registry as prompt text.

    Part 10.2 Phase H: adding a capability now means registering a Capability,
    not editing an ever-growing prose list. The hand-written disambiguation
    rules above this section are retained deliberately -- they encode specific
    regressions (conceptual Gmail questions, collection follow-ups) that a
    generated summary would lose -- but no NEW tool requires prose edits.
    """
    lines = [
        "REGISTERED CAPABILITY CATALOG (generated from the Capability Registry; "
        "the only selectable values for \"tool\"):",
    ]
    for entry in _registered_capabilities():
        approval = " [REQUIRES EXPLICIT APPROVAL]" if entry["requires_approval"] else ""
        lines.append(f"- {entry['name']} ({entry['risk_level']}){approval}: {entry['description']}")
        if entry.get("selection_guidance"):
            lines.append(f"    when: {entry['selection_guidance']}")
        argument_names = sorted(entry["arguments"].get("properties", {}))
        if argument_names:
            lines.append(f"    arguments: {', '.join(argument_names)}")
        required = entry["arguments"].get("required") or []
        if required:
            lines.append(f"    required: {', '.join(sorted(required))}")
    lines.append(
        "\nArguments must match the named capability's argument list. Omit an argument "
        "you do not know rather than inventing a value. Unknown argument names are "
        "discarded, and an invalid value fails the turn closed to a clarification."
    )
    return "\n".join(lines)


_JSON_SCALARS: Final[Mapping[str, str]] = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
}


def _argument_property_union() -> dict[str, Any]:
    """Flat union of every capability's argument fields, for the response schema.

    Provider structured-output schemas do not express "one of these argument
    objects depending on the tool", so the envelope declares the union of all
    argument names as optional fields. Pydantic then validates the ones that
    matter for the selected tool and discards the rest, so the schema constrains
    shape while tool_requests remains the authority on validity.
    """
    properties: dict[str, Any] = {}
    for entry in _registered_capabilities():
        for name, spec in entry["arguments"].get("properties", {}).items():
            declared = spec.get("type")
            if isinstance(declared, str) and declared in _JSON_SCALARS:
                kind = _JSON_SCALARS[declared]
            elif declared == "array":
                kind = "ARRAY"
            else:
                # Unions, enums-with-null and anyOf collapse to STRING: the
                # model still emits a usable value and Pydantic coerces it.
                kind = "STRING"
            existing = properties.get(name)
            if existing is None:
                properties[name] = (
                    {"type": kind, "items": {"type": "STRING"}}
                    if kind == "ARRAY"
                    else {"type": kind}
                )
            elif existing.get("type") != kind:
                properties[name] = {"type": "STRING"}
    return properties


def decision_response_schema() -> dict[str, Any]:
    """Provider-native response schema for BrainDecisionV2."""
    tools = sorted(entry["name"] for entry in _registered_capabilities())
    return {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "enum": ["answer", "clarify", "tool"]},
            # Gemini rejects an empty string as an enum member with HTTP 400, and
            # nullable enums are unreliable across providers. `tool` is simply
            # omitted for answer/clarify turns -- it is absent from `required`,
            # and _parse_decision treats missing/empty as "no tool".
            "tool": {"type": "STRING", "enum": list(tools)},
            "confidence": {"type": "NUMBER"},
            "arguments": {
                "type": "OBJECT",
                "properties": _argument_property_union(),
            },
            "reply": {"type": "STRING"},
            "spoken_reply": {"type": "STRING"},
            "reason": {"type": "STRING"},
            "reason_code": {"type": "STRING"},
            "requires_approval": {"type": "BOOLEAN"},
            "response_policy": {
                "type": "STRING",
                "enum": ["full_ui", "concise_spoken", "both"],
            },
        },
        "required": ["mode", "confidence", "reply", "spoken_reply"],
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        try:
            payload = json.loads(cleaned[first : last + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return None


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _fail_closed(mode: str, tool: str | None, confidence: float, arguments: Mapping[str, Any]) -> tuple[str, str]:
    """Return (mode, reason_suffix), downgrading uncertain writes to clarify."""
    if mode == "tool" and tool in _WRITE_TOOLS:
        if confidence < _MIN_WRITE_CONFIDENCE:
            return "clarify", f"low confidence ({confidence}) for write tool {tool}"
        required = _REQUIRED_ARGS_FOR_TOOL.get(tool or "", ())
        missing = [key for key in required if not str(arguments.get(key, "")).strip()]
        if missing:
            return "clarify", f"missing required arguments {missing} for write tool {tool}"
    return mode, ""


def _parse_decision(raw_text: str, user_message: str) -> BrainDecision:
    payload = _extract_json_object(raw_text)
    if payload is None:
        # A malformed envelope must never crash the turn; fail closed to a plain answer
        # using the raw text as the reply.
        text = raw_text.strip() or "I couldn't process that right now. Could you rephrase?"
        return BrainDecision(mode="answer", tool=None, confidence=0.0, reply=text, spoken_reply=text, reason="malformed brain JSON")

    mode = str(payload.get("mode", "answer")).strip().casefold()
    if mode not in ("answer", "clarify", "tool"):
        mode = "answer"

    # The structured-output schema uses "" as the no-tool value because nullable
    # enums are unreliable across providers; both forms map to None here.
    tool = payload.get("tool")
    tool = str(tool).strip() if tool else None
    if tool is not None and tool not in _selectable_tool_names():
        logger.warning("Brain proposed unregistered tool=%r; discarding", tool)
        tool = None

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    arguments = _coerce_arguments(payload.get("arguments"))
    reply = str(payload.get("reply", "")).strip()
    spoken_reply = str(payload.get("spoken_reply", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    reason_code = str(payload.get("reason_code", "")).strip()[:64]
    raw_requires_approval = payload.get("requires_approval")
    requires_approval = (
        bool(raw_requires_approval) if isinstance(raw_requires_approval, bool) else None
    )
    response_policy = str(payload.get("response_policy", "both")).strip().casefold()
    if response_policy not in ("full_ui", "concise_spoken", "both"):
        response_policy = "both"

    if mode == "tool" and tool is None:
        mode = "clarify"
        reason = f"{reason}; tool missing/invalid, downgraded".strip("; ")

    mode, fail_reason = _fail_closed(mode, tool, confidence, arguments)
    if fail_reason:
        reason = f"{reason}; fail-closed: {fail_reason}".strip("; ")
        tool = None

    if mode == "clarify" and not reply:
        reply = "Could you clarify what you'd like me to do?"
        spoken_reply = spoken_reply or reply

    if mode == "answer" and not reply:
        reply = raw_text.strip() or "I couldn't answer that right now. Please try again."
        spoken_reply = spoken_reply or reply

    return BrainDecision(
        mode=mode,  # type: ignore[arg-type]
        tool=tool if mode == "tool" else None,
        confidence=confidence,
        arguments=arguments,
        reply=reply,
        spoken_reply=spoken_reply,
        reason=reason,
        requires_approval=requires_approval,
        reason_code=reason_code,
        response_policy=response_policy,  # type: ignore[arg-type]
    )


def decide(user_message: str, session_id: str | None = None) -> BrainDecision:
    """Single semantic decision point: conversational answer, clarification, or tool call.

    session_id (Part 10.2 Phase D) confines the memory context to the active
    conversation. It is optional so existing callers keep working.
    """
    text = user_message.strip()

    if not text:
        greeting = time_appropriate_greeting()
        return BrainDecision(mode="answer", tool=None, confidence=1.0, reply=greeting, spoken_reply=greeting, reason="empty message")

    if SIMPLE_GREETING_PATTERN.match(text):
        greeting = time_appropriate_greeting()
        return BrainDecision(mode="answer", tool=None, confidence=1.0, reply=greeting, spoken_reply=greeting, reason="simple greeting fast path")

    identity_reply = local_identity_reply(text)
    if identity_reply:
        return BrainDecision(mode="answer", tool=None, confidence=1.0, reply=identity_reply, spoken_reply=identity_reply, reason="local identity fast path")

    saved_fact = try_save_stated_fact(text, session_id=session_id)
    if saved_fact:
        ack = fact_saved_reply(saved_fact)
        return BrainDecision(mode="answer", tool=None, confidence=1.0, reply=ack, spoken_reply=ack, reason="personal fact saved")

    memory_context = build_memory_context(text, session_id=session_id)
    facts_context = build_personal_facts_context()
    if facts_context:
        memory_context = f"{memory_context}\n\n{facts_context}"
    local_now = datetime.now().astimezone()
    inference_profile = general_chat_inference_profile(text)
    spoken_language = detect_spoken_language(text)
    spoken_output_directive = (
        "Hindi in natural Devanagari. Do not write Roman Hindi in spoken_reply; "
        "Latin letters are allowed only for unavoidable technical acronyms."
        if spoken_language == "hi"
        else "English. Do not switch spoken_reply into Hindi or Hinglish."
    )

    user_content = (
        f"Current local date/time: {local_now.strftime('%A, %B %d, %Y %I:%M %p %Z')}\n"
        "Use the date/time only when relevant.\n\n"
        f"{memory_context}\n\n"
        "TRUSTED SPOKEN OUTPUT LANGUAGE FOR THIS TURN:\n"
        f"{spoken_output_directive}\n\n"
        "CURRENT TURN POLICY:\n"
        "Decide mode/tool per the routing responsibility above, then answer the current "
        "user message directly if mode is answer/clarify. Previous tool history must not "
        "change the meaning of this request.\n\n"
        "CURRENT USER MESSAGE (decide and answer this now, possibly voice-transcribed and "
        "imperfect):\n"
        f"{text}"
    )

    try:
        # ONE generation per turn. The same call decides the route AND produces
        # the conversational reply, so ordinary chat never pays a routing call
        # plus a second answer call.
        generator = generate_fast_text if inference_profile == "fast" else generate_text
        result = generator(
            system_instruction=brain_system_instruction(),
            user_content=user_content,
            response_schema=decision_response_schema(),
        )
        decision = _parse_decision(result.text, text)
        logger.info(
            "brain_agent decision mode=%s tool=%s confidence=%.2f provider=%s reason=%s",
            decision.mode,
            decision.tool,
            decision.confidence,
            result.provider,
            decision.reason,
        )
        return decision
    except LLMConfigurationError:
        message = (
            "I can't access a cloud language model yet. Configure Gemini or Groq and "
            "I'll continue normally."
        )
        return BrainDecision(mode="answer", tool=None, confidence=0.0, reply=message, spoken_reply="Cloud language access is not configured. Your local services are unaffected.", reason="llm configuration error")
    except LLMUnavailableError:
        message = "My cloud AI providers are temporarily unavailable right now. Please try again shortly."
        return BrainDecision(mode="answer", tool=None, confidence=0.0, reply=message, spoken_reply=message, reason="llm unavailable")
    except LLMServiceError as exc:
        logger.exception("brain_agent LLM call failed: %s", exc)
        message = "I couldn't answer that right now. Please try again."
        return BrainDecision(mode="answer", tool=None, confidence=0.0, reply=message, spoken_reply=message, reason="llm service error")
