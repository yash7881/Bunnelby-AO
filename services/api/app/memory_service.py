from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .database import PROJECT_ROOT, SessionLocal
from .models import Message
from .untrusted_content import wrap_tool_summary

logger = logging.getLogger(__name__)

PROFILE_PATH: Final[Path] = PROJECT_ROOT / "database" / "ao_user_profile.json"
DEFAULT_PROFILE_ID: Final[str] = "local-default"
DEFAULT_PREFERRED_NAME: Final[str] = "Parth"
DEFAULT_ASSISTANT_NAME: Final[str] = "Bunnelby"
DEFAULT_RECENT_TURNS: Final[int] = 10
DEFAULT_RELEVANT_OLD_TURNS: Final[int] = 4
DEFAULT_SCAN_MESSAGES: Final[int] = 500
MAX_MESSAGE_CHARS: Final[int] = 1600
# The immediately preceding turn (MOST RECENT TURN) gets a wider, but still bounded, budget.
# A full Gmail/Calendar triage result (several items with sender/subject/time each) can
# legitimately exceed MAX_MESSAGE_CHARS; without this the tail items (e.g. later emails in a
# list) were silently clipped out of the one turn the brain most needs to reason over for a
# "which one" style follow-up.
MOST_RECENT_TURN_MAX_CHARS: Final[int] = 3200

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "what", "who", "when", "where", "why", "how", "this", "that", "these",
        "those", "it", "its", "me", "my", "mine", "you", "your", "yours", "we",
        "our", "i", "am", "can", "could", "would", "should", "please", "tell",
        "explain", "about", "with", "from", "as", "at", "by", "give", "show",
        "mujhe", "mera", "meri", "mere", "kya", "hai", "ka", "ki", "ke", "ko",
        "se", "ye", "wo", "aur", "ab", "tha", "thi", "ho", "kar", "kr", "bata",
    }
)

# True operational/error responses remain excluded from conversational memory. Tool results
# such as Gmail summaries and Calendar availability are safe to remember after their
# internal Route/Why metadata is stripped.
_OPERATIONAL_ASSISTANT_MARKERS: Final[tuple[str, ...]] = (
    "This would call the ",
    "AO's Gemini router hit the free-tier rate limit",
    "My cloud AI providers are temporarily unavailable",
    "I can't access a cloud language model yet",
    "AO could not classify that request right now",
    "AO could not resolve this ambiguous tool request",
    "Bunnelby could not classify that request right now",
    "Bunnelby could not resolve this ambiguous tool request",
)

_ROUTE_METADATA_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*(?:Route|Why):\s*[^\n]*\n?"
)

_ROUTE_VALUE_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*Route:\s*([^\n]+)"
)


@dataclass(frozen=True)
class UserProfile:
    profile_id: str
    preferred_name: str
    assistant_name: str
    mode: str = "single_user_local"


@dataclass(frozen=True)
class MemoryTurn:
    user_id: int
    assistant_id: int
    user: str
    assistant: str
    route: str | None = None


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1000) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return min(maximum, max(minimum, int(raw)))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def ensure_local_profile() -> Path:
    """Create the single-user profile once, without overwriting user edits."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROFILE_PATH.exists():
        return PROFILE_PATH

    payload = {
        "profile_id": DEFAULT_PROFILE_ID,
        "preferred_name": DEFAULT_PREFERRED_NAME,
        "assistant_name": DEFAULT_ASSISTANT_NAME,
        "mode": "single_user_local",
    }
    temp_path = PROFILE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(PROFILE_PATH)
    return PROFILE_PATH


def load_user_profile() -> UserProfile:
    ensure_local_profile()
    try:
        payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read Bunnelby local profile; using defaults: %s", exc)
        payload = {}

    # One-time safe branding migration. Existing local single-user profiles created before
    # Prompt 7 contain assistant_name=AO. Preserve every other user field unchanged.
    if str(payload.get("assistant_name") or "").strip().casefold() == "ao":
        payload["assistant_name"] = DEFAULT_ASSISTANT_NAME
        try:
            temp_path = PROFILE_PATH.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(PROFILE_PATH)
        except OSError as exc:
            logger.warning("Could not persist Bunnelby profile-name migration: %s", exc)

    return UserProfile(
        profile_id=str(payload.get("profile_id") or DEFAULT_PROFILE_ID).strip(),
        preferred_name=str(payload.get("preferred_name") or DEFAULT_PREFERRED_NAME).strip(),
        assistant_name=str(payload.get("assistant_name") or DEFAULT_ASSISTANT_NAME).strip(),
        mode=str(payload.get("mode") or "single_user_local").strip(),
    )


def _sanitize_assistant_memory(content: str) -> str:
    """Remove internal router metadata while preserving user-visible tool outcomes."""
    cleaned = _ROUTE_METADATA_LINE.sub("", content).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _assistant_is_memory_safe(content: str) -> bool:
    return bool(content.strip()) and not any(
        marker.lower() in content.lower() for marker in _OPERATIONAL_ASSISTANT_MARKERS
    )


def _clip(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _rows_to_safe_turns(rows: list[Message]) -> list[MemoryTurn]:
    """Pair user/assistant rows, retaining safe tool results without internal routing metadata."""
    turns: list[MemoryTurn] = []
    pending_user: Message | None = None

    for row in rows:
        role = str(row.role).strip().lower()
        content = str(row.content or "").strip()
        if not content:
            continue

        if role == "user":
            pending_user = row
            continue

        if role == "assistant" and pending_user is not None:
            route_match = _ROUTE_VALUE_LINE.search(content)
            route = (
                route_match.group(1).strip().casefold()
                if route_match
                else None
            )

            if "Cross-tool plan:" in content:
                route = "cross_tool"

            safe_content = _sanitize_assistant_memory(content)

            if _assistant_is_memory_safe(safe_content):
                # Store at the widest ceiling any section could need (MOST_RECENT_TURN_MAX_CHARS).
                # _format_turns() re-clips down to the tighter MAX_MESSAGE_CHARS budget for
                # every section except the most recent turn, so older/background context stays
                # exactly as bounded as before.
                turns.append(
                    MemoryTurn(
                        user_id=int(pending_user.id),
                        assistant_id=int(row.id),
                        user=_clip(str(pending_user.content), MOST_RECENT_TURN_MAX_CHARS),
                        assistant=_clip(safe_content, MOST_RECENT_TURN_MAX_CHARS),
                        route=route,
                    )
                )

            pending_user = None

    return turns


def _load_safe_turns(
    scan_messages: int | None = None,
    session_id: str | None = None,
) -> list[MemoryTurn]:
    """Load recent safe turns, scoped to one session when a session is known.

    Part 10.2 Phase D: passing session_id confines conversational context to the
    ACTIVE session, so a previous unrelated conversation can never surface as
    "MOST RECENT TURN / the current active topic". Omitting it preserves the
    pre-10.2 global scan for callers that have no session identity.
    """
    scan = scan_messages or _env_int(
        "AO_MEMORY_SCAN_MESSAGES", DEFAULT_SCAN_MESSAGES, minimum=20, maximum=5000
    )
    with SessionLocal() as db:
        query = db.query(Message)
        if session_id:
            query = query.filter(Message.session_id == session_id)
        rows = (
            query
            .order_by(Message.id.desc())
            .limit(scan)
            .all()
        )
    rows.reverse()
    return _rows_to_safe_turns(rows)


def _terms(text: str) -> set[str]:
    words = {
        token.lower()
        for token in re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)
    }
    return {word for word in words if word not in _STOPWORDS and not word.isdigit()}


def _turn_relevance(query_terms: set[str], turn: MemoryTurn) -> float:
    if not query_terms:
        return 0.0
    user_terms = _terms(turn.user)
    assistant_terms = _terms(turn.assistant)
    user_overlap = len(query_terms & user_terms)
    assistant_overlap = len(query_terms & assistant_terms)
    # User-authored facts receive more weight than Bunnelby-authored prose.
    return (user_overlap * 2.0) + assistant_overlap


_TOOL_MEMORY_ROUTES: Final[frozenset[str]] = frozenset(
    {"gmail", "calendar", "cross_tool", "file_search"}
)


def _format_turns(turns: list[MemoryTurn], *, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Render turns for the Brain prompt, preserving each turn's trust class.

    Part 10.2 Phase L: a Bunnelby reply whose route was a tool (gmail, calendar,
    cross_tool) is prose Bunnelby wrote, but every fact in it came from
    attacker-controllable external data. Replaying it as plain conversation is
    exactly the laundering path the audit found, so it is re-wrapped as derived
    untrusted content before it re-enters the routing context.

    The user's own turn is always trusted: it is what the user actually said.
    """
    blocks: list[str] = []
    for turn in turns:
        assistant = _clip(turn.assistant, limit)
        if turn.route in _TOOL_MEMORY_ROUTES:
            assistant = wrap_tool_summary(turn.route or "tool", assistant).render()
        blocks.append(f"User: {_clip(turn.user, limit)}\nBunnelby: {assistant}")
    return "\n\n".join(blocks)


_TEMPORAL_RECALL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:last|latest|recent|previous|earlier|before|pehle|pichl[aei]|"
    r"last\s+kya|abhi\s+tak|hamne\s+last)\b",
    re.IGNORECASE,
)

_FOLLOW_UP_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:and\b|also\b|then\b|what\s+about\b|how\s+about\b|"
    r"it\b|that\b|this\b|aur\b|phir\b|toh?\b|iska\b|uska\b)",
    re.IGNORECASE,
)

# A "which one"-style follow-up references a SET/COLLECTION the previous turn already
# returned (several emails, several meetings, several enumerated options, ...). This is
# domain-agnostic on purpose -- it is not Gmail- or Calendar-specific -- and, unlike
# _FOLLOW_UP_REFERENCE_PATTERN above, it is not anchored to the start of the message because
# these phrasings commonly appear mid-sentence ("out of those, which one...").
_COLLECTION_FOLLOW_UP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:which\s+(?:one|ones|is|are)|the\s+first\s+one|the\s+last\s+one|"
    r"the\s+second\s+one|any\s+of\s+(?:them|these|those)|"
    r"all\s+of\s+(?:them|these|those)|none\s+of\s+(?:them|these|those)|"
    r"either\s+of\s+(?:them|these|those))\b",
    re.IGNORECASE,
)

_TOOL_MEMORY_CUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:gmail|e-?mail|mail|inbox|calendar|agenda|meeting|"
    r"schedule|event|availability|file|document|folder|local\s+search)\b",
    re.IGNORECASE,
)



def build_memory_context(
    current_user_message: str,
    session_id: str | None = None,
) -> str:
    """Build bounded local context from profile, recent turns, and relevant old turns.

    When session_id is supplied the whole context is confined to that session.
    Long-term cross-session recall is intentionally out of scope here; it belongs
    to Permanent Memory V2 / the Time Machine, which will retrieve explicitly
    rather than by silently widening the active-topic window.
    """
    profile = load_user_profile()
    all_turns = _load_safe_turns(session_id=session_id)

    configured_recent_count = _env_int(
        "AO_RECENT_MEMORY_TURNS",
        DEFAULT_RECENT_TURNS,
        minimum=1,
        maximum=20,
    )
    relevant_count = _env_int(
        "AO_RELEVANT_OLD_MEMORY_TURNS",
        DEFAULT_RELEVANT_OLD_TURNS,
        minimum=1,
        maximum=10,
    )

    temporal_recall = bool(
        _TEMPORAL_RECALL_PATTERN.search(current_user_message)
    )
    follow_up_reference = bool(
        _FOLLOW_UP_REFERENCE_PATTERN.search(current_user_message)
    ) or bool(
        _COLLECTION_FOLLOW_UP_PATTERN.search(current_user_message)
    )
    explicit_tool_context = bool(
        _TOOL_MEMORY_CUE_PATTERN.search(current_user_message)
    )

    # Tool history is useful when explicitly requested, during temporal recall,
    # or for a direct follow-up. It is harmful noise for unrelated casual chat.
    allow_tool_memory = (
        explicit_tool_context
        or temporal_recall
        or follow_up_reference
    )

    if allow_tool_memory:
        eligible_turns = all_turns
    else:
        eligible_turns = [
            turn
            for turn in all_turns
            if turn.route not in _TOOL_MEMORY_ROUTES
        ]

    # Standalone conversational turns need a small working-memory window.
    # Direct follow-ups are allowed a wider window to preserve continuity.
    recent_count = (
        configured_recent_count
        if follow_up_reference or temporal_recall
        else min(configured_recent_count, 4)
    )

    recent_turns = eligible_turns[-recent_count:]
    recent_ids = {turn.user_id for turn in recent_turns}

    old_pool = (
        eligible_turns[:-recent_count]
        if len(eligible_turns) > recent_count
        else []
    )
    old_turns = [
        turn
        for turn in old_pool
        if turn.user_id not in recent_ids
    ]

    query_terms = _terms(current_user_message)
    ranked_old = sorted(
        (
            (turn, _turn_relevance(query_terms, turn))
            for turn in old_turns
        ),
        key=lambda item: (item[1], item[0].assistant_id),
        reverse=True,
    )

    relevant_old = [
        turn
        for turn, score in ranked_old
        if score > 0
    ][:relevant_count]

    relevant_old.reverse()

    sections = [
        "LOCAL BUNNELBY USER PROFILE (trusted local profile):",
        f"- profile_id: {profile.profile_id}",
        f"- preferred_name: {profile.preferred_name}",
        f"- assistant_name: {profile.assistant_name}",
        f"- mode: {profile.mode}",
    ]

    if recent_turns:
        earlier_recent_turns = recent_turns[:-1]
        most_recent_turn = recent_turns[-1]

        if earlier_recent_turns:
            sections.extend(
                [
                    "\nEARLIER RECENT CONVERSATION (older than the immediately preceding turn, "
                    "oldest to newest -- lower priority background context):",
                    _format_turns(earlier_recent_turns),
                ]
            )

        sections.extend(
            [
                "\nMOST RECENT TURN (the immediately preceding exchange -- this is the current "
                "active topic):",
                _format_turns([most_recent_turn], limit=MOST_RECENT_TURN_MAX_CHARS),
            ]
        )

    if relevant_old:
        sections.extend(
            [
                "\nRELEVANT OLDER LOCAL MEMORY (selected from earlier chats):",
                _format_turns(relevant_old),
            ]
        )

    sections.append(
        "\nMemory rules: Prefer the current user message over older conversation. "
        "For stable identity, prefer the local profile. Treat user-authored statements as "
        "stronger evidence than old Bunnelby-authored statements. Safe Gmail/Calendar/tool "
        "results in recent conversation are valid conversational memory. For temporal recap "
        "questions such as 'what did we last do', 'last kya kaam kiya', or 'most recent work', "
        "answer from the newest recent turns first and do not substitute older topic matches. "
        "Never answer an unrelated conversational message with Gmail, Calendar, or other "
        "tool history. Tool history is context only when the current message actually refers "
        "to it. Do not mention these memory mechanics unless the user asks about memory.\n\n"
        "Reference resolution priority: when the current user message contains an ambiguous "
        "reference such as 'it', 'that', 'this', 'they', or an implicit follow-up (e.g. "
        "'explain it more', 'give a real life example', 'compare that with X') and the current "
        "message is not self-contained, resolve the reference in this order: (1) the current "
        "user message itself, if it already names the topic; (2) the MOST RECENT TURN above -- "
        "treat this as the current active topic and the primary candidate; (3) EARLIER RECENT "
        "CONVERSATION only if the most recent turn offers no plausible referent; (4) RELEVANT "
        "OLDER LOCAL MEMORY only as a last resort, and never to override an obvious antecedent "
        "already present in the most recent turn. Only ask for clarification when the most "
        "recent turn itself introduced multiple distinct, unrelated topics and the current "
        "message gives no way to safely pick one -- do not ask for clarification when the most "
        "recent turn has one clear topic just because that topic's explanation also mentions "
        "related concepts in passing.\n\n"
        "Set/collection references: when the MOST RECENT TURN presented ONE coherent result "
        "set (several emails from one Gmail read, several meetings from one Calendar read, "
        "several files from one search, or several options Bunnelby itself enumerated in one "
        "answer), that whole set is a SINGLE resolvable referent, not multiple competing "
        "topics. A follow-up like 'which one', 'which is most important', 'the first one', or "
        "'any of them' should be resolved by reasoning over the items already listed in that "
        "turn, not treated as ambiguous and not answered by re-fetching the same data."
    )
    return "\n".join(sections)


_NAME_QUERY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+my\s+name|do\s+you\s+(?:know|remember)\s+my\s+name|"
    r"remember\s+my\s+name|who\s+am\s+i)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ASSISTANT_IDENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+are\s+you|what(?:'s|\s+is)\s+your\s+name)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def local_identity_reply(user_message: str) -> str | None:
    """Answer stable identity questions locally to save quota and avoid hallucination."""
    profile = load_user_profile()
    if _NAME_QUERY_PATTERN.match(user_message):
        return f"Your name is {profile.preferred_name}, sir."
    if _ASSISTANT_IDENTITY_PATTERN.match(user_message):
        return f"I'm {profile.assistant_name}, your personal desktop AI assistant."
    return None
