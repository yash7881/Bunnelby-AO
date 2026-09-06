from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Final

# Part 10.2 Phase D: explicit conversational boundaries.
#
# Before this module a Bunnelby turn had no identity. Memory retrieval scanned
# the last N message rows globally, so a fresh desktop launch or a new wake
# conversation inherited the previous session's final exchange as its "current
# active topic" -- the bug class the Brain prompt was compensating for with ~40
# lines of anaphora prose.
#
# A session is one continuous conversation: one desktop chat session, or one
# wake -> follow-up -> timeout voice conversation. A turn is one user message
# plus the assistant response it produced; both rows share the turn id.

# Every message written before Part 10.2 belongs to this archival session. It
# must match database/migrations/versions/0004_add_message_session_turn.py.
LEGACY_SESSION_ID: Final[str] = "legacy-pre-10.2"

SESSION_ID_PREFIX: Final[str] = "sess"
TURN_ID_PREFIX: Final[str] = "turn"
RESULT_SET_ID_PREFIX: Final[str] = "rs"

MAX_IDENTIFIER_CHARS: Final[int] = 128

# Identifiers reach SQL as bound parameters, so this is not an injection guard;
# it keeps caller-supplied ids inspectable, log-safe and bounded.
_SAFE_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True)
class TurnContext:
    """Identity for one user->assistant exchange, threaded through the turn."""

    session_id: str
    turn_id: str


def _token() -> str:
    return uuid.uuid4().hex[:16]


def new_session_id() -> str:
    return f"{SESSION_ID_PREFIX}-{_token()}"


def new_turn_id() -> str:
    return f"{TURN_ID_PREFIX}-{_token()}"


def new_result_set_id(source: str) -> str:
    """Stable id for one coherent result collection (a Gmail read, an agenda...).

    Part 10.2 uses this to label a returned set so a later "which one?" follow-up
    can be tied to the exact collection that produced it, and so untrusted
    external content keeps a provenance handle. Permanent Memory will build on
    these ids; this phase only mints and records them.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", source.strip().casefold()).strip("-") or "unknown"
    return f"{RESULT_SET_ID_PREFIX}-{slug}-{_token()}"


def is_valid_identifier(value: object) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or len(candidate) > MAX_IDENTIFIER_CHARS:
        return False
    return bool(_SAFE_IDENTIFIER.match(candidate))


def resolve_session_id(provided: object = None) -> str:
    """Accept a caller's session id, or mint one.

    Backward compatibility is deliberate: a caller that predates Part 10.2 and
    omits session_id gets a fresh single-turn session rather than an error, and
    is therefore still isolated from earlier conversations.
    """
    if is_valid_identifier(provided):
        return str(provided).strip()
    return new_session_id()


def resolve_turn_context(
    session_id: object = None,
    turn_id: object = None,
) -> TurnContext:
    return TurnContext(
        session_id=resolve_session_id(session_id),
        turn_id=str(turn_id).strip() if is_valid_identifier(turn_id) else new_turn_id(),
    )
