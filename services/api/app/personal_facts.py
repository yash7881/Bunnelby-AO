from __future__ import annotations

import logging
import re
from typing import Final

from .database import SessionLocal
from .models import UserFact, utc_now

logger = logging.getLogger(__name__)

# Personal Facts Memory V1.
#
# Deterministic, pattern-matched extraction only. There is no LLM inference
# step here: a fact is saved only when the user's own message matches one of
# the explicit "X's name is Y" / "X is called/named Y" shapes below. Anything
# ambiguous is left unsaved rather than guessed -- this module never fills in
# a relation Bunnelby was not directly told.
#
# One row per relation key (upsert): a later restatement replaces the earlier
# value. This keeps storage bounded and simple; it does not attempt to track
# multiple people under the same relation (e.g. two brothers).

MAX_FACTS: Final[int] = 50  # generous headroom over the current relation set

_RELATION_CANONICAL: Final[dict[str, str]] = {
    "father": "father",
    "dad": "father",
    "papa": "father",
    "mother": "mother",
    "mom": "mother",
    "mum": "mother",
    "maa": "mother",
    "wife": "spouse",
    "husband": "spouse",
    "spouse": "spouse",
    "brother": "brother",
    "sister": "sister",
    "son": "son",
    "daughter": "daughter",
    "best friend": "best_friend",
    "friend": "friend",
    "boss": "boss",
    "manager": "manager",
}

_RELATION_LABELS: Final[dict[str, str]] = {
    "self": "your name",
    "father": "your father's name",
    "mother": "your mother's name",
    "spouse": "your spouse's name",
    "brother": "your brother's name",
    "sister": "your sister's name",
    "son": "your son's name",
    "daughter": "your daughter's name",
    "best_friend": "your best friend's name",
    "friend": "your friend's name",
    "boss": "your boss's name",
    "manager": "your manager's name",
}

_RELATION_WORDS_PATTERN: Final[str] = "|".join(
    sorted(_RELATION_CANONICAL, key=len, reverse=True)
).replace(" ", r"\s+")

# "'s" is optional and may be typed/transcribed as a bare trailing "s"
# ("fathers name is" from voice STT with no apostrophe).
_NAME_STATEMENT: Final[re.Pattern[str]] = re.compile(
    r"\bmy\s+(?P<relation_word>" + _RELATION_WORDS_PATTERN + r")(?:'s|s)?\s+"
    r"(?:name\s+is|is\s+(?:called|named))\s+"
    r"(?P<value>\S+(?:\s+\S+){0,3}?)"
    r"(?=\s*[.,!?\n]|\s+(?:and|who|because|that|which)\b|\s*$)",
    re.IGNORECASE,
)

_SELF_NAME_STATEMENT: Final[re.Pattern[str]] = re.compile(
    r"\bmy\s+name\s+is\s+"
    r"(?P<value>\S+(?:\s+\S+){0,3}?)"
    r"(?=\s*[.,!?\n]|\s+(?:and|who|because|that|which)\b|\s*$)",
    re.IGNORECASE,
)

_LEADING_REJECT_WORDS: Final[frozenset[str]] = frozenset(
    {"not", "no", "n't", "unsure", "sure", "unknown", "none", "null", "same"}
)


def _clean_value(raw: str) -> str | None:
    value = re.sub(r"\s+", " ", raw).strip(" .,!?'\"‘’“”")
    if not value or len(value) > 80:
        return None
    first_word = value.split(" ", 1)[0].casefold()
    if first_word in _LEADING_REJECT_WORDS:
        return None
    return value


def extract_personal_fact(user_message: str) -> tuple[str, str] | None:
    """Return (relation, value) if the message explicitly states a fact, else None.

    Only ever returns one fact per message. Never guesses: a message that
    does not match one of the explicit naming shapes yields None.
    """
    text = (user_message or "").strip()
    if not text:
        return None

    match = _NAME_STATEMENT.search(text)
    if match:
        relation_word = " ".join(match.group("relation_word").lower().split())
        relation = _RELATION_CANONICAL.get(relation_word)
        if relation:
            value = _clean_value(match.group("value"))
            if value:
                return relation, value

    match = _SELF_NAME_STATEMENT.search(text)
    if match:
        value = _clean_value(match.group("value"))
        if value:
            return "self", value

    return None


def save_personal_fact(
    relation: str,
    value: str,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> None:
    """Upsert one fact by relation key. Local SQLite only; no cloud sync."""
    db = SessionLocal()
    try:
        existing = db.query(UserFact).filter(UserFact.relation == relation).one_or_none()
        now = utc_now()
        if existing is not None:
            existing.value = value
            existing.source_session_id = session_id
            existing.source_turn_id = turn_id
            existing.updated_at = now
        else:
            db.add(
                UserFact(
                    relation=relation,
                    value=value,
                    source_session_id=session_id,
                    source_turn_id=turn_id,
                    created_at=now,
                    updated_at=now,
                )
            )
        db.commit()
    finally:
        db.close()


def load_personal_facts() -> dict[str, str]:
    """All currently known facts, relation -> value. Bounded, local read only."""
    db = SessionLocal()
    try:
        rows = (
            db.query(UserFact)
            .order_by(UserFact.relation.asc())
            .limit(MAX_FACTS)
            .all()
        )
        return {row.relation: row.value for row in rows}
    finally:
        db.close()


def try_save_stated_fact(
    user_message: str,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> tuple[str, str] | None:
    """Detect and persist an explicitly stated fact in one step. Returns it, or None."""
    extracted = extract_personal_fact(user_message)
    if extracted is None:
        return None
    relation, value = extracted
    save_personal_fact(relation, value, session_id=session_id, turn_id=turn_id)
    logger.info("personal_facts: saved relation=%s", relation)
    return extracted


def fact_saved_reply(fact: tuple[str, str]) -> str:
    """A short, local, deterministic acknowledgment -- no LLM call needed."""
    relation, value = fact
    label = _RELATION_LABELS.get(relation, relation)
    return f"Got it — I'll remember that {label} is {value}."


def build_personal_facts_context() -> str:
    """Render known facts for Brain context injection. Empty string if none.

    Kept as a section separate from build_memory_context()'s session-scoped
    conversational memory: these facts are global and must survive a new
    session or app restart, by design.
    """
    facts = load_personal_facts()
    if not facts:
        return ""
    lines = [
        "PERSISTED PERSONAL FACTS (user-stated, local only, survives restarts -- "
        "use these to answer identity/family questions):"
    ]
    for relation, value in facts.items():
        label = _RELATION_LABELS.get(relation, relation)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


__all__ = [
    "MAX_FACTS",
    "extract_personal_fact",
    "save_personal_fact",
    "load_personal_facts",
    "try_save_stated_fact",
    "fact_saved_reply",
    "build_personal_facts_context",
]
