"""Local Fast Path: deterministic routing for a few unambiguous desktop commands.

WHAT THIS IS
------------
"Open Notepad" is a local Windows operation, but today it still costs a cloud
Brain call to classify. This module recognises a small, closed set of literal
desktop commands and produces the SAME canonical BrainDecision the Brain would
have produced, so the turn skips the provider round-trip.

It is a ROUTING optimisation and nothing else. A match here still flows through
tool_executor.execute -> DesktopControlRequest -> Capability Registry -> risk
policy -> DesktopController -> verifier -> evidence, exactly as a Brain-routed
turn does. This module never touches subprocess, never touches Win32, never
reports success, and never returns a reply. A parser hit is not proof of
execution; the verifier still owns that.

WHY THIS IS NOT THE MISTAKE WE ALREADY MADE
-------------------------------------------
`intelligence_dispatch` once ran a Gmail+Calendar KEYWORD gate before the Brain
saw the message. It was removed because "Explain the difference between Gmail
and Google Calendar" contained both keyword sets and fired real API calls on a
purely conceptual question. That failure had four ingredients, and this module
has none of them:

  1. It matched SUBSTRINGS anywhere in the message. Here the ENTIRE normalised
     utterance must fullmatch a reviewed template, so any extra clause
     ("Open Notepad and send an email") is a miss.
  2. It could reach EXTERNAL, irreversible effects (sending mail, creating
     events). Here the reachable set is four local desktop actions, all
     reversible, all still policy-gated downstream.
  3. It resolved targets loosely. Here the target must resolve EXACTLY through
     app_registry, which has no fuzzy matching, no PATH search and no path
     interpretation. Anything unresolvable is a miss.
  4. It had no negation handling. Here negation short-circuits to the Brain
     before any action matching happens.

The rule that follows: WHEN IN DOUBT, RETURN None. Falling through to the Brain
costs a cloud call. Guessing costs correctness.

SCOPE
-----
open_app, focus_app, close_app, list_windows. Nothing else -- no shortcuts, no
UI inspection, no Gmail, no Calendar, no file search, no shell.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final, Mapping, Pattern

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Bounds. A command is a handful of words; anything longer is prose.
# --------------------------------------------------------------------------- #
MAX_COMMAND_CHARS: Final[int] = 80
MAX_COMMAND_WORDS: Final[int] = 8
MAX_TARGET_CHARS: Final[int] = 48

#: Target shape, mirroring DesktopControlRequest._alias_only. The registry is
#: already the authority, but a path- or switch-shaped string must not even be
#: offered to it: every token has to start with a letter or digit, so
#: "c:/windows/system32/cmd.exe" and "app --quiet" cannot be alias-shaped.
_TARGET_SHAPE: Final[Pattern[str]] = re.compile(r"[a-z][a-z0-9]*(?:[ _-][a-z0-9]+)*")

#: Negation hands the turn to the Brain, unconditionally. "Don't open
#: Calculator" and "Calculator mat kholo" must never become open_app. No attempt
#: is made to reason about scope or double negatives: presence is enough.
_NEGATION_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "no",
        "not",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "cant",
        "cannot",
        "never",
        "mat",
        "nahi",
        "nahin",
        "na",
    }
)

#: Words that make an utterance a question or a request for explanation rather
#: than a command. The anchored templates already exclude most of these; the
#: explicit guard documents the intent and survives future template edits.
_CONCEPTUAL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "what",
        "whats",
        "how",
        "why",
        "when",
        "who",
        "whether",
        "explain",
        "describe",
        "tell",
        "mean",
        "means",
        "meaning",
        "difference",
        "can",
        "could",
        "should",
        "would",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "if",
        "because",
        "then",
        "and",
        "kya",
        "kaise",
        "kyun",
        "matlab",
    }
)

#: Exact utterances that ask for the open window list. A fixed set rather than a
#: template: there is no target to extract, so there is nothing to parameterise
#: and no reason to accept anything approximate. Because each phrase is reviewed
#: in full, it is matched BEFORE the heuristic guards below -- "which windows
#: are open" legitimately contains "are", and a negated variant simply is not a
#: member of the set.
_LIST_WINDOWS_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "which windows are open",
        "which windows do i have open",
        "what windows are open",
        "what windows do i have open",
        "show open windows",
        "show the open windows",
        "show me open windows",
        "show me the open windows",
        "show my open windows",
        "list open windows",
        "list the open windows",
        "list my open windows",
        "list all open windows",
        "open windows",
        "windows dikhao",
        "open windows dikhao",
        "khuli windows dikhao",
    }
)

# --------------------------------------------------------------------------- #
# Grammar. Every pattern is fullmatch-anchored against the WHOLE normalised
# utterance, so an extra clause can never be ignored.
# --------------------------------------------------------------------------- #
_TEMPLATES: Final[tuple[tuple[str, Pattern[str]], ...]] = (
    # -- English: verb first ------------------------------------------------- #
    ("open_app", re.compile(r"(?:open|launch|start)\s+(?P<target>.+)")),
    ("focus_app", re.compile(r"(?:switch\s+to|switch|focus\s+on|focus)\s+(?P<target>.+)")),
    ("close_app", re.compile(r"(?:close|quit)\s+(?P<target>.+)")),
    # -- Hinglish: target first, verb last ----------------------------------- #
    # A closed list of exact phrasings. No general Hindi parsing is attempted.
    (
        "open_app",
        re.compile(r"(?P<target>.+?)\s+(?:kholo|khol\s+do|open\s+karo|chalu\s+karo)"),
    ),
    (
        "focus_app",
        re.compile(r"(?P<target>.+?)\s+(?:pe|par)\s+switch\s+(?:karo|kar\s+do)"),
    ),
    (
        "close_app",
        re.compile(
            r"(?P<target>.+?)\s+(?:band\s+karo|band\s+kar\s+do|bandh\s+karo|close\s+karo)"
        ),
    ),
)

#: Filler that may wrap a command without changing it. Removed once, exactly.
_LEADING_VOCATIVE: Final[Pattern[str]] = re.compile(r"^(?:hey\s+|ok\s+)?bunnelby\s*[,:]?\s+")
_LEADING_PLEASE: Final[Pattern[str]] = re.compile(r"^please\s+")
_TRAILING_PLEASE: Final[Pattern[str]] = re.compile(r"\s+please$")
_TERMINAL_PUNCTUATION: Final[str] = ".!?"
_APOSTROPHES: Final[str] = "'\u2019\u02bc"


def normalize(message: str) -> str:
    """Fold an utterance to its comparable form. Deterministic and bounded.

    Whitespace, case and terminal punctuation only, plus a single pass at
    stripping a leading vocative and a leading/trailing "please". There is no
    spell correction, no fuzzy folding, no synonym rewriting and no attempt to
    extract a target out of prose -- each of those would turn a lookup into an
    interpretation.
    """
    text = " ".join(str(message or "").strip().casefold().split())
    text = text.rstrip(_TERMINAL_PUNCTUATION).strip()
    text = _LEADING_VOCATIVE.sub("", text, count=1)
    text = _LEADING_PLEASE.sub("", text, count=1)
    text = _TRAILING_PLEASE.sub("", text, count=1)
    return " ".join(text.split())


def _bare(word: str) -> str:
    """A word with apostrophes removed, so "don't" compares as "dont"."""
    for mark in _APOSTROPHES:
        word = word.replace(mark, "")
    return word


def _is_guarded(words: tuple[str, ...]) -> bool:
    """True when the utterance must go to the Brain whatever its shape."""
    return any(
        _bare(word) in _NEGATION_TOKENS or _bare(word) in _CONCEPTUAL_TOKENS
        for word in words
    )


def _resolved_app_id(raw_target: str) -> str | None:
    """Canonical app_id for a spoken target, or None.

    app_registry is the SINGLE authority for what may be controlled. This adds
    no allowlist of its own; it only declines to hand the registry anything that
    is not alias-shaped in the first place.
    """
    from .desktop import app_registry

    target = " ".join(str(raw_target or "").strip().split())
    if not target or len(target) > MAX_TARGET_CHARS:
        return None
    if not _TARGET_SHAPE.fullmatch(target):
        return None
    entry = app_registry.try_resolve(target)
    return entry.app_id if entry is not None else None


def match_local_command(message: str) -> Mapping[str, str] | None:
    """Canonical desktop arguments for a recognised command, else None.

    Returns only what DesktopControlRequest already accepts: an `action` from
    the canonical vocabulary and, where the action needs one, a `target` that is
    a registry app_id. Never a reply, never a verdict, never a side effect.
    """
    text = normalize(message)
    if not text or len(text) > MAX_COMMAND_CHARS:
        return None

    words = tuple(text.split())
    if len(words) > MAX_COMMAND_WORDS:
        return None

    # Reviewed exact phrases first: they are read-only, carry no target, and
    # cannot smuggle a clause past a whole-string equality test.
    if text in _LIST_WINDOWS_PHRASES:
        return {"action": "list_windows"}

    if _is_guarded(words):
        return None

    for action, pattern in _TEMPLATES:
        found = pattern.fullmatch(text)
        if found is None:
            continue
        app_id = _resolved_app_id(found.group("target"))
        if app_id is None:
            # A recognised verb with an unresolvable target is a MISS, not a
            # refusal: "Open random.exe" and "Open Notepad and Calculator" both
            # belong to the Brain and the typed boundary, which already refuse
            # them correctly. Inventing a local refusal here would make this
            # module a semantic authority, which it must not become.
            return None
        return {"action": action, "target": app_id}

    return None


def try_local_fast_path(message: str) -> Any | None:
    """A canonical BrainDecision for a recognised desktop command, else None.

    The returned decision is indistinguishable from one the Brain would have
    produced for the same command, so the caller feeds it straight into the
    existing tool_executor path with no special casing.
    """
    from .brain_agent import BrainDecision

    arguments = match_local_command(message)
    if arguments is None:
        # Deliberately DEBUG: a miss is the common case and must not spam logs.
        logger.debug("local_fast_path miss")
        return None

    # Bounded and non-sensitive: a canonical action and a registry app_id, never
    # the user's raw text.
    logger.info(
        "local_fast_path matched tool=desktop_control action=%s target=%s",
        arguments["action"],
        arguments.get("target", ""),
    )
    return BrainDecision(
        mode="tool",
        tool="desktop_control",
        confidence=1.0,
        arguments=dict(arguments),
        # No reply and no spoken_reply: the desktop capability produces the
        # user-facing text from VERIFIED outcome state. A parser must never
        # pre-write "Opened Notepad."
        reply="",
        spoken_reply="",
        reason="matched a reviewed local desktop command; no cloud call needed",
        reason_code="local_fast_path",
    )


__all__ = ["match_local_command", "normalize", "try_local_fast_path"]
