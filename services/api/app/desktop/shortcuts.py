"""Allowlisted keyboard shortcuts. Closed set, individually safety-reviewed.

There is no "press these keys" capability in Bunnelby. A request names a
SHORTCUT ID (`show_desktop`), and this module is the only place an id becomes
key codes. A model therefore cannot express `Ctrl+Alt+Del`, `Win+L`, `Alt+F4`
or an arbitrary string at all -- those are not ids, so they fail to resolve.

Two layers, because "not on the allowlist" and "explicitly dangerous" deserve
different answers:

  1. ALLOWLIST  -- the only combinations that can ever be sent.
  2. DENYLIST   -- combinations we recognise by name purely so an attempt is
                   refused with a specific reason and audited, instead of
                   producing a vague "unknown shortcut".

The denylist is defence in depth, not the control. Even if it were empty,
nothing outside the allowlist can be synthesised.

WHY Ctrl+Alt+Del AND Win+L CAN NEVER WORK ANYWAY
------------------------------------------------
Both are Secure Attention Sequences handled by the OS below the input queue.
`SendInput` from a normal-integrity process cannot generate them; Windows
reserves them precisely so software cannot fake a credential prompt. We refuse
them explicitly so the refusal is intentional and logged rather than incidental.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Final, Mapping

from ..risk_policy import RiskLevel
from .models import ShortcutNotAllowedError

# Virtual-key codes (winuser.h).
VK_TAB: Final[int] = 0x09
VK_LWIN: Final[int] = 0x5B
VK_MENU: Final[int] = 0x12  # ALT
VK_D: Final[int] = 0x44
VK_E: Final[int] = 0x45


@dataclass(frozen=True, slots=True)
class Shortcut:
    """One reviewed key combination."""

    shortcut_id: str
    display: str
    key_codes: tuple[int, ...]
    risk_level: RiskLevel
    rationale: str
    #: True when the combination only rearranges what is on screen and cannot
    #: destroy work, dismiss a dialog, or answer a prompt.
    non_destructive: bool = True


_ALLOWED: Final[tuple[Shortcut, ...]] = (
    Shortcut(
        shortcut_id="show_desktop",
        display="Win+D",
        key_codes=(VK_LWIN, VK_D),
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        rationale=(
            "Minimises/restores all windows. Reversible by repeating it, closes "
            "nothing, and cannot answer a dialog."
        ),
    ),
    Shortcut(
        shortcut_id="open_file_explorer",
        display="Win+E",
        key_codes=(VK_LWIN, VK_E),
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        rationale="Opens a new File Explorer window. Additive only.",
    ),
    Shortcut(
        shortcut_id="switch_window",
        display="Alt+Tab",
        key_codes=(VK_MENU, VK_TAB),
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        rationale=(
            "Switches foreground window. Included because FOCUS_APP is the "
            "deterministic path and this is the generic fallback; it changes "
            "focus only."
        ),
    ),
)

# Recognised-and-refused. Each carries the reason so the audit row is specific.
_DENIED: Final[Mapping[str, str]] = {
    "ctrl_alt_delete": "Secure Attention Sequence; the secure desktop must never be automated.",
    "win_l": "Locks the workstation and reaches the secure desktop.",
    "alt_f4": "Closes the foreground window destructively and can bypass a save prompt.",
    "ctrl_shift_esc": "Opens Task Manager, which can terminate arbitrary processes.",
    "win_r": "Opens the Run dialog, which is an arbitrary-command surface.",
    "win_x": "Opens the power-user menu, which exposes elevation and shutdown.",
    "ctrl_w": "Closes the active document/tab; can discard unsaved work.",
    "alt_tab_hold": "Held-key sequences are not expressible; only discrete combinations are.",
}

_BY_ID: Final[Mapping[str, Shortcut]] = {item.shortcut_id: item for item in _ALLOWED}


def allowed_shortcut_ids() -> tuple[str, ...]:
    return tuple(sorted(_BY_ID))


def all_shortcuts() -> tuple[Shortcut, ...]:
    return _ALLOWED


def is_explicitly_denied(shortcut_id: str) -> str | None:
    """Return the refusal reason for a recognised-dangerous id, else None."""
    return _DENIED.get(_normalize(shortcut_id))


def _normalize(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def resolve(shortcut_id: str) -> Shortcut:
    """Resolve an id to a reviewed shortcut, or refuse with a specific reason."""
    key = _normalize(shortcut_id)
    denied = _DENIED.get(key)
    if denied is not None:
        raise ShortcutNotAllowedError(f"{shortcut_id!r} is not permitted: {denied}")
    shortcut = _BY_ID.get(key)
    if shortcut is None:
        raise ShortcutNotAllowedError(
            f"{shortcut_id!r} is not an allowlisted shortcut. "
            f"Allowed: {', '.join(allowed_shortcut_ids())}"
        )
    return shortcut


# --------------------------------------------------------------------------- #
# Sending. Isolated here so the allowlist and the only SendInput call in the
# codebase live in the same file and are reviewed together.
# --------------------------------------------------------------------------- #

_KEYEVENTF_KEYUP: Final[int] = 0x0002
_INPUT_KEYBOARD: Final[int] = 1


def send(shortcut: Shortcut) -> bool:
    """Press and release a reviewed combination. Windows only.

    Accepts a `Shortcut` OBJECT, not an id or a string of keys: the only way to
    obtain one is through `resolve()`, so an unreviewed combination cannot
    reach this function even by mistake.
    """
    if not isinstance(shortcut, Shortcut):
        raise ShortcutNotAllowedError("send() accepts allowlisted Shortcut objects only")
    canonical = _BY_ID.get(shortcut.shortcut_id)
    if canonical is None or shortcut != canonical:
        raise ShortcutNotAllowedError("send() accepts allowlisted Shortcut objects only")

    if sys.platform != "win32":
        return False

    import ctypes
    from ctypes import wintypes

    class _KeyBdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _InputUnion(ctypes.Union):
        _fields_ = [("ki", _KeyBdInput)]

    class _Input(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    def _event(code: int, keyup: bool) -> _Input:
        return _Input(
            type=_INPUT_KEYBOARD,
            union=_InputUnion(
                ki=_KeyBdInput(
                    wVk=code,
                    wScan=0,
                    dwFlags=_KEYEVENTF_KEYUP if keyup else 0,
                    time=0,
                    dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
                )
            ),
        )

    # Press in order, release in reverse: that is what makes it a chord rather
    # than a sequence of independent keystrokes.
    events = [_event(code, False) for code in shortcut.key_codes]
    events += [_event(code, True) for code in reversed(shortcut.key_codes)]
    array = (_Input * len(events))(*events)
    sent = user32.SendInput(len(events), array, ctypes.sizeof(_Input))
    return sent == len(events)
