"""Part 12.1 domain model for bounded Windows desktop control.

Everything the desktop capability can express lives here as a closed type. The
governing rule mirrors `tool_requests`: THE TYPE FIXES THE ACTION CLASS. A
caller may refine fields inside a chosen action; it can never widen the action
set, and it can never name an executable, a path, a window handle it invented,
or a keystroke that is not on the allowlist.

Nothing in this module touches Windows. It is pure data so the policy layer is
fully testable on any platform, which is also why the Windows backends sit
behind Protocols rather than being imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Literal, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Bounds. Every one of these exists to stop an unbounded read reaching either
# the model's context or the evidence ledger.
# --------------------------------------------------------------------------- #
MAX_WINDOWS_RETURNED: Final[int] = 40
MAX_UI_DEPTH: Final[int] = 4
MAX_UI_NODES: Final[int] = 120
MAX_UI_TEXT_CHARS: Final[int] = 120
MAX_TITLE_CHARS: Final[int] = 160
MAX_EVIDENCE_CHARS: Final[int] = 800

# Wall-clock ceilings. No desktop call may block indefinitely.
DEFAULT_ACTION_TIMEOUT_SECONDS: Final[float] = 10.0
MIN_ACTION_TIMEOUT_SECONDS: Final[float] = 1.0
MAX_ACTION_TIMEOUT_SECONDS: Final[float] = 30.0
LAUNCH_VERIFY_TIMEOUT_SECONDS: Final[float] = 12.0
FOCUS_VERIFY_TIMEOUT_SECONDS: Final[float] = 3.0
CLOSE_VERIFY_TIMEOUT_SECONDS: Final[float] = 6.0
POLL_INTERVAL_SECONDS: Final[float] = 0.1

#: How long a registered application may be invisible to window enumeration
#: before it is declared absent. Packaged (Store/UWP) apps are re-hosted
#: between CalculatorApp.exe and ApplicationFrameHost.exe during their
#: lifetime, and a single enumeration taken mid-transition sees no window at
#: all. Hardware evidence: open_app verified Calculator, and a later single
#: snapshot reported it "not currently running" while both processes were
#: alive. This is a bounded rediscovery window, not a debounce or a retry
#: budget -- the poll uses the same deadline machinery as every other wait.
DISCOVERY_GRACE_SECONDS: Final[float] = 1.5


class DesktopAction(str, Enum):
    """The complete, closed set of Part 12.1 desktop actions.

    Deliberately absent, and deferred to later milestones: arbitrary clicking,
    coordinate input, text entry into fields, screen capture, and any form of
    scripted multi-step automation.
    """

    LIST_WINDOWS = "list_windows"
    INSPECT_WINDOW = "inspect_window"
    FIND_CONTROL = "find_control"
    OPEN_APP = "open_app"
    FOCUS_APP = "focus_app"
    CLOSE_APP = "close_app"
    SAFE_SHORTCUT = "safe_shortcut"

    @property
    def changes_state(self) -> bool:
        """True when the action can alter desktop state, so it needs a verifier."""
        return self in _STATE_CHANGING


_STATE_CHANGING: Final[frozenset[DesktopAction]] = frozenset(
    {
        DesktopAction.OPEN_APP,
        DesktopAction.FOCUS_APP,
        DesktopAction.CLOSE_APP,
        DesktopAction.SAFE_SHORTCUT,
    }
)

READ_ONLY_ACTIONS: Final[frozenset[DesktopAction]] = frozenset(
    {
        DesktopAction.LIST_WINDOWS,
        DesktopAction.INSPECT_WINDOW,
        DesktopAction.FIND_CONTROL,
    }
)

class MatchEvidence(str, Enum):
    """How confidently a window was attributed to a registered application.

    Identity is graded rather than boolean because Windows itself is graded. A
    window owned by `notepad.exe` IS Notepad. A window owned by
    `ApplicationFrameHost.exe` -- the shared frame host for every packaged
    (Store/UWP) app -- is only whichever app its title says it is, because the
    process name alone identifies nothing. And a window whose owning process we
    were not permitted to read is a guess, however plausible.

    The distinction is load-bearing: TITLE_ONLY evidence is enough to LOOK at a
    window, and never enough to CLOSE one. An arbitrary program can name its
    window anything it likes.
    """

    #: The owning process is one this app exclusively uses. Sufficient alone.
    DEDICATED_PROCESS = "dedicated_process"
    #: The owning process is a declared shared host AND the title corroborates
    #: this specific app. Both halves are required.
    SHARED_HOST_TITLE = "shared_host_title"
    #: The owning process could not be read; only the title suggests this app.
    TITLE_ONLY = "title_only"

    @property
    def is_strong(self) -> bool:
        """True when the window's identity was corroborated by its process.

        Only strong evidence may authorise a state-changing action on the
        window itself.
        """
        return self is not MatchEvidence.TITLE_ONLY


# Outcome vocabulary. `unverified` is first-class and distinct from `failed`:
# "I did the thing but could not prove the result" must never be reported as
# success, and must also not be reported as a clean failure.
DesktopStatus = Literal[
    "succeeded",
    "failed",
    "unverified",
    "blocked",
    "needs_clarification",
]

# Stable, loggable error taxonomy. Free-text reasons are for humans; these are
# for the evidence ledger and for tests.
ErrorCode = Literal[
    "unknown_application",
    "application_not_running",
    "application_already_closed",
    "ambiguous_target",
    "window_not_found",
    "close_not_permitted",
    "shortcut_not_allowed",
    "confirmation_required",
    "foreground_denied",
    "launch_failed",
    "timeout",
    "provider_unavailable",
    "platform_unsupported",
    "internal_error",
]


class DesktopError(RuntimeError):
    """Base for every desktop failure that must not become a success claim."""

    code: ErrorCode = "internal_error"

    def __init__(self, message: str, *, code: ErrorCode | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class UnknownApplicationError(DesktopError):
    """The requested target is not in the deterministic application registry."""

    code: ErrorCode = "unknown_application"


class ActionBlockedError(DesktopError):
    """Policy refused the action outright; it was never attempted."""

    code: ErrorCode = "close_not_permitted"


class ShortcutNotAllowedError(DesktopError):
    """The requested key combination is not on the reviewed allowlist."""

    code: ErrorCode = "shortcut_not_allowed"


class DesktopTimeoutError(DesktopError):
    """A bounded wait elapsed without the expected state being observed."""

    code: ErrorCode = "timeout"


class ProviderUnavailableError(DesktopError):
    """A required backend (Win32 or UI Automation) is not usable here."""

    code: ErrorCode = "provider_unavailable"


def _clip(value: str | None, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """One visible top-level window.

    `title` is UNTRUSTED external data: it is attacker-influenceable by any
    running program. It is clipped here and must be wrapped by
    `untrusted_content` before it can reach a prompt.
    """

    handle: int
    title: str
    pid: int
    process_name: str
    is_foreground: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _clip(self.title, MAX_TITLE_CHARS))
        object.__setattr__(self, "process_name", _clip(self.process_name, 64).lower())

    def summary(self) -> dict[str, Any]:
        """Safe projection for evidence and API responses."""
        return {
            "handle": self.handle,
            "title": self.title,
            "pid": self.pid,
            "process_name": self.process_name,
            "is_foreground": self.is_foreground,
        }


@dataclass(frozen=True, slots=True)
class UiNode:
    """One bounded UI Automation element.

    Values are deliberately NOT captured. A text box's contents can be a
    password or an OAuth code, so Part 12.1 records only the element's identity
    and state, never what it holds.
    """

    name: str
    control_type: str
    automation_id: str = ""
    enabled: bool = True
    offscreen: bool = False
    depth: int = 0
    children: tuple["UiNode", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clip(self.name, MAX_UI_TEXT_CHARS))
        object.__setattr__(self, "automation_id", _clip(self.automation_id, 64))

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)

    def flatten(self) -> tuple["UiNode", ...]:
        out: list[UiNode] = [self]
        for child in self.children:
            out.extend(child.flatten())
        return tuple(out)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control_type": self.control_type,
            "automation_id": self.automation_id,
            "enabled": self.enabled,
            "offscreen": self.offscreen,
            "depth": self.depth,
            "children": [child.summary() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class DesktopOutcome:
    """The single result type every desktop action returns.

    A caller must consult `status`. There is intentionally no boolean `ok`:
    `unverified` is neither success nor failure and must not collapse into one.
    """

    action: DesktopAction
    status: DesktopStatus
    target: str = ""
    detail: str = ""
    error_code: ErrorCode | None = None
    windows: tuple[WindowInfo, ...] = ()
    nodes: tuple[UiNode, ...] = ()
    latency_ms: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def audit_payload(self) -> dict[str, Any]:
        """Bounded, privacy-safe projection for the evidence ledger.

        Window titles and UI names are counted and identified, not dumped: a
        tool_runs row must never become a transcript of the user's screen.
        """
        payload: dict[str, Any] = {
            "action": self.action.value,
            "status": self.status,
            "target": self.target,
            "latency_ms": round(self.latency_ms, 1),
            "window_count": len(self.windows),
            "node_count": sum(node.node_count() for node in self.nodes),
        }
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.detail:
            payload["detail"] = _clip(self.detail, 200)
        for key, value in self.evidence.items():
            payload[f"evidence_{key}"] = value
        return payload


def bounded_timeout(value: float | None) -> float:
    """Clamp any caller-supplied timeout into the permitted band."""
    if value is None:
        return DEFAULT_ACTION_TIMEOUT_SECONDS
    return max(MIN_ACTION_TIMEOUT_SECONDS, min(float(value), MAX_ACTION_TIMEOUT_SECONDS))


def limit_windows(windows: Sequence[WindowInfo]) -> tuple[WindowInfo, ...]:
    """Enforce the window-count ceiling at the boundary, not at each caller."""
    return tuple(windows[:MAX_WINDOWS_RETURNED])
