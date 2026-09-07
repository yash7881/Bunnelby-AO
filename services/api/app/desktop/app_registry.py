"""Deterministic allowlist of applications Bunnelby may control.

WHY A REGISTRY AND NOT A PATH
-----------------------------
The single most dangerous shape this feature could take is "the model names a
program and we run it". That is arbitrary code execution wearing a helpful hat.
So the desktop request carries a canonical APP ID (`notepad`), never a path,
never a command line, and never a search term. This module is the only place an
id becomes something executable, and every launch argv here is a literal
written by a human.

Resolution is therefore closed: an id either matches a hand-reviewed entry or
the action fails. There is no fallback search of PATH, no Start-menu lookup, no
`where.exe`, and no "did you mean".

LAUNCH STRATEGY
---------------
Two strategies, both argv-list based (never a shell string):

  EXECUTABLE  -> a bare executable name resolved by the OS loader against the
                 system directories, e.g. ["notepad.exe"].
  SHELL_URI   -> a Windows shell/URI target launched via the Explorer handler,
                 e.g. ["explorer.exe", "shell:AppsFolder\\..."]. Used for the
                 packaged (Store/UWP) apps that have no plain executable.

No shell is ever invoked anywhere in this package: every launch goes through
`subprocess.Popen(argv, shell=False)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Mapping

from ..risk_policy import RiskLevel
from .models import MatchEvidence, UnknownApplicationError

# An app id is an alias, exactly like FileSearchRequest.root_scope. The pattern
# is what makes "C:\\Windows\\System32\\cmd.exe" and "notepad & calc" unable to
# even parse as a target.
APP_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_]{0,31}")


class LaunchStrategy(Enum):
    EXECUTABLE = "executable"
    SHELL_URI = "shell_uri"


@dataclass(frozen=True, slots=True)
class ApplicationEntry:
    """One reviewed, controllable application."""

    app_id: str
    display_name: str
    launch_strategy: LaunchStrategy
    launch_argv: tuple[str, ...]
    #: Executable basenames (lowercase) used EXCLUSIVELY by this app. A window
    #: owned by one of these is this app, with no further corroboration.
    process_names: tuple[str, ...] = ()
    #: Executable basenames (lowercase) that host this app but also host
    #: OTHERS -- ApplicationFrameHost.exe hosts every packaged app on the
    #: machine. A window owned by one of these is attributed to this app only
    #: when its title also corroborates it. Listing a shared host in
    #: `process_names` instead would let any packaged window impersonate this
    #: entry, which is exactly the bug this field exists to prevent.
    shared_host_process_names: tuple[str, ...] = ()
    #: Lowercase substrings that identify this app by window title. Corroborate
    #: a shared host, and -- only where `allow_title_only_match` is set -- stand
    #: alone when the owning process could not be read at all.
    title_hints: tuple[str, ...] = ()
    #: Natural-language aliases a user may say. Matched exactly after
    #: normalization; never fuzzy-matched.
    aliases: tuple[str, ...] = ()
    #: Whether CLOSE_APP may target this entry at all.
    closable: bool = True
    #: Risk of the *most dangerous* action permitted on this entry.
    risk_level: RiskLevel = RiskLevel.L1_SAFE_CONTROL
    #: Set for shell/system infrastructure that must never be terminated.
    system_critical: bool = False
    #: Reviewed opt-in: when the owning process name is UNREADABLE, may a title
    #: hint alone attribute a window to this app? Only meaningful for packaged
    #: apps, whose host process can legitimately deny a query. Such a match is
    #: always weak (MatchEvidence.TITLE_ONLY) and never authorises a close.
    allow_title_only_match: bool = False
    notes: str = ""

    def matches_process(self, process_name: str) -> bool:
        """True for a process this app uses EXCLUSIVELY. Shared hosts are not
        included here -- see `match_window` for the full rule."""
        return process_name.strip().lower() in self.process_names

    def matches_title(self, title: str) -> bool:
        lowered = title.strip().lower()
        return any(hint in lowered for hint in self.title_hints)

    def match_window(self, process_name: str, title: str) -> MatchEvidence | None:
        """Attribute a window to this app, reporting HOW it was identified.

        The order is deliberate and closed:

          1. a dedicated process is sufficient on its own;
          2. a declared shared host requires the title to corroborate;
          3. any OTHER readable process name is a definitive non-match -- a
             window that belongs to a known different program is never this
             app, whatever its title claims;
          4. only when the process could not be read at all may a reviewed
             title hint stand alone, and then only weakly.

        Returns None for no match. Never fuzzy-matches.
        """
        name = process_name.strip().lower()
        if name:
            if name in self.process_names:
                return MatchEvidence.DEDICATED_PROCESS
            if name in self.shared_host_process_names:
                return (
                    MatchEvidence.SHARED_HOST_TITLE if self.matches_title(title) else None
                )
            # Step 3. Without this, a title hint would let any program claim to
            # be this app simply by naming its window.
            return None
        if self.allow_title_only_match and self.matches_title(title):
            return MatchEvidence.TITLE_ONLY
        return None


def _entry(**kwargs) -> ApplicationEntry:
    entry = ApplicationEntry(**kwargs)
    if not APP_ID_PATTERN.fullmatch(entry.app_id):
        raise ValueError(f"invalid app_id: {entry.app_id!r}")
    if not entry.launch_argv:
        raise ValueError(f"{entry.app_id} has no launch argv")
    # A launch argv must never contain shell metacharacters: even though no
    # shell is ever invoked, an argv element carrying them signals a mistake.
    if any(any(ch in part for ch in "&|;<>^`") for part in entry.launch_argv):
        raise ValueError(f"{entry.app_id} launch argv contains shell metacharacters")
    if entry.system_critical and entry.closable:
        raise ValueError(f"{entry.app_id} is system-critical and must not be closable")
    # A shared host declared as dedicated would defeat the whole distinction,
    # so the two sets must be disjoint.
    overlap = set(entry.process_names) & set(entry.shared_host_process_names)
    if overlap:
        raise ValueError(
            f"{entry.app_id} lists {sorted(overlap)} as both dedicated and shared-host"
        )
    # A shared-host entry with no title hints could never match anything, which
    # would be a silent hole rather than a policy.
    if entry.shared_host_process_names and not entry.title_hints:
        raise ValueError(
            f"{entry.app_id} declares a shared host but no title hints to corroborate it"
        )
    if entry.allow_title_only_match and not entry.title_hints:
        raise ValueError(f"{entry.app_id} allows title-only matching but has no title hints")
    return entry


# --------------------------------------------------------------------------- #
# The allowlist. Every entry is hand-reviewed; adding one is a code change.
# --------------------------------------------------------------------------- #
_ENTRIES: Final[tuple[ApplicationEntry, ...]] = (
    _entry(
        app_id="notepad",
        display_name="Notepad",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("notepad.exe",),
        process_names=("notepad.exe",),
        title_hints=("notepad",),
        aliases=("notepad", "note pad", "text editor"),
        closable=True,
        risk_level=RiskLevel.L2_MODIFY_LOCAL,
        notes="Can hold unsaved work; close is graceful only and stops on any prompt.",
    ),
    _entry(
        app_id="calculator",
        display_name="Calculator",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("calc.exe",),
        # calc.exe is a launcher stub; the real packaged window is owned by
        # CalculatorApp.exe.
        process_names=("calculatorapp.exe", "calculator.exe"),
        # Observed on this hardware: the visible top-level window is sometimes
        # owned by ApplicationFrameHost.exe with the title "Calculator", and
        # the app is re-hosted between the two during its lifetime. AFH hosts
        # every packaged app, so it is a SHARED host and the title must
        # corroborate before a window is attributed to Calculator.
        shared_host_process_names=("applicationframehost.exe",),
        title_hints=("calculator",),
        aliases=("calculator", "calc"),
        closable=True,
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        allow_title_only_match=True,
        notes="Stateless; safe to close. Packaged: may be hosted by ApplicationFrameHost.",
    ),
    _entry(
        app_id="file_explorer",
        display_name="File Explorer",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("explorer.exe",),
        process_names=("explorer.exe",),
        aliases=("file explorer", "explorer", "windows explorer", "my computer"),
        # explorer.exe is also the Windows shell. Closing it by process would
        # take down the taskbar and desktop, so it is permanently not closable.
        closable=False,
        system_critical=True,
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        notes="Shell host. Open/focus only; close is structurally refused.",
    ),
    _entry(
        app_id="settings",
        display_name="Windows Settings",
        launch_strategy=LaunchStrategy.SHELL_URI,
        launch_argv=("explorer.exe", "ms-settings:"),
        process_names=("systemsettings.exe",),
        # ApplicationFrameHost was previously listed as a DEDICATED process
        # here, which meant any packaged window it hosted -- Calculator
        # included -- was accepted as Settings on process name alone, and was
        # therefore closable as Settings. It is a shared host; the title must
        # corroborate.
        shared_host_process_names=("applicationframehost.exe",),
        title_hints=("settings",),
        aliases=("settings", "windows settings", "system settings"),
        closable=True,
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        allow_title_only_match=True,
        notes="Packaged app; window may be hosted by ApplicationFrameHost.",
    ),
    _entry(
        app_id="edge",
        display_name="Microsoft Edge",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("msedge.exe",),
        process_names=("msedge.exe",),
        title_hints=("microsoft edge",),
        aliases=("edge", "microsoft edge"),
        closable=True,
        risk_level=RiskLevel.L2_MODIFY_LOCAL,
        notes="Browser CONTROL is deferred to the Part 16 browser bridge.",
    ),
    _entry(
        app_id="chrome",
        display_name="Google Chrome",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("chrome.exe",),
        process_names=("chrome.exe",),
        title_hints=("google chrome",),
        aliases=("chrome", "google chrome"),
        closable=True,
        risk_level=RiskLevel.L2_MODIFY_LOCAL,
        notes="Browser CONTROL is deferred to the Part 16 browser bridge.",
    ),
    _entry(
        app_id="vscode",
        display_name="Visual Studio Code",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("code.cmd",),
        process_names=("code.exe",),
        title_hints=("visual studio code",),
        aliases=("vscode", "vs code", "visual studio code", "code editor"),
        # An editor routinely holds unsaved buffers. Open/focus only.
        closable=False,
        risk_level=RiskLevel.L1_SAFE_CONTROL,
        notes="Not closable in 12.1: unsaved editor buffers are too easy to lose.",
    ),
    _entry(
        app_id="terminal",
        display_name="Windows Terminal",
        launch_strategy=LaunchStrategy.EXECUTABLE,
        launch_argv=("wt.exe",),
        process_names=("windowsterminal.exe",),
        title_hints=("terminal",),
        aliases=("windows terminal", "terminal"),
        # Opening a terminal WINDOW is not the same as executing commands in
        # it. Part 12.1 has no keystroke-into-window capability at all, so this
        # cannot become an arbitrary shell.
        closable=False,
        risk_level=RiskLevel.L2_MODIFY_LOCAL,
        notes=(
            "Window may be opened/focused. Bunnelby cannot type into it: there is "
            "no text-entry action in Part 12.1. Not closable (may hold a running job)."
        ),
    ),
)

_BY_ID: Final[Mapping[str, ApplicationEntry]] = {entry.app_id: entry for entry in _ENTRIES}


def _build_alias_index() -> Mapping[str, str]:
    index: dict[str, str] = {}
    for entry in _ENTRIES:
        for alias in (entry.app_id, *entry.aliases):
            key = _normalize(alias)
            existing = index.get(key)
            if existing and existing != entry.app_id:
                raise ValueError(f"ambiguous alias {alias!r}: {existing} vs {entry.app_id}")
            index[key] = entry.app_id
    return index


def _normalize(value: str) -> str:
    """Fold a spoken/typed name to a comparable key. Exact match only after this."""
    return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())


_ALIAS_INDEX: Final[Mapping[str, str]] = _build_alias_index()


def known_app_ids() -> tuple[str, ...]:
    return tuple(sorted(_BY_ID))


def all_entries() -> tuple[ApplicationEntry, ...]:
    return _ENTRIES


def get(app_id: str) -> ApplicationEntry:
    """Look up by canonical id. Raises UnknownApplicationError, never guesses."""
    entry = _BY_ID.get(_normalize(app_id).replace(" ", "_"))
    if entry is None:
        raise UnknownApplicationError(
            f"{app_id!r} is not a registered application. "
            f"Known: {', '.join(known_app_ids())}"
        )
    return entry


def resolve(spoken_target: str) -> ApplicationEntry:
    """Resolve a user-facing name to a registered entry.

    Exact alias match only. There is deliberately no fuzzy matching, no prefix
    matching and no path handling: "open C:/Windows/System32/cmd.exe" and
    "open some_random_tool" must both fail closed rather than resolve to
    something plausible.
    """
    key = _normalize(spoken_target)
    app_id = _ALIAS_INDEX.get(key)
    if app_id is None:
        raise UnknownApplicationError(
            f"{spoken_target!r} is not a registered application. "
            f"Known: {', '.join(known_app_ids())}"
        )
    return _BY_ID[app_id]


def try_resolve(spoken_target: str) -> ApplicationEntry | None:
    try:
        return resolve(spoken_target)
    except UnknownApplicationError:
        return None
