"""The desktop controller: policy, serialisation, execution and verification.

This is the only module that composes the registry, the window backend, the
shortcut allowlist and the UIA provider. Everything above it (the capability,
the executor) deals in `DesktopOutcome` and never in handles or key codes.

THE VERIFICATION RULE
---------------------
A state-changing action is NOT reported as succeeded because the API call
returned. `subprocess.Popen` returning a pid proves a process was created, not
that Notepad is on screen; `SetForegroundWindow` returning non-zero does not
mean the window is actually in front. Every state-changing action therefore
observes the world afterwards and downgrades to `unverified` when it cannot
prove the intended end state. `unverified` is not `failed`, and neither is
`succeeded`.

SERIALISATION
-------------
One re-entrant lock guards state-changing actions only. Two concurrent focus
calls would race the foreground window and make both verifications
meaningless. Read-only actions (list/inspect/find) do not take the lock, so a
UI listing never blocks behind a slow app launch.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Sequence

from . import app_registry, shortcuts as shortcut_policy
from .app_registry import ApplicationEntry
from .models import (
    CLOSE_VERIFY_TIMEOUT_SECONDS,
    DISCOVERY_GRACE_SECONDS,
    DesktopAction,
    DesktopError,
    DesktopOutcome,
    FOCUS_VERIFY_TIMEOUT_SECONDS,
    LAUNCH_VERIFY_TIMEOUT_SECONDS,
    MAX_UI_DEPTH,
    MAX_UI_NODES,
    MatchEvidence,
    UiNode,
    UnknownApplicationError,
    WindowInfo,
    bounded_timeout,
    limit_windows,
)
from .ui_automation import UiTreeProvider, default_provider, find_controls
from .windows_api import WindowBackend, default_backend, wait_until

logger = logging.getLogger(__name__)

#: Titles that indicate an application is asking the user to confirm something.
#: Seeing one after a close request means Bunnelby must STOP: answering a save
#: prompt is a decision about the user's data that only the user may make.
_CONFIRMATION_TITLE_HINTS: tuple[str, ...] = (
    "save",
    "unsaved",
    "do you want to",
    "confirm",
    "discard",
    "changes you made",
)


class DesktopController:
    """Bounded, verified desktop operations over injected backends."""

    def __init__(
        self,
        *,
        backend: WindowBackend | None = None,
        ui_provider: UiTreeProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        discovery_grace_seconds: float = DISCOVERY_GRACE_SECONDS,
    ) -> None:
        self._backend = backend if backend is not None else default_backend()
        self._ui = ui_provider if ui_provider is not None else default_provider()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._discovery_grace = max(0.0, float(discovery_grace_seconds))

    # -- helpers ------------------------------------------------------------ #

    def _attributed_windows(
        self, entry: ApplicationEntry
    ) -> tuple[tuple[WindowInfo, MatchEvidence], ...]:
        """Every window attributable to this app, paired with HOW it matched.

        The rule itself lives in `ApplicationEntry.match_window` so that the
        shared-host distinction is a reviewable property of the registry rather
        than a conditional buried here. This function only carries the evidence
        forward, because callers act on it differently: a read may use a
        title-only match, a close may not.
        """
        matched: list[tuple[WindowInfo, MatchEvidence]] = []
        for window in self._backend.list_windows():
            evidence = entry.match_window(window.process_name, window.title)
            if evidence is not None:
                matched.append((window, evidence))
        return tuple(matched)

    def _windows_for(
        self, entry: ApplicationEntry, *, strong_only: bool = False
    ) -> tuple[WindowInfo, ...]:
        """Windows belonging to a registered app.

        `strong_only` drops title-only matches, which is what state-changing
        actions on a specific window require: an arbitrary program can title
        its window "Calculator", and that must never be enough to close it.
        """
        return tuple(
            window
            for window, evidence in self._attributed_windows(entry)
            if evidence.is_strong or not strong_only
        )

    def _discover_windows(
        self,
        entry: ApplicationEntry,
        *,
        grace: float | None = None,
        strong_only: bool = False,
    ) -> tuple[WindowInfo, ...]:
        """Find an app's windows, tolerating a transient enumeration miss.

        A packaged app is re-hosted between its own process and
        ApplicationFrameHost during its lifetime, and an enumeration taken
        mid-transition returns nothing at all. Declaring the app absent on that
        single sample produced the false "Calculator is not currently running"
        seen on real hardware, seconds after open_app had verified it.

        So absence is confirmed rather than assumed: poll until the bounded
        grace elapses, then believe it. This is the ordinary `wait_until`
        deadline machinery -- no fixed sleep, no retry counter, no unbounded
        loop -- and with grace 0 it collapses to exactly one enumeration.
        """
        window_grace = self._discovery_grace if grace is None else max(0.0, grace)
        found: list[tuple[WindowInfo, ...]] = [()]

        def _look() -> bool:
            found[0] = self._windows_for(entry, strong_only=strong_only)
            return bool(found[0])

        wait_until(_look, timeout=window_grace, clock=self._clock, sleep=self._sleep)
        return found[0]

    def _elapsed_ms(self, started: float) -> float:
        return (self._clock() - started) * 1000.0

    def _failure(
        self,
        action: DesktopAction,
        target: str,
        error: DesktopError,
        started: float,
    ) -> DesktopOutcome:
        return DesktopOutcome(
            action=action,
            status="blocked" if isinstance(error, UnknownApplicationError) else "failed",
            target=target,
            detail=str(error),
            error_code=error.code,
            latency_ms=self._elapsed_ms(started),
        )

    # -- read-only actions (no lock) ---------------------------------------- #

    def list_windows(self) -> DesktopOutcome:
        started = self._clock()
        try:
            windows = limit_windows(self._backend.list_windows())
        except DesktopError as exc:
            return self._failure(DesktopAction.LIST_WINDOWS, "", exc, started)
        return DesktopOutcome(
            action=DesktopAction.LIST_WINDOWS,
            status="succeeded",
            windows=windows,
            detail=f"{len(windows)} visible window(s).",
            latency_ms=self._elapsed_ms(started),
            evidence={"window_count": len(windows)},
        )

    def inspect_window(
        self,
        target: str,
        *,
        max_depth: int = MAX_UI_DEPTH,
        max_nodes: int = MAX_UI_NODES,
    ) -> DesktopOutcome:
        started = self._clock()
        try:
            entry = app_registry.resolve(target)
        except UnknownApplicationError as exc:
            return self._failure(DesktopAction.INSPECT_WINDOW, target, exc, started)

        # Rediscovery applies to reads too: inspect and find_control reported
        # the same false "not currently running" as focus on real hardware.
        windows = self._discover_windows(entry)
        if not windows:
            return DesktopOutcome(
                action=DesktopAction.INSPECT_WINDOW,
                status="failed",
                target=entry.app_id,
                detail=f"{entry.display_name} is not currently running.",
                error_code="application_not_running",
                latency_ms=self._elapsed_ms(started),
            )
        # Clamp here as well as in the provider: a caller reaching the
        # controller directly (tests, a future internal caller) must not be able
        # to request an unbounded tree just because it bypassed the request model.
        depth = max(1, min(int(max_depth), MAX_UI_DEPTH))
        nodes = max(1, min(int(max_nodes), MAX_UI_NODES))
        try:
            tree = self._ui.inspect(windows[0].handle, max_depth=depth, max_nodes=nodes)
        except DesktopError as exc:
            return self._failure(DesktopAction.INSPECT_WINDOW, entry.app_id, exc, started)

        return DesktopOutcome(
            action=DesktopAction.INSPECT_WINDOW,
            status="succeeded",
            target=entry.app_id,
            windows=windows[:1],
            nodes=(tree,),
            detail=f"Inspected {entry.display_name}.",
            latency_ms=self._elapsed_ms(started),
            evidence={"node_count": tree.node_count(), "max_depth": depth},
        )

    def find_control(
        self,
        target: str,
        *,
        name_contains: str = "",
        control_type: str = "",
        limit: int = 20,
    ) -> DesktopOutcome:
        started = self._clock()
        inspected = self.inspect_window(target)
        if inspected.status != "succeeded" or not inspected.nodes:
            return DesktopOutcome(
                action=DesktopAction.FIND_CONTROL,
                status=inspected.status,
                target=inspected.target,
                detail=inspected.detail,
                error_code=inspected.error_code,
                latency_ms=self._elapsed_ms(started),
            )
        matches = find_controls(
            inspected.nodes[0],
            name_contains=name_contains,
            control_type=control_type,
            limit=limit,
        )
        return DesktopOutcome(
            action=DesktopAction.FIND_CONTROL,
            status="succeeded",
            target=inspected.target,
            windows=inspected.windows,
            nodes=matches,
            detail=f"{len(matches)} matching control(s).",
            latency_ms=self._elapsed_ms(started),
            evidence={"match_count": len(matches)},
        )

    # -- state-changing actions (serialised) -------------------------------- #

    def open_app(self, target: str, *, timeout: float | None = None) -> DesktopOutcome:
        started = self._clock()
        try:
            entry = app_registry.resolve(target)
        except UnknownApplicationError as exc:
            return self._failure(DesktopAction.OPEN_APP, target, exc, started)

        wait_for = min(bounded_timeout(timeout), LAUNCH_VERIFY_TIMEOUT_SECONDS)
        with self._lock:
            existing = self._windows_for(entry)
            if existing:
                # Already running: the honest action is to surface it, not to
                # spawn a second instance the user did not ask for.
                focused = self.focus_app(entry.app_id, timeout=timeout, _locked=True)
                return DesktopOutcome(
                    action=DesktopAction.OPEN_APP,
                    status=focused.status,
                    target=entry.app_id,
                    windows=focused.windows or existing,
                    detail=(
                        f"{entry.display_name} was already running; "
                        f"{focused.detail[:1].lower()}{focused.detail[1:]}"
                        if focused.detail
                        else f"{entry.display_name} was already running."
                    ),
                    error_code=focused.error_code,
                    latency_ms=self._elapsed_ms(started),
                    evidence={"already_running": True, **dict(focused.evidence)},
                )

            try:
                pid = self._backend.launch(entry.launch_argv)
            except DesktopError as exc:
                return self._failure(DesktopAction.OPEN_APP, entry.app_id, exc, started)
            except (OSError, ValueError) as exc:
                return DesktopOutcome(
                    action=DesktopAction.OPEN_APP,
                    status="failed",
                    target=entry.app_id,
                    detail=f"Could not start {entry.display_name}: {exc}",
                    error_code="launch_failed",
                    latency_ms=self._elapsed_ms(started),
                )

            # VERIFY: a pid is not a window. Poll for a real, matching window.
            appeared = wait_until(
                lambda: bool(self._windows_for(entry)),
                timeout=wait_for,
                clock=self._clock,
                sleep=self._sleep,
            )
            windows = self._windows_for(entry)

        if not appeared or not windows:
            return DesktopOutcome(
                action=DesktopAction.OPEN_APP,
                status="unverified",
                target=entry.app_id,
                detail=(
                    f"Started {entry.display_name} (pid {pid}) but no matching window "
                    f"appeared within {wait_for:.0f}s."
                ),
                error_code="timeout",
                latency_ms=self._elapsed_ms(started),
                evidence={"launched_pid": pid, "verified": False},
            )
        return DesktopOutcome(
            action=DesktopAction.OPEN_APP,
            status="succeeded",
            target=entry.app_id,
            windows=windows,
            detail=f"{entry.display_name} is open.",
            latency_ms=self._elapsed_ms(started),
            evidence={"launched_pid": pid, "verified": True, "window_count": len(windows)},
        )

    def focus_app(
        self, target: str, *, timeout: float | None = None, _locked: bool = False
    ) -> DesktopOutcome:
        started = self._clock()
        try:
            entry = app_registry.resolve(target)
        except UnknownApplicationError as exc:
            return self._failure(DesktopAction.FOCUS_APP, target, exc, started)

        wait_for = min(bounded_timeout(timeout), FOCUS_VERIFY_TIMEOUT_SECONDS)
        guard = _NullContext() if _locked else self._lock
        with guard:
            # Confirm absence before reporting it. open_app verifying a window
            # and focus_app declaring the same app missing moments later was
            # the exact false negative observed on hardware.
            windows = self._discover_windows(entry)
            if not windows:
                return DesktopOutcome(
                    action=DesktopAction.FOCUS_APP,
                    status="failed",
                    target=entry.app_id,
                    detail=f"{entry.display_name} is not currently running.",
                    error_code="application_not_running",
                    latency_ms=self._elapsed_ms(started),
                )

            chosen = _preferred_window(windows)
            self._backend.focus(chosen.handle)
            # VERIFY: ask the OS what is actually in front, not what we asked for.
            became = wait_until(
                lambda: self._backend.foreground_handle() == chosen.handle,
                timeout=wait_for,
                clock=self._clock,
                sleep=self._sleep,
            )
            observed = self._backend.foreground_handle()

        if not became:
            # Windows restricts which processes may steal focus. Reporting this
            # honestly is required; pretending it worked is not an option.
            return DesktopOutcome(
                action=DesktopAction.FOCUS_APP,
                status="unverified",
                target=entry.app_id,
                windows=(chosen,),
                detail=(
                    f"Asked Windows to bring {entry.display_name} to the front, but it is "
                    "not the foreground window. Windows restricts focus changes from "
                    "background processes."
                ),
                error_code="foreground_denied",
                latency_ms=self._elapsed_ms(started),
                evidence={
                    "expected_handle": chosen.handle,
                    "observed_handle": observed,
                    "verified": False,
                },
            )
        return DesktopOutcome(
            action=DesktopAction.FOCUS_APP,
            status="succeeded",
            target=entry.app_id,
            windows=(chosen,),
            detail=f"{entry.display_name} is now in front.",
            latency_ms=self._elapsed_ms(started),
            evidence={"expected_handle": chosen.handle, "verified": True},
        )

    def close_app(self, target: str, *, timeout: float | None = None) -> DesktopOutcome:
        started = self._clock()
        try:
            entry = app_registry.resolve(target)
        except UnknownApplicationError as exc:
            return self._failure(DesktopAction.CLOSE_APP, target, exc, started)

        # POLICY GATE, before anything is attempted.
        if entry.system_critical or not entry.closable:
            reason = (
                "it is part of the Windows shell and closing it would take down the "
                "desktop"
                if entry.system_critical
                else "it may be holding unsaved work or a running job"
            )
            return DesktopOutcome(
                action=DesktopAction.CLOSE_APP,
                status="blocked",
                target=entry.app_id,
                detail=f"Bunnelby will not close {entry.display_name}: {reason}.",
                error_code="close_not_permitted",
                latency_ms=self._elapsed_ms(started),
                evidence={"closable": False, "system_critical": entry.system_critical},
            )

        wait_for = min(bounded_timeout(timeout), CLOSE_VERIFY_TIMEOUT_SECONDS)
        with self._lock:
            # A close may only ever target windows whose identity was
            # corroborated by their owning process. Title-only evidence is
            # enough to describe a window and never enough to close it.
            windows = self._discover_windows(entry, strong_only=True)
            if not windows:
                # Distinguish "gone" from "present but unidentifiable". Saying
                # "already closed" about a running app is a false success, and
                # closing a window we cannot identify is the unsafe close this
                # rule exists to prevent -- so neither is allowed to happen.
                unconfirmed = self._windows_for(entry)
                if unconfirmed:
                    return DesktopOutcome(
                        action=DesktopAction.CLOSE_APP,
                        status="blocked",
                        target=entry.app_id,
                        windows=unconfirmed,
                        detail=(
                            f"Bunnelby will not close {entry.display_name}: a window "
                            "looks like it, but its owning process could not be "
                            "confirmed, and a window title alone is not proof of "
                            "identity."
                        ),
                        error_code="close_not_permitted",
                        latency_ms=self._elapsed_ms(started),
                        evidence={
                            "identity_confirmed": False,
                            "title_only_candidates": len(unconfirmed),
                        },
                    )
                return DesktopOutcome(
                    action=DesktopAction.CLOSE_APP,
                    status="succeeded",
                    target=entry.app_id,
                    detail=f"{entry.display_name} was already closed.",
                    error_code="application_already_closed",
                    latency_ms=self._elapsed_ms(started),
                    evidence={"already_closed": True},
                )

            handles = tuple(window.handle for window in windows)
            for handle in handles:
                self._backend.request_close(handle)

            gone = wait_until(
                lambda: not self._windows_for(entry, strong_only=True),
                timeout=wait_for,
                clock=self._clock,
                sleep=self._sleep,
            )
            remaining = self._windows_for(entry, strong_only=True)
            confirmation = _confirmation_window(self._backend.list_windows(), entry)

        if confirmation is not None:
            # STOP. The app is asking the user something. Bunnelby does not get
            # to answer it -- clicking "Don't Save" would destroy work.
            return DesktopOutcome(
                action=DesktopAction.CLOSE_APP,
                status="needs_clarification",
                target=entry.app_id,
                windows=(confirmation,),
                detail=(
                    f"{entry.display_name} asked for confirmation before closing. "
                    "No destructive choice was made; please answer it yourself."
                ),
                error_code="confirmation_required",
                latency_ms=self._elapsed_ms(started),
                evidence={"confirmation_seen": True, "closed": False},
            )
        if not gone:
            return DesktopOutcome(
                action=DesktopAction.CLOSE_APP,
                status="unverified",
                target=entry.app_id,
                windows=remaining,
                detail=(
                    f"Asked {entry.display_name} to close, but {len(remaining)} window(s) "
                    f"were still open after {wait_for:.0f}s. Nothing was forced."
                ),
                error_code="timeout",
                latency_ms=self._elapsed_ms(started),
                evidence={"closed": False, "remaining": len(remaining), "forced": False},
            )
        return DesktopOutcome(
            action=DesktopAction.CLOSE_APP,
            status="succeeded",
            target=entry.app_id,
            detail=f"{entry.display_name} closed.",
            latency_ms=self._elapsed_ms(started),
            evidence={"closed": True, "closed_windows": len(handles), "forced": False},
        )

    def safe_shortcut(self, shortcut_id: str) -> DesktopOutcome:
        started = self._clock()
        try:
            shortcut = shortcut_policy.resolve(shortcut_id)
        except DesktopError as exc:
            return DesktopOutcome(
                action=DesktopAction.SAFE_SHORTCUT,
                status="blocked",
                target=str(shortcut_id),
                detail=str(exc),
                error_code=exc.code,
                latency_ms=self._elapsed_ms(started),
            )

        with self._lock:
            sent = shortcut_policy.send(shortcut)

        if not sent:
            return DesktopOutcome(
                action=DesktopAction.SAFE_SHORTCUT,
                status="failed",
                target=shortcut.shortcut_id,
                detail=f"Windows did not accept the {shortcut.display} input.",
                error_code="internal_error",
                latency_ms=self._elapsed_ms(started),
            )
        # A shortcut's effect is intentionally not asserted: "Win+D minimised
        # everything" has no single observable end state we can prove without
        # guessing. Delivery is what is verified, and the status says so.
        return DesktopOutcome(
            action=DesktopAction.SAFE_SHORTCUT,
            status="succeeded",
            target=shortcut.shortcut_id,
            detail=f"Sent {shortcut.display}.",
            latency_ms=self._elapsed_ms(started),
            evidence={"shortcut": shortcut.display, "delivery_verified": True},
        )


class _NullContext:
    """Re-entrancy helper for the open->focus path, which already holds the lock."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc) -> bool:
        return False


def _preferred_window(windows: Sequence[WindowInfo]) -> WindowInfo:
    """Deterministic choice when an app has several windows.

    Rule: prefer the one already in the foreground, otherwise the lowest handle.
    Lowest-handle is arbitrary but STABLE, which is what matters -- the same
    request must pick the same window every time so verification is meaningful.
    """
    for window in windows:
        if window.is_foreground:
            return window
    return sorted(windows, key=lambda item: item.handle)[0]


def _confirmation_window(
    windows: Sequence[WindowInfo], entry: ApplicationEntry
) -> WindowInfo | None:
    """Detect an app-owned confirmation dialog raised by our close request.

    Dedicated process only, deliberately: a save prompt is owned by the app's
    real process, and widening this to shared hosts or titles would let an
    unrelated window titled "Save..." stall an unrelated close.
    """
    for window in windows:
        if not entry.matches_process(window.process_name):
            continue
        lowered = window.title.casefold()
        if any(hint in lowered for hint in _CONFIRMATION_TITLE_HINTS):
            return window
    return None


_DEFAULT: DesktopController | None = None
_DEFAULT_LOCK = threading.Lock()


def default_controller() -> DesktopController:
    """Process-wide controller, so the serialisation lock is actually shared."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = DesktopController()
    return _DEFAULT
