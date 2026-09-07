"""Windows window layer: enumerate, focus, close. Pure ctypes, zero dependencies.

The executor never calls Win32 directly. It talks to `WindowBackend`, which has
exactly two implementations: the real `Win32WindowBackend` here, and an
in-memory fake in the tests. That is what lets the entire policy layer -- app
resolution, ambiguity rules, close permissions, verification -- be tested on any
platform without mocking `ctypes` itself.

DELIBERATELY ABSENT, and not an oversight:
  * TerminateProcess / taskkill  -- close is graceful WM_CLOSE only.
  * SetWindowsHookEx             -- no global hooks, no keylogging.
  * mouse_event / SendInput(mouse) -- no coordinate clicking in Part 12.1.
  * ReadProcessMemory / injection of any kind.

`SendInput` IS used, but only by shortcuts.py and only for combinations on a
reviewed allowlist that cannot reach the secure desktop.

`FlashWindowEx` IS used, and is not an activation call: it asks the shell to
draw the user's eye to a window that Windows would not let us raise. It moves
no focus, sends no input, and never converts an unverified focus into a
success.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Final, Protocol, Sequence

from .models import (
    POLL_INTERVAL_SECONDS,
    ProviderUnavailableError,
    WindowInfo,
)

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# Win32 constants used below, named so the calls stay readable.
_WM_CLOSE = 0x0010
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SW_RESTORE = 9
_SW_SHOW = 5

# FlashWindowEx (winuser.h). FLASHW_TRAY flashes the taskbar button only --
# deliberately not FLASHW_ALL, which would also flash the caption of a window
# the user did not ask to see.
_FLASHW_STOP = 0x00000000
_FLASHW_TRAY = 0x00000002
_FLASHW_TIMERNOFG = 0x0000000C

#: How many times the taskbar button may blink. Small and finite: this is a
#: nudge, not an alarm, and an unbounded flash would be a nuisance the user
#: cannot dismiss.
MAX_FLASH_COUNT: Final[int] = 3


@dataclass(frozen=True, slots=True)
class FocusAttempt:
    """What the OS was ASKED to do, and what it said -- never the final truth.

    The previous `focus() -> bool` conflated the request with the outcome by
    sampling GetForegroundWindow() synchronously right after
    SetForegroundWindow. Activation is asynchronous, so that sample raced: live
    diagnostics caught it returning False for a focus that had in fact
    succeeded, and it would equally return True for one that was about to be
    overridden.

    So the backend now reports only the ATTEMPT. Whether the target actually
    ended up in front is decided by the controller, which observes real state
    afterwards. A Win32 BOOL may never by itself produce a success verdict.
    """

    handle: int
    was_minimized: bool = False
    restore_requested: bool = False
    set_foreground_returned: bool = False
    #: GetLastError() after the call. SetForegroundWindow does not document a
    #: meaningful error code, so this is diagnostic only.
    last_error: int = 0


class WindowBackend(Protocol):
    """The only surface the desktop controller may use to touch windows."""

    def is_available(self) -> bool: ...

    def list_windows(self) -> tuple[WindowInfo, ...]:
        """Every visible top-level window with a non-empty title."""
        ...

    def foreground_handle(self) -> int: ...

    def focus(self, handle: int) -> FocusAttempt:
        """Ask Windows to restore and foreground a window.

        Reports what was attempted. It does NOT report whether the window ended
        up in front -- only the controller's after-observation decides that.
        """
        ...

    def flash(self, handle: int, count: int = MAX_FLASH_COUNT) -> bool:
        """Request the user's attention for a window. Never moves focus."""
        ...

    def request_close(self, handle: int) -> bool:
        """Post a graceful close request. Never forces, never terminates."""
        ...

    def is_window(self, handle: int) -> bool: ...

    def launch(self, argv: Sequence[str]) -> int:
        """Start a process from a literal argv list. Returns the child pid."""
        ...


class UnavailableWindowBackend:
    """Used off-Windows so imports and tests never explode on platform."""

    def is_available(self) -> bool:
        return False

    def _refuse(self) -> None:
        raise ProviderUnavailableError(
            "Windows window control is unavailable on this platform.",
            code="platform_unsupported",
        )

    def list_windows(self) -> tuple[WindowInfo, ...]:
        self._refuse()
        return ()

    def foreground_handle(self) -> int:
        self._refuse()
        return 0

    def focus(self, handle: int) -> FocusAttempt:
        self._refuse()
        return FocusAttempt(handle)

    def flash(self, handle: int, count: int = MAX_FLASH_COUNT) -> bool:
        self._refuse()
        return False

    def request_close(self, handle: int) -> bool:
        self._refuse()
        return False

    def is_window(self, handle: int) -> bool:
        self._refuse()
        return False

    def launch(self, argv: Sequence[str]) -> int:
        self._refuse()
        return 0


class Win32WindowBackend:
    """Real backend over user32/kernel32 via ctypes."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Declaring signatures matters on 64-bit: an HWND returned through the
        # default c_int restype is truncated, which silently breaks handle
        # comparisons used by the focus verifier.
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self._user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.IsIconic.argtypes = [wintypes.HWND]
        self._user32.SetForegroundWindow.restype = wintypes.BOOL
        self._user32.ShowWindow.restype = wintypes.BOOL
        self._user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE

    def is_available(self) -> bool:
        return True

    # -- enumeration -------------------------------------------------------- #

    def list_windows(self) -> tuple[WindowInfo, ...]:
        ctypes, wintypes = self._ctypes, self._wintypes
        user32 = self._user32
        found: list[WindowInfo] = []
        foreground = self.foreground_handle()

        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _on_window(hwnd, _lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                found.append(
                    WindowInfo(
                        handle=int(hwnd),
                        title=buffer.value,
                        pid=int(pid.value),
                        process_name=self._process_name(int(pid.value)),
                        is_foreground=int(hwnd) == foreground,
                        # A minimized window is still IsWindowVisible, so this
                        # is the only signal that the user cannot see it.
                        is_minimized=bool(user32.IsIconic(hwnd)),
                    )
                )
            except Exception:  # noqa: BLE001 - one bad window must not abort the scan
                logger.debug("Skipped a window during enumeration", exc_info=True)
            return True

        user32.EnumWindows(callback_type(_on_window), 0)
        return tuple(found)

    def _process_name(self, pid: int) -> str:
        """Executable basename for a pid, or '' when access is denied.

        PROCESS_QUERY_LIMITED_INFORMATION is the least privilege that answers
        this, and it works for elevated processes without elevating us.
        """
        if pid <= 0:
            return ""
        ctypes, wintypes = self._ctypes, self._wintypes
        handle = self._kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(260)
            buffer = ctypes.create_unicode_buffer(260)
            if self._kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                return buffer.value.rsplit("\\", 1)[-1]
            return ""
        finally:
            self._kernel32.CloseHandle(handle)

    def foreground_handle(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def is_window(self, handle: int) -> bool:
        return bool(self._user32.IsWindow(self._wintypes.HWND(handle)))

    # -- state changes ------------------------------------------------------ #

    def focus(self, handle: int) -> FocusAttempt:
        """Restore if minimized, then ask once for the foreground.

        ONE activation attempt. Windows enforces foreground rules a background
        process cannot override, and we do not try to: no AttachThreadInput,
        no AllowSetForegroundWindow abuse, no foreground-lock registry change,
        no synthetic input. Live measurement showed repeated attempts never
        recover a refusal (9/9 denials across three trials), so retrying here
        would cost latency and buy nothing.

        Restoring, by contrast, DOES work even when activation is refused --
        which is exactly the partial reality the caller must be able to report.
        """
        ctypes = self._ctypes
        hwnd = self._wintypes.HWND(handle)

        was_minimized = bool(self._user32.IsIconic(hwnd))
        self._user32.ShowWindow(hwnd, _SW_RESTORE if was_minimized else _SW_SHOW)

        ctypes.set_last_error(0)
        raised = bool(self._user32.SetForegroundWindow(hwnd))
        last_error = ctypes.get_last_error()

        return FocusAttempt(
            handle=handle,
            was_minimized=was_minimized,
            restore_requested=was_minimized,
            set_foreground_returned=raised,
            last_error=int(last_error),
        )

    def flash(self, handle: int, count: int = MAX_FLASH_COUNT) -> bool:
        """Blink the taskbar button to request attention. Not activation.

        Used only when Windows has refused to raise a window we were asked to
        show. It changes no focus and synthesises no input; the user stays in
        control of what comes forward. Bounded blink count, and FLASHW_TIMERNOFG
        stops it as soon as the window does come to the foreground.
        """
        ctypes, wintypes = self._ctypes, self._wintypes

        class _FlashInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("hwnd", wintypes.HWND),
                ("dwFlags", wintypes.DWORD),
                ("uCount", wintypes.UINT),
                ("dwTimeout", wintypes.DWORD),
            ]

        bounded = max(1, min(int(count), MAX_FLASH_COUNT))
        info = _FlashInfo(
            cbSize=ctypes.sizeof(_FlashInfo),
            hwnd=wintypes.HWND(handle),
            dwFlags=_FLASHW_TRAY | _FLASHW_TIMERNOFG,
            uCount=bounded,
            dwTimeout=0,
        )
        try:
            self._user32.FlashWindowEx(ctypes.byref(info))
        except Exception:  # noqa: BLE001 - attention is a nicety, never a failure
            logger.debug("FlashWindowEx failed for handle %s", handle, exc_info=True)
            return False
        # FlashWindowEx returns the window's PREVIOUS flash state, not success,
        # so it cannot be used as a result. Reaching here without raising is the
        # only thing we can honestly claim.
        return True

    def request_close(self, handle: int) -> bool:
        """Ask the window to close, exactly as clicking its X would.

        PostMessage(WM_CLOSE) is asynchronous and fully respects the app: if
        there is unsaved work the app shows its own confirmation dialog and
        stays open. Bunnelby never answers that dialog.
        """
        return bool(
            self._user32.PostMessageW(self._wintypes.HWND(handle), _WM_CLOSE, 0, 0)
        )

    def launch(self, argv: Sequence[str]) -> int:
        """Start a registry-supplied argv. No shell, ever."""
        parts = [str(part) for part in argv]
        if not parts:
            raise ValueError("launch requires a non-empty argv")
        creation_flags = 0
        if IS_WINDOWS:
            # Detach from Bunnelby's console so a launched GUI app is not a
            # child that dies with the API process.
            creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        process = subprocess.Popen(  # noqa: S603 - argv is a registry literal
            parts,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            close_fds=True,
        )
        return process.pid


def default_backend() -> WindowBackend:
    """The backend for this machine. Never raises at import time."""
    if not IS_WINDOWS:
        return UnavailableWindowBackend()
    try:
        return Win32WindowBackend()
    except Exception as exc:  # noqa: BLE001 - a broken backend must fail closed
        logger.warning("Win32 window backend unavailable: %s", exc)
        return UnavailableWindowBackend()


def wait_until(
    predicate,
    *,
    timeout: float,
    interval: float = POLL_INTERVAL_SECONDS,
    clock=time.monotonic,
    sleep=time.sleep,
) -> bool:
    """Bounded poll. Returns False on timeout; never loops forever.

    Injectable clock/sleep so timeout behaviour is asserted in tests without
    spending real wall-clock time.
    """
    deadline = clock() + max(0.0, timeout)
    satisfied = bool(predicate())
    while not satisfied and clock() < deadline:
        sleep(interval)
        satisfied = bool(predicate())
    return satisfied
