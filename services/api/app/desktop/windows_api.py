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
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Protocol, Sequence

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


class WindowBackend(Protocol):
    """The only surface the desktop controller may use to touch windows."""

    def is_available(self) -> bool: ...

    def list_windows(self) -> tuple[WindowInfo, ...]:
        """Every visible top-level window with a non-empty title."""
        ...

    def foreground_handle(self) -> int: ...

    def focus(self, handle: int) -> bool:
        """Attempt to foreground a window. False means Windows refused."""
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

    def focus(self, handle: int) -> bool:
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

    def focus(self, handle: int) -> bool:
        """Restore-then-foreground, then report what actually happened.

        Windows enforces foreground-activation rules that a background process
        cannot override, and we do not try to: no AttachThreadInput trickery,
        no AllowSetForegroundWindow abuse. If the OS declines, the caller
        reports a truthful partial state instead of claiming success.
        """
        hwnd = self._wintypes.HWND(handle)
        if self._user32.IsIconic(hwnd):
            self._user32.ShowWindow(hwnd, _SW_RESTORE)
        else:
            self._user32.ShowWindow(hwnd, _SW_SHOW)
        self._user32.SetForegroundWindow(hwnd)
        return self.foreground_handle() == handle

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
