"""Part 12.1 bounded Windows desktop control.

Public surface is intentionally small: the capability layer needs the action
enum, the outcome type and a controller. Backends stay internal so no caller
outside this package can reach raw Win32 or COM.
"""

from .controller import DesktopController, default_controller
from .models import (
    DesktopAction,
    DesktopError,
    DesktopOutcome,
    DesktopStatus,
    UiNode,
    UnknownApplicationError,
    WindowInfo,
)

__all__ = [
    "DesktopAction",
    "DesktopController",
    "DesktopError",
    "DesktopOutcome",
    "DesktopStatus",
    "UiNode",
    "UnknownApplicationError",
    "WindowInfo",
    "default_controller",
]
