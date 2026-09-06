"""Bounded Windows UI Automation inspection.

READ ONLY. Part 12.1 can describe what is on a window; it cannot click, type,
invoke a control, or read a control's VALUE. That last exclusion is deliberate
and load-bearing: a text field's contents may be a password, a one-time code or
a private message, so this layer records an element's IDENTITY and STATE and
nothing else.

Three properties make the read safe rather than merely useful:

  BOUNDED   Depth, node count and text length are capped by the provider, not
            by the caller, so no request can dump an entire accessibility tree
            into the model's context.
  UNTRUSTED Every name that comes back originates from another program and is
            treated as hostile input by the caller.
  OPTIONAL  The COM backend is imported lazily. Without `comtypes` the provider
            reports unavailable and every other desktop action keeps working.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Final, Mapping, Protocol

from .models import (
    MAX_UI_DEPTH,
    MAX_UI_NODES,
    ProviderUnavailableError,
    UiNode,
)

logger = logging.getLogger(__name__)

# UIA control-type ids (UIAutomationClient.h) mapped to stable names. Only the
# semantically useful ones are named; anything else surfaces as its raw id so
# an unmapped control is still visible rather than silently dropped.
_CONTROL_TYPES: Final[Mapping[int, str]] = {
    50000: "Button",
    50001: "Calendar",
    50002: "CheckBox",
    50003: "ComboBox",
    50004: "Edit",
    50005: "Hyperlink",
    50006: "Image",
    50007: "ListItem",
    50008: "List",
    50009: "Menu",
    50010: "MenuBar",
    50011: "MenuItem",
    50012: "ProgressBar",
    50013: "RadioButton",
    50014: "ScrollBar",
    50015: "Slider",
    50016: "Spinner",
    50017: "StatusBar",
    50018: "Tab",
    50019: "TabItem",
    50020: "Text",
    50021: "ToolBar",
    50022: "ToolTip",
    50023: "Tree",
    50024: "TreeItem",
    50025: "Custom",
    50026: "Group",
    50027: "Thumb",
    50028: "DataGrid",
    50029: "DataItem",
    50030: "Document",
    50031: "SplitButton",
    50032: "Window",
    50033: "Pane",
    50034: "Header",
    50035: "HeaderItem",
    50036: "Table",
    50037: "TitleBar",
    50038: "Separator",
}

#: Control types that count as an interactive, semantically meaningful control.
#: FIND_CONTROL reports these; decorative panes and separators are ignored.
INTERACTIVE_CONTROL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Button",
        "CheckBox",
        "ComboBox",
        "Edit",
        "Hyperlink",
        "ListItem",
        "MenuItem",
        "RadioButton",
        "Slider",
        "SplitButton",
        "TabItem",
        "Tree",
        "TreeItem",
    }
)


def control_type_name(raw: Any) -> str:
    try:
        return _CONTROL_TYPES.get(int(raw), f"Unknown({int(raw)})")
    except (TypeError, ValueError):
        return "Unknown"


class UiTreeProvider(Protocol):
    """The only surface through which UI structure may be read."""

    def is_available(self) -> bool: ...

    def inspect(
        self, window_handle: int, *, max_depth: int = MAX_UI_DEPTH, max_nodes: int = MAX_UI_NODES
    ) -> UiNode:
        """Return a bounded subtree rooted at the given window."""
        ...


class UnavailableUiTreeProvider:
    """Used when comtypes is absent or the platform is not Windows."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def inspect(self, window_handle: int, **_kwargs) -> UiNode:
        raise ProviderUnavailableError(
            f"UI Automation is unavailable: {self._reason}", code="provider_unavailable"
        )


class ComtypesUiTreeProvider:
    """Real UIA provider over the COM IUIAutomation interface.

    The COM objects are created lazily on first use rather than in __init__, so
    importing this module never spins up COM in a worker that will not use it.
    """

    def __init__(self) -> None:
        self._automation: Any | None = None
        self._walker: Any | None = None

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            self._ensure()
        except Exception:  # noqa: BLE001 - availability probing must not raise
            return False
        return self._automation is not None

    def _ensure(self) -> None:
        if self._automation is not None:
            return
        import comtypes.client as client

        module = client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as uia_client

        self._automation = client.CreateObject(
            module.CUIAutomation, interface=uia_client.IUIAutomation
        )
        # RawViewWalker shows the full tree. ControlViewWalker would hide
        # structurally-relevant containers we need for accurate depth limits.
        self._walker = self._automation.RawViewWalker

    def inspect(
        self,
        window_handle: int,
        *,
        max_depth: int = MAX_UI_DEPTH,
        max_nodes: int = MAX_UI_NODES,
    ) -> UiNode:
        self._ensure()
        assert self._automation is not None and self._walker is not None

        try:
            root = self._automation.ElementFromHandle(window_handle)
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailableError(
                f"No UI Automation element for window {window_handle}: {exc}",
                code="window_not_found",
            ) from exc
        if root is None:
            raise ProviderUnavailableError(
                f"No UI Automation element for window {window_handle}",
                code="window_not_found",
            )

        depth_cap = max(1, min(int(max_depth), MAX_UI_DEPTH))
        node_cap = max(1, min(int(max_nodes), MAX_UI_NODES))
        # A mutable budget shared by the whole walk: the cap is on the TOTAL
        # tree, not per level, so a wide first level cannot smuggle in
        # thousands of nodes while respecting a per-level limit.
        budget = [node_cap]
        return self._walk(root, depth=0, depth_cap=depth_cap, budget=budget)

    def _walk(self, element: Any, *, depth: int, depth_cap: int, budget: list[int]) -> UiNode:
        budget[0] -= 1
        node = self._read(element, depth)
        if depth >= depth_cap or budget[0] <= 0:
            return node

        children: list[UiNode] = []
        try:
            child = self._walker.GetFirstChildElement(element)
        except Exception:  # noqa: BLE001 - a hostile/racing UI tree must not crash us
            logger.debug("UIA child walk failed at depth %s", depth, exc_info=True)
            child = None

        while child is not None and budget[0] > 0:
            children.append(
                self._walk(child, depth=depth + 1, depth_cap=depth_cap, budget=budget)
            )
            try:
                child = self._walker.GetNextSiblingElement(child)
            except Exception:  # noqa: BLE001
                logger.debug("UIA sibling walk failed at depth %s", depth, exc_info=True)
                break

        return UiNode(
            name=node.name,
            control_type=node.control_type,
            automation_id=node.automation_id,
            enabled=node.enabled,
            offscreen=node.offscreen,
            depth=depth,
            children=tuple(children),
        )

    @staticmethod
    def _read(element: Any, depth: int) -> UiNode:
        """Read identity and state only. Never CurrentValue / ValuePattern."""

        def safe(getter, default):
            try:
                value = getter()
            except Exception:  # noqa: BLE001 - properties race with a live UI
                return default
            return default if value is None else value

        return UiNode(
            name=str(safe(lambda: element.CurrentName, "")),
            control_type=control_type_name(safe(lambda: element.CurrentControlType, 0)),
            automation_id=str(safe(lambda: element.CurrentAutomationId, "")),
            enabled=bool(safe(lambda: element.CurrentIsEnabled, True)),
            offscreen=bool(safe(lambda: element.CurrentIsOffscreen, False)),
            depth=depth,
        )


def default_provider() -> UiTreeProvider:
    """The provider for this machine. Never raises at import time."""
    if sys.platform != "win32":
        return UnavailableUiTreeProvider("not running on Windows")
    try:
        import comtypes  # noqa: F401
    except ImportError:
        return UnavailableUiTreeProvider(
            "the optional 'comtypes' package is not installed"
        )
    return ComtypesUiTreeProvider()


def find_controls(
    tree: UiNode,
    *,
    name_contains: str = "",
    control_type: str = "",
    limit: int = 20,
) -> tuple[UiNode, ...]:
    """Filter an already-bounded tree for semantically interesting controls.

    Operates on a tree that was captured under the provider's caps, so this can
    never be the thing that makes a read unbounded.
    """
    needle = " ".join(str(name_contains or "").split()).casefold()
    wanted = str(control_type or "").strip()
    matches: list[UiNode] = []
    for node in tree.flatten():
        if wanted:
            if node.control_type != wanted:
                continue
        elif node.control_type not in INTERACTIVE_CONTROL_TYPES:
            continue
        if needle and needle not in node.name.casefold():
            continue
        matches.append(node)
        if len(matches) >= max(1, limit):
            break
    return tuple(matches)
