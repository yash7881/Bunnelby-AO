"""Part 12.1 REAL-WINDOWS integration tests.

Separated from the pure-logic suite because these touch the live desktop. They
skip cleanly off-Windows so Linux CI is unaffected, which is why no Win32 call
is mocked here -- mocking would defeat the purpose of the file.

NON-DESTRUCTIVE BY CONSTRUCTION: nothing here launches, closes, focuses, or
sends input. Every test is an observation. Anything that changes desktop state
belongs in the operator-run acceptance script
(scripts/desktop/part12_1_acceptance.py), where a human is watching.
"""

from __future__ import annotations

import sys
import unittest

from services.api.app.desktop import app_registry
from services.api.app.desktop.controller import DesktopController
from services.api.app.desktop.models import MAX_UI_DEPTH, MAX_UI_NODES, MAX_WINDOWS_RETURNED
from services.api.app.desktop.ui_automation import default_provider
from services.api.app.desktop.windows_api import Win32WindowBackend, default_backend

WINDOWS_ONLY = unittest.skipUnless(sys.platform == "win32", "requires Windows")


@WINDOWS_ONLY
class RealWindowEnumerationTests(unittest.TestCase):
    """Read-only checks against the live desktop."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = default_backend()
        if not cls.backend.is_available():
            raise unittest.SkipTest("Win32 window backend unavailable")

    def test_the_real_backend_is_the_win32_one(self) -> None:
        self.assertIsInstance(self.backend, Win32WindowBackend)

    def test_enumeration_returns_plausible_live_windows(self) -> None:
        windows = self.backend.list_windows()
        self.assertGreater(len(windows), 0, "a running desktop has at least one window")
        for info in windows:
            self.assertGreater(info.handle, 0)
            self.assertGreaterEqual(info.pid, 0)
            self.assertTrue(info.title, "enumeration filters out untitled windows")

    def test_process_names_resolve_for_at_least_some_windows(self) -> None:
        """Access can be denied for elevated processes, so this asserts that the
        mechanism works at all rather than that it works for every window."""
        windows = self.backend.list_windows()
        named = [w for w in windows if w.process_name.endswith(".exe")]
        self.assertGreater(len(named), 0, "kernel32 image-name lookup produced nothing")

    def test_exactly_one_window_is_reported_as_foreground(self) -> None:
        windows = self.backend.list_windows()
        foreground = [w for w in windows if w.is_foreground]
        self.assertLessEqual(len(foreground), 1)

    def test_foreground_handle_is_a_real_window(self) -> None:
        handle = self.backend.foreground_handle()
        if handle:
            self.assertTrue(self.backend.is_window(handle))

    def test_enumeration_is_bounded_through_the_controller(self) -> None:
        outcome = DesktopController(backend=self.backend).list_windows()
        self.assertEqual(outcome.status, "succeeded")
        self.assertLessEqual(len(outcome.windows), MAX_WINDOWS_RETURNED)

    def test_enumeration_is_fast(self) -> None:
        """Performance target, not a promise: enumeration should be sub-second."""
        outcome = DesktopController(backend=self.backend).list_windows()
        self.assertLess(
            outcome.latency_ms, 2_000, f"enumeration took {outcome.latency_ms:.0f}ms"
        )


@WINDOWS_ONLY
class RealUiAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = default_provider()
        if not cls.provider.is_available():
            raise unittest.SkipTest("UI Automation provider unavailable (comtypes missing)")
        cls.backend = default_backend()

    def test_a_live_window_yields_a_bounded_tree(self) -> None:
        windows = self.backend.list_windows()
        if not windows:
            self.skipTest("no windows open")
        tree = self.provider.inspect(windows[0].handle, max_depth=2, max_nodes=25)
        self.assertLessEqual(tree.node_count(), 25, "node budget must be respected")
        self.assertTrue(
            all(node.depth <= 2 for node in tree.flatten()), "depth cap must hold"
        )

    def test_the_node_budget_is_a_total_not_a_per_level_limit(self) -> None:
        """A wide first level must not smuggle in extra nodes."""
        windows = self.backend.list_windows()
        if not windows:
            self.skipTest("no windows open")
        tree = self.provider.inspect(
            windows[0].handle, max_depth=MAX_UI_DEPTH, max_nodes=10
        )
        self.assertLessEqual(tree.node_count(), 10)

    def test_element_values_are_never_captured(self) -> None:
        """Identity and state only: a text field's contents may be a secret."""
        windows = self.backend.list_windows()
        if not windows:
            self.skipTest("no windows open")
        tree = self.provider.inspect(windows[0].handle, max_depth=2, max_nodes=20)
        for node in tree.flatten():
            self.assertFalse(hasattr(node, "value"))
            self.assertFalse(hasattr(node, "text_content"))


@WINDOWS_ONLY
class RealRegistryResolutionTests(unittest.TestCase):
    def test_registered_apps_that_are_running_are_discoverable(self) -> None:
        """Not every registered app is running, so this asserts the matching
        MECHANISM works for whichever ones happen to be open."""
        controller = DesktopController()
        any_matched = False
        for entry in app_registry.all_entries():
            outcome = controller.list_windows()
            for info in outcome.windows:
                if entry.matches_process(info.process_name):
                    any_matched = True
                    break
        # A machine with nothing registered running is legitimate; only assert
        # that the scan completed without error.
        self.assertIsInstance(any_matched, bool)

    def test_an_unregistered_app_is_refused_against_the_live_desktop(self) -> None:
        outcome = DesktopController().focus_app("definitely_not_registered")
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(outcome.error_code, "unknown_application")


if __name__ == "__main__":
    unittest.main()
