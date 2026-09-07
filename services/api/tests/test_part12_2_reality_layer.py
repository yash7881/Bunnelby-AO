"""Part 12.2: the desktop reality layer.

WHAT CHANGED AND WHY. Part 12.1 decided focus success by comparing
`GetForegroundWindow()` to the exact handle it had chosen, and it threw away
everything else it knew. Live measurement on real Windows showed three problems
with that:

  * The Win32 return value is not the truth. `SetForegroundWindow` was observed
    returning False for an activation that had in fact succeeded, because
    activation is asynchronous and the backend sampled foreground immediately.
  * Retrying is futile. Nine consecutive refusals across three trials, and a
    second full focus call never recovered. Windows denies background processes
    foreground activation BY DESIGN, and it is right to.
  * Real changes went unreported. Restoring a minimized window succeeds even
    when activation is refused, and the old outcome could not say so.

So the layer reasons from BEFORE -> ACTION -> AFTER -> DIFFERENCE -> VERDICT.
It never pushes harder; it just knows more. No test here touches a real window.
"""

from __future__ import annotations

import unittest

from services.api.app.desktop.controller import DesktopController, _preferred_window
from services.api.app.desktop.models import DesktopAction, WindowInfo
from services.api.app.desktop.presentation import screen_reply, spoken_reply

from test_part12_1_desktop_control import (  # noqa: E402  (shared fixtures)
    FakeUiProvider,
    FakeWindowBackend,
    controller,
    window,
)

NOTEPAD = window(101, "Untitled - Notepad", 900, "notepad.exe")
CALC = window(202, "Calculator", 901, "CalculatorApp.exe")
#: Calculator as this hardware actually hosts it: the shared frame host, whose
#: identity is only strong because the title corroborates it.
CALC_HOSTED = window(303, "Calculator", 950, "applicationframehost.exe")


# --------------------------------------------------------------------------- #
# A-C: the happy paths, now with observable evidence
# --------------------------------------------------------------------------- #


class AlreadyForegroundTests(unittest.TestCase):
    def test_a_an_already_foreground_target_is_a_verified_no_op(self) -> None:
        backend = FakeWindowBackend([CALC, NOTEPAD])  # CALC is foreground
        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.evidence["already_foreground"])
        self.assertTrue(outcome.evidence["verified"])
        self.assertIn("already in front", outcome.detail)

    def test_a_no_activation_call_is_made(self) -> None:
        """Asking Windows to raise what is already raised is a pointless state
        change, and a refusal of it would look like a failure of something that
        was already true."""
        backend = FakeWindowBackend([CALC, NOTEPAD])
        controller(backend).focus_app("calculator")
        self.assertEqual(backend.focused, [], "SetForegroundWindow must not run")

    def test_p_an_already_foreground_target_is_never_flashed(self) -> None:
        backend = FakeWindowBackend([CALC, NOTEPAD])
        controller(backend).focus_app("calculator")
        self.assertEqual(backend.flashed, [])


class NormalFocusTests(unittest.TestCase):
    def test_b_a_successful_focus_is_verified_from_after_state(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC])
        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.evidence["foreground_after"])
        self.assertFalse(outcome.evidence["already_foreground"])
        self.assertEqual(backend.focused, [CALC.handle])
        self.assertEqual(backend.flashed, [], "no attention needed on success")


class MinimizedTargetTests(unittest.TestCase):
    def test_c_a_minimized_target_is_restored_and_verified(self) -> None:
        minimized = window(202, "Calculator", 901, "CalculatorApp.exe", minimized=True)
        backend = FakeWindowBackend([NOTEPAD, minimized])
        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.evidence["was_minimized"])
        self.assertTrue(outcome.evidence["restored"])
        self.assertTrue(outcome.evidence["foreground_after"])

    def test_d_restored_but_denied_keeps_the_restoration_evidence(self) -> None:
        """The half that worked must not be erased by the half that did not."""
        minimized = window(202, "Calculator", 901, "CalculatorApp.exe", minimized=True)
        backend = FakeWindowBackend([NOTEPAD, minimized], focus_succeeds=False)
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(outcome.status, "unverified")
        self.assertTrue(outcome.evidence["was_minimized"])
        self.assertTrue(outcome.evidence["restored"])
        self.assertFalse(outcome.evidence["foreground_after"])
        self.assertIn("was restored", outcome.detail)
        self.assertNotIn("is now in front", outcome.detail)


# --------------------------------------------------------------------------- #
# E-G: the Win32 BOOL is not the truth
# --------------------------------------------------------------------------- #


class ApiReturnIsNotTruthTests(unittest.TestCase):
    def test_e_a_denied_activation_is_unverified(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(outcome.status, "unverified")
        self.assertEqual(outcome.error_code, "foreground_denied")
        self.assertFalse(outcome.evidence["foreground_after"])

    def test_f_a_false_return_with_a_real_focus_still_succeeds(self) -> None:
        """Measured on hardware: SetForegroundWindow returned False for a focus
        that worked. Observation must win over the API's claim."""
        backend = FakeWindowBackend(
            [NOTEPAD, CALC], focus_succeeds=True, set_foreground_returns=False
        )
        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(outcome.status, "succeeded")
        self.assertFalse(outcome.evidence["set_foreground_returned"])
        self.assertTrue(outcome.evidence["foreground_after"])

    def test_g_a_true_return_with_no_real_focus_is_unverified(self) -> None:
        """The mirror image: the API says yes, the desktop says otherwise."""
        backend = FakeWindowBackend(
            [NOTEPAD, CALC], focus_succeeds=False, set_foreground_returns=True
        )
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(outcome.status, "unverified")
        self.assertTrue(outcome.evidence["set_foreground_returned"])
        self.assertFalse(outcome.evidence["foreground_after"])


# --------------------------------------------------------------------------- #
# H-J: stale handles, and the absence of a general retry
# --------------------------------------------------------------------------- #


class StaleHandleTests(unittest.TestCase):
    def test_h_a_vanished_handle_earns_exactly_one_retry(self) -> None:
        replacement = window(404, "Calculator", 901, "CalculatorApp.exe")
        backend = FakeWindowBackend([NOTEPAD, CALC])
        backend.invalidate_on_focus = {CALC.handle}
        backend.replacement_windows = [replacement]

        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(backend.focused, [CALC.handle, replacement.handle])
        self.assertTrue(outcome.evidence["stale_handle_detected"])
        self.assertEqual(outcome.evidence["replacement_handle"], replacement.handle)
        self.assertEqual(outcome.status, "succeeded")

    def test_i_a_valid_handle_that_was_merely_denied_gets_no_retry(self) -> None:
        """Nine of nine refusals on hardware. Retrying costs latency and buys
        nothing, so the handle must actually be GONE to earn a second attempt."""
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(backend.focused, [CALC.handle], "exactly one attempt")
        self.assertNotIn("stale_handle_detected", outcome.evidence)
        self.assertEqual(outcome.status, "unverified")

    def test_j_a_title_only_replacement_is_never_retried(self) -> None:
        """An impostor window must not inherit a dead target's identity."""
        impostor = window(505, "Calculator", 902, "")  # process unreadable
        backend = FakeWindowBackend([NOTEPAD, CALC])
        backend.invalidate_on_focus = {CALC.handle}
        backend.replacement_windows = [impostor]

        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(backend.focused, [CALC.handle], "weak evidence earns nothing")
        self.assertEqual(outcome.status, "unverified")

    def test_q_a_dead_handle_is_never_flashed(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        backend.invalidate_on_focus = {CALC.handle}
        controller(backend).focus_app("calculator", timeout=1.0)
        self.assertNotIn(CALC.handle, backend.flashed)


# --------------------------------------------------------------------------- #
# K-M: equivalent-handle verification, and identity
# --------------------------------------------------------------------------- #


class EquivalentHandleTests(unittest.TestCase):
    def test_k_a_different_but_strongly_attributed_window_verifies(self) -> None:
        """The user asked for an APP. If the app is in front under another of
        its own windows, their intent is satisfied."""
        sibling = window(606, "Calculator", 901, "CalculatorApp.exe")
        backend = FakeWindowBackend([NOTEPAD, CALC, sibling])
        backend.foreground_override = {CALC.handle: sibling.handle}

        outcome = controller(backend).focus_app("calculator")

        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.evidence["foreground_equivalent_handle"])
        self.assertEqual(outcome.evidence["expected_handle"], CALC.handle)
        self.assertEqual(outcome.evidence["observed_handle"], sibling.handle)
        self.assertEqual(
            outcome.evidence["foreground_owner_evidence"], "dedicated_process"
        )

    def test_k_shared_host_identity_also_counts_as_strong(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC_HOSTED])
        outcome = controller(backend).focus_app("calculator")
        self.assertEqual(outcome.status, "succeeded")

    def test_l_a_title_only_foreground_window_never_verifies(self) -> None:
        """An arbitrary program naming its window "Calculator" is not
        Calculator, however convincing the foreground looks."""
        impostor = window(707, "Calculator", 903, "")  # unreadable process
        backend = FakeWindowBackend([NOTEPAD, CALC, impostor])
        backend.foreground_override = {CALC.handle: impostor.handle}

        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(outcome.status, "unverified")
        self.assertNotIn("foreground_equivalent_handle", outcome.evidence)

    def test_l_a_foreign_process_window_never_verifies(self) -> None:
        foreign = window(808, "Calculator", 904, "notepad.exe")
        backend = FakeWindowBackend([NOTEPAD, CALC, foreign])
        backend.foreground_override = {CALC.handle: foreign.handle}

        outcome = controller(backend).focus_app("calculator", timeout=1.0)
        self.assertEqual(outcome.status, "unverified")


class DeterministicSelectionTests(unittest.TestCase):
    def test_m_selection_prefers_foreground_then_visible_then_lowest(self) -> None:
        foreground = WindowInfo(9, "a", 1, "x.exe", is_foreground=True)
        visible_high = WindowInfo(7, "b", 1, "x.exe")
        visible_low = WindowInfo(3, "c", 1, "x.exe")
        minimized_low = WindowInfo(1, "d", 1, "x.exe", is_minimized=True)

        self.assertEqual(
            _preferred_window([minimized_low, visible_high, foreground, visible_low]).handle,
            9,
        )
        self.assertEqual(
            _preferred_window([minimized_low, visible_high, visible_low]).handle, 3
        )
        self.assertEqual(_preferred_window([minimized_low]).handle, 1)

    def test_m_selection_is_stable_across_input_order(self) -> None:
        windows = [
            WindowInfo(5, "a", 1, "x.exe"),
            WindowInfo(2, "b", 1, "x.exe"),
            WindowInfo(8, "c", 1, "x.exe"),
        ]
        chosen = {_preferred_window(list(reversed(windows))).handle,
                  _preferred_window(windows).handle}
        self.assertEqual(chosen, {2}, "same set, same choice, whatever the order")

    def test_m_a_minimized_target_is_still_reachable_when_it_is_the_only_one(self) -> None:
        minimized = window(202, "Calculator", 901, "CalculatorApp.exe", minimized=True)
        backend = FakeWindowBackend([NOTEPAD, minimized])
        outcome = controller(backend).focus_app("calculator")
        self.assertEqual(outcome.status, "succeeded")


# --------------------------------------------------------------------------- #
# N-Q: the attention fallback
# --------------------------------------------------------------------------- #


class AttentionFallbackTests(unittest.TestCase):
    def test_n_a_denied_focus_requests_attention_without_claiming_success(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(backend.flashed, [CALC.handle])
        self.assertTrue(outcome.evidence["attention_requested"])
        self.assertEqual(outcome.status, "unverified", "flashing is NOT focus")
        self.assertFalse(outcome.evidence["foreground_after"])
        self.assertIn("highlighted it in the taskbar", outcome.detail)

    def test_o_a_failed_flash_changes_nothing_and_does_not_crash(self) -> None:
        backend = FakeWindowBackend(
            [NOTEPAD, CALC], focus_succeeds=False, flash_succeeds=False
        )
        outcome = controller(backend).focus_app("calculator", timeout=1.0)

        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.evidence["attention_requested"])
        self.assertNotIn("taskbar", outcome.detail)

    def test_o_a_raising_flash_is_swallowed(self) -> None:
        class ExplodingFlash(FakeWindowBackend):
            def flash(self, handle, count=3):
                raise OSError("user32 said no")

        backend = ExplodingFlash([NOTEPAD, CALC], focus_succeeds=False)
        outcome = controller(backend).focus_app("calculator", timeout=1.0)
        self.assertEqual(outcome.status, "unverified")
        self.assertFalse(outcome.evidence["attention_requested"])

    def test_attention_can_never_upgrade_a_verdict(self) -> None:
        """Structural sweep: no denied focus, under any flash result, is ever
        reported as succeeded."""
        for flash_ok in (True, False):
            with self.subTest(flash=flash_ok):
                backend = FakeWindowBackend(
                    [NOTEPAD, CALC], focus_succeeds=False, flash_succeeds=flash_ok
                )
                outcome = controller(backend).focus_app("calculator", timeout=1.0)
                self.assertNotEqual(outcome.status, "succeeded")
                self.assertFalse(outcome.evidence["verified"])

    def test_the_flash_count_is_bounded(self) -> None:
        from services.api.app.desktop.windows_api import MAX_FLASH_COUNT

        self.assertGreaterEqual(MAX_FLASH_COUNT, 1)
        self.assertLessEqual(MAX_FLASH_COUNT, 5, "a nudge, not an alarm")


# --------------------------------------------------------------------------- #
# R: open_app partial reality
# --------------------------------------------------------------------------- #


class OpenAppPartialRealityTests(unittest.TestCase):
    def test_r_an_already_open_app_whose_focus_is_denied(self) -> None:
        """The app IS open. Saying nothing about that, or implying the open
        failed, would both be wrong."""
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        outcome = controller(backend).open_app("calculator", timeout=1.0)

        self.assertTrue(outcome.evidence["application_open"])
        self.assertTrue(outcome.evidence["already_running"])
        self.assertFalse(outcome.evidence["foreground_verified"])
        self.assertNotEqual(outcome.status, "succeeded")
        self.assertEqual(backend.launched, [], "must not spawn a second instance")

    def test_r_an_already_open_and_focusable_app_succeeds(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC])
        outcome = controller(backend).open_app("calculator")
        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.evidence["foreground_verified"])

    def test_the_already_running_sentence_is_composed_not_spliced(self) -> None:
        """It used to lower-case the focus outcome's own sentence and glue it on,
        producing "Calculator was already running; calculator is now in front."
        The name belongs in the sentence once, and each branch must still match
        the observed reality rather than round up to success."""
        minimized = window(202, "Calculator", 901, "CalculatorApp.exe", minimized=True)
        cases = {
            "brought_forward": (
                FakeWindowBackend([NOTEPAD, CALC]),
                "Calculator was already running and is now in front.",
            ),
            "already_in_front": (
                FakeWindowBackend([CALC, NOTEPAD]),
                "Calculator was already running and in front.",
            ),
            "denied_with_attention": (
                FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False),
                "Calculator was already running, but Windows did not allow "
                "Bunnelby to bring it to the front, so I highlighted it in the "
                "taskbar.",
            ),
            "denied_without_attention": (
                FakeWindowBackend(
                    [NOTEPAD, CALC], focus_succeeds=False, flash_succeeds=False
                ),
                "Calculator was already running, but Windows did not allow "
                "Bunnelby to bring it to the front.",
            ),
            "restored_but_denied": (
                FakeWindowBackend([NOTEPAD, minimized], focus_succeeds=False),
                "Calculator was already running and has been restored, but "
                "Windows did not allow Bunnelby to bring it to the front, so I "
                "highlighted it in the taskbar.",
            ),
        }
        for label, (backend, expected) in cases.items():
            with self.subTest(case=label):
                outcome = controller(backend).open_app("calculator", timeout=1.0)
                self.assertEqual(outcome.detail, expected)
                self.assertEqual(outcome.detail, screen_reply(outcome))
                # One mention of the app, correctly capitalised, no splice.
                self.assertEqual(outcome.detail.count("alculator"), 1)
                self.assertNotIn("; calculator", outcome.detail)
                self.assertNotIn("running; ", outcome.detail)
                # A denied focus still never claims the window came forward.
                if outcome.status != "succeeded":
                    self.assertNotIn("is now in front", outcome.detail)
                    self.assertFalse(outcome.evidence["foreground_verified"])
                    self.assertTrue(outcome.evidence["application_open"])


# --------------------------------------------------------------------------- #
# S-U: Part 12.1 safety must not regress
# --------------------------------------------------------------------------- #


class SafetyUnchangedTests(unittest.TestCase):
    def test_s_closing_the_windows_shell_is_still_blocked(self) -> None:
        backend = FakeWindowBackend([window(1, "Explorer", 5, "explorer.exe")])
        outcome = controller(backend).close_app("file_explorer")
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(outcome.error_code, "close_not_permitted")
        self.assertEqual(backend.closed, [])

    def test_t_a_title_spoof_still_cannot_be_closed(self) -> None:
        ghost = window(404, "Calculator", 951, "")  # unreadable process
        backend = FakeWindowBackend([ghost])
        outcome = controller(backend).close_app("calculator")
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(backend.closed, [])

    def test_t_an_impostor_process_is_not_closed_as_the_target(self) -> None:
        impostor = window(505, "Calculator", 952, "notepad.exe")
        backend = FakeWindowBackend([impostor])
        controller(backend).close_app("calculator")
        self.assertEqual(backend.closed, [])

    def test_u_an_unregistered_target_is_still_refused(self) -> None:
        for target in ("definitely_not_real", "cmd", "powershell"):
            with self.subTest(target=target):
                outcome = controller(FakeWindowBackend([])).focus_app(target)
                self.assertEqual(outcome.status, "blocked")
                self.assertEqual(outcome.error_code, "unknown_application")

    def test_the_reality_layer_added_no_dangerous_technique(self) -> None:
        import io
        import pathlib
        import tokenize

        forbidden = (
            "AttachThreadInput",
            "AllowSetForegroundWindow",
            "SendInput",
            "mouse_event",
            "SetCursorPos",
            "keybd_event",
            "SetWindowsHookEx",
            "TerminateProcess",
            "taskkill",
            "ReadProcessMemory",
            "SetFocus",
            "InvokePattern",
            "ValuePattern",
        )
        for name in ("controller", "windows_api", "presentation", "models"):
            module = pathlib.Path(
                f"services/api/app/desktop/{name}.py"
            ).read_text(encoding="utf-8")
            kept: list[str] = []
            with io.StringIO(module) as handle:
                for token in tokenize.generate_tokens(handle.readline):
                    if token.type in (tokenize.COMMENT, tokenize.STRING):
                        continue
                    kept.append(token.string)
            code = " ".join(kept)
            for item in forbidden:
                with self.subTest(module=name, forbidden=item):
                    self.assertNotIn(item, code)


# --------------------------------------------------------------------------- #
# Presentation reads evidence, never prose
# --------------------------------------------------------------------------- #


class PresentationTests(unittest.TestCase):
    def focus(self, backend, **kwargs):
        return controller(backend).focus_app("calculator", **kwargs)

    def test_wording_matches_the_observed_reality(self) -> None:
        cases = {
            "already": (FakeWindowBackend([CALC, NOTEPAD]), "already in front"),
            "verified": (FakeWindowBackend([NOTEPAD, CALC]), "is now in front"),
        }
        for label, (backend, expected) in cases.items():
            with self.subTest(case=label):
                outcome = self.focus(backend)
                self.assertIn(expected, screen_reply(outcome))

    def test_a_denied_focus_never_speaks_success(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        outcome = self.focus(backend, timeout=1.0)
        for text in (screen_reply(outcome), spoken_reply(outcome)):
            self.assertNotIn("is now in front", text)
            self.assertNotIn("already in front", text)

    def test_the_spoken_form_reports_restoration_and_attention(self) -> None:
        minimized = window(202, "Calculator", 901, "CalculatorApp.exe", minimized=True)
        backend = FakeWindowBackend([NOTEPAD, minimized], focus_succeeds=False)
        outcome = self.focus(backend, timeout=1.0)
        spoken = spoken_reply(outcome)
        self.assertIn("restored", spoken)
        self.assertIn("taskbar", spoken)
        self.assertEqual(outcome.action, DesktopAction.FOCUS_APP)

    def test_the_audit_payload_carries_the_new_evidence(self) -> None:
        backend = FakeWindowBackend([NOTEPAD, CALC], focus_succeeds=False)
        payload = self.focus(backend, timeout=1.0).audit_payload()
        self.assertIn("evidence_foreground_after", payload)
        self.assertIn("evidence_attention_requested", payload)
        self.assertEqual(payload["status"], "unverified")

    def test_the_audit_payload_still_carries_no_window_titles(self) -> None:
        secret = window(909, "SECRET-Q3-PLAN", 901, "CalculatorApp.exe")
        backend = FakeWindowBackend([NOTEPAD, secret], focus_succeeds=False)
        payload = self.focus(backend, timeout=1.0).audit_payload()
        self.assertNotIn("SECRET", repr(payload))


if __name__ == "__main__":
    unittest.main()
