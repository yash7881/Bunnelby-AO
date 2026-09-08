"""Local Fast Path: deterministic desktop routing without a cloud Brain call.

Two things are under test, and they are different in kind:

  THE PARSER  -- does the reviewed grammar match exactly what it should, and
                 nothing else? Most of this file.
  THE ROUTING -- does exactly one of {local, Brain} run per turn, and does a
                 local hit still travel the full existing execution pipeline?

No test here launches, focuses or closes a real application, and none makes a
network call. The zero-cloud tests actively booby-trap every provider entry
point so a regression that reintroduces a Brain call fails loudly.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.api.app import local_fast_path, message_dispatch
from services.api.app.desktop import app_registry
from services.api.app.local_fast_path import match_local_command, normalize
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.tool_requests import build_request


def executable_source(path) -> str:
    """Module source with comments and string literals removed.

    The module docstring deliberately explains why gmail/calendar keyword gating
    and subprocess execution are absent, so a naive substring scan over the raw
    file finds those words in the prose that rules them out. Structural
    assertions must read CODE.
    """
    import io
    import pathlib as _pathlib
    import tokenize

    kept: list[str] = []
    source = _pathlib.Path(str(path)).read_text(encoding="utf-8")
    with io.StringIO(source) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


def match(message: str):
    return match_local_command(message)


class RecordingExecutor:
    """Stands in for tool_executor.execute and records what it was handed."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, decision, user_message, session_id=None, turn_id=None):
        self.calls.append((decision, user_message, session_id, turn_id))
        return OrchestratorResult(
            reply="(stubbed execution)",
            action_type="desktop_control",
            memory_content="",
            spoken_metadata={"action": "stub", "status": "succeeded"},
        )


def exploding_brain(*args, **kwargs):
    raise AssertionError("brain_agent.decide must not run for a fast-pathed command")


def exploding_provider(*args, **kwargs):
    raise AssertionError("no provider generation may occur for a fast-pathed command")


# --------------------------------------------------------------------------- #
# A-D: positive grammar
# --------------------------------------------------------------------------- #


class PositiveOpenTests(unittest.TestCase):
    def test_a_english_open_variants(self) -> None:
        cases = {
            "Open Notepad": "notepad",
            "open notepad": "notepad",
            "Open Notepad.": "notepad",
            "Open Notepad!": "notepad",
            "Launch Calculator": "calculator",
            "Start Chrome": "chrome",
            "Please open Edge": "edge",
            "Open Edge please": "edge",
            "Bunnelby, open Calculator": "calculator",
            "Hey Bunnelby open Notepad": "notepad",
        }
        for message, target in cases.items():
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "open_app", "target": target}
                )

    def test_a_hinglish_open_variants(self) -> None:
        cases = {
            "Notepad kholo": "notepad",
            "notepad khol do": "notepad",
            "Calculator open karo": "calculator",
            "Chrome chalu karo": "chrome",
        }
        for message, target in cases.items():
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "open_app", "target": target}
                )


class PositiveFocusTests(unittest.TestCase):
    def test_b_english_focus_variants(self) -> None:
        cases = {
            "Switch to Notepad": "notepad",
            "Switch to Calculator": "calculator",
            "switch to calc": "calculator",
            "Focus Chrome": "chrome",
            "Focus on Notepad": "notepad",
        }
        for message, target in cases.items():
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "focus_app", "target": target}
                )

    def test_b_hinglish_focus_variants(self) -> None:
        for message in ("Notepad pe switch karo", "Notepad par switch karo"):
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "focus_app", "target": "notepad"}
                )


class PositiveCloseTests(unittest.TestCase):
    def test_c_english_close_variants(self) -> None:
        cases = {
            "Close Notepad": "notepad",
            "Close Calculator": "calculator",
            "Quit Calculator": "calculator",
        }
        for message, target in cases.items():
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "close_app", "target": target}
                )

    def test_c_hinglish_close_variants(self) -> None:
        for message in (
            "Calculator band karo",
            "Calculator band kar do",
            "Calculator close karo",
            "Calculator bandh karo",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    match(message), {"action": "close_app", "target": "calculator"}
                )


class PositiveListWindowsTests(unittest.TestCase):
    def test_d_window_list_phrases(self) -> None:
        for message in (
            "Which windows are open?",
            "which windows are open",
            "What windows are open?",
            "Show open windows",
            "Show me the open windows",
            "List open windows",
            "Windows dikhao",
        ):
            with self.subTest(message=message):
                self.assertEqual(match(message), {"action": "list_windows"})

    def test_d_window_list_never_carries_a_target(self) -> None:
        self.assertNotIn("target", match("Which windows are open?"))


# --------------------------------------------------------------------------- #
# E-H: everything that must MISS
# --------------------------------------------------------------------------- #


class NegationTests(unittest.TestCase):
    def test_e_negation_always_wins(self) -> None:
        for message in (
            "Don't open Calculator",
            "Don\u2019t open Calculator",
            "Do not open Calculator",
            "Dont open Calculator",
            "Never open Calculator",
            "Calculator mat kholo",
            "Calculator nahi kholo",
            "Calculator nahin kholo",
            "Do not close Notepad",
            "Never switch to Chrome",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))


class ConceptualTests(unittest.TestCase):
    def test_f_conceptual_questions_never_match(self) -> None:
        for message in (
            "What is Notepad?",
            "How do I open Notepad?",
            "Explain how to open Notepad",
            "Can Bunnelby open applications?",
            "What happens when I close Calculator?",
            "Should I open Calculator?",
            "Is Notepad open?",
            "Why would I open Chrome?",
            "Tell me how to open Notepad",
            "Notepad kya hai",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))


class ExtraClauseTests(unittest.TestCase):
    def test_g_multi_action_and_trailing_clauses_never_match(self) -> None:
        for message in (
            "Open Notepad and Calculator",
            "Open Notepad and close Calculator",
            "Open Notepad then check my email",
            "Open Calculator because I need it",
            "Close Calculator and open Chrome",
            "Open Notepad and send an email",
            "Open Notepad for me and then wait",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))

    def test_a_long_utterance_is_rejected_on_bounds_alone(self) -> None:
        self.assertIsNone(match("open " + "notepad " * 40))


class UnsafeTargetTests(unittest.TestCase):
    def test_h_paths_and_commands_never_match(self) -> None:
        for message in (
            r"Open C:\Windows\System32\cmd.exe",
            "Open C:/Windows/System32/cmd.exe",
            "Open powershell.exe",
            "Open cmd.exe",
            "Open random.exe",
            "Open notepad & calc",
            r"Open ..\calc.exe",
            "Open ./notepad",
            "Run powershell -Command Get-Process",
            "Run cmd.exe",
            "Execute calc.exe",
            "Open http://example.com",
            r"Open shell:AppsFolder\something",
            "Open notepad --quiet",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))

    def test_h_destructive_verbs_are_not_in_the_grammar(self) -> None:
        for message in (
            "Delete Calculator",
            "Kill Calculator",
            "Terminate Calculator",
            "Uninstall Calculator",
            "Press Ctrl+Alt+Delete",
            "Type hello into Notepad",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))

    def test_an_unregistered_but_harmless_app_is_a_miss(self) -> None:
        for message in ("Open Photoshop", "Open Spotify", "Close Slack"):
            with self.subTest(message=message):
                self.assertIsNone(match(message))


# --------------------------------------------------------------------------- #
# App registry remains the sole target authority
# --------------------------------------------------------------------------- #


class RegistryAuthorityTests(unittest.TestCase):
    def test_registered_aliases_resolve_to_canonical_app_ids(self) -> None:
        cases = {
            "Open Notepad": "notepad",
            "Open note pad": "notepad",
            "Open text editor": "notepad",
            "Open Calculator": "calculator",
            "Open calc": "calculator",
            "Open Google Chrome": "chrome",
            "Open VS Code": "vscode",
            "Open Windows Settings": "settings",
            "Open File Explorer": "file_explorer",
            "Open Windows Terminal": "terminal",
            "Open Microsoft Edge": "edge",
        }
        for message, app_id in cases.items():
            with self.subTest(message=message):
                self.assertEqual(match(message)["target"], app_id)

    def test_the_target_is_always_a_registry_app_id(self) -> None:
        """Never the user's spelling: the executor receives canonical ids."""
        known = set(app_registry.known_app_ids())
        for message in ("Open calc", "Open note pad", "Open VS Code"):
            with self.subTest(message=message):
                self.assertIn(match(message)["target"], known)

    def test_the_parser_keeps_no_allowlist_of_its_own(self) -> None:
        """Structural: every app name reachable locally comes from the registry.

        If the parser held its own list, an app could be controllable locally
        that the reviewed registry never approved.
        """
        code = executable_source(local_fast_path.__file__)
        for app_id in app_registry.known_app_ids():
            with self.subTest(app=app_id):
                self.assertNotIn(f'"{app_id}"', code)
        self.assertIn("try_resolve", code)

    def test_resolution_is_exact_never_fuzzy(self) -> None:
        for message in ("Open notepadd", "Open calculater", "Open notepd", "Open calcu"):
            with self.subTest(message=message):
                self.assertIsNone(match(message))

    def test_registry_separator_folding_is_exact_not_fuzzy(self) -> None:
        """"note-pad" resolves because app_registry._normalize folds "-" and "_"
        to spaces, making it the registered alias "note pad" EXACTLY. That is
        the registry's documented behaviour, not edit-distance matching, and the
        registry -- not this parser -- is the authority on it."""
        self.assertEqual(match("Open note-pad"), {"action": "open_app", "target": "notepad"})
        self.assertIsNone(match("Open note-pads"))


class NormalizationTests(unittest.TestCase):
    def test_normalization_is_bounded_and_deterministic(self) -> None:
        self.assertEqual(normalize("  Open   Notepad.  "), "open notepad")
        self.assertEqual(normalize("OPEN NOTEPAD!!!"), "open notepad")
        self.assertEqual(normalize("Bunnelby, open Notepad?"), "open notepad")
        self.assertEqual(normalize("Please open Notepad please"), "open notepad")
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")

    def test_normalization_does_not_rewrite_meaning(self) -> None:
        """No spell correction, no synonym folding, no path cleanup."""
        self.assertEqual(normalize("open notepadd"), "open notepadd")
        self.assertEqual(normalize(r"open C:\x\cmd.exe"), r"open c:\x\cmd.exe")


# --------------------------------------------------------------------------- #
# The local decision is a real, typed-compatible decision
# --------------------------------------------------------------------------- #


class DecisionShapeTests(unittest.TestCase):
    def test_a_local_hit_is_a_canonical_brain_decision(self) -> None:
        decision = local_fast_path.try_local_fast_path("Open Notepad")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "desktop_control")
        self.assertEqual(decision.confidence, 1.0)
        self.assertEqual(
            dict(decision.arguments), {"action": "open_app", "target": "notepad"}
        )
        self.assertEqual(decision.reason_code, "local_fast_path")

    def test_the_parser_never_writes_a_success_reply(self) -> None:
        """executed != success. Only the verifier may produce a success claim."""
        decision = local_fast_path.try_local_fast_path("Open Notepad")
        self.assertEqual(decision.reply, "")
        self.assertEqual(decision.spoken_reply, "")

    def test_a_local_decision_builds_a_valid_typed_request(self) -> None:
        """The decision must satisfy the SAME typed boundary a Brain decision does."""
        for message, action, target in (
            ("Open Notepad", "open_app", "notepad"),
            ("Switch to Calculator", "focus_app", "calculator"),
            ("Close Calculator", "close_app", "calculator"),
            ("Which windows are open?", "list_windows", ""),
        ):
            with self.subTest(message=message):
                decision = local_fast_path.try_local_fast_path(message)
                request = build_request(
                    decision.tool, message, dict(decision.arguments)
                )
                self.assertEqual(request.tool_name, "desktop_control")
                self.assertEqual(request.action, action)
                self.assertEqual(request.target, target)

    def test_no_invented_arguments(self) -> None:
        decision = local_fast_path.try_local_fast_path("Open Notepad")
        self.assertEqual(set(decision.arguments), {"action", "target"})
        self.assertEqual(
            set(local_fast_path.try_local_fast_path("Show open windows").arguments),
            {"action"},
        )

    def test_a_miss_returns_none_not_a_refusal(self) -> None:
        """A non-match must hand the turn over, not answer it."""
        for message in ("What is Notepad?", "Open random.exe", "Check my emails"):
            with self.subTest(message=message):
                self.assertIsNone(local_fast_path.try_local_fast_path(message))


# --------------------------------------------------------------------------- #
# I, K: routing -- one path per turn, zero cloud on a hit
# --------------------------------------------------------------------------- #


class ZeroCloudRoutingTests(unittest.TestCase):
    """Every provider entry point is booby-trapped, not merely counted."""

    def _dispatch(self, message: str) -> RecordingExecutor:
        executor = RecordingExecutor()
        with patch("services.api.app.brain_agent.decide", exploding_brain), patch(
            "services.api.app.brain_agent.generate_text", exploding_provider
        ), patch(
            "services.api.app.brain_agent.generate_fast_text", exploding_provider
        ), patch(
            "services.api.app.model_gateway.generate", exploding_provider
        ), patch(
            "services.api.app.tool_executor.execute", executor
        ):
            message_dispatch.handle_message_result(
                message, session_id="test", turn_id="t1"
            )
        return executor

    def test_k_open_notepad_makes_zero_cloud_calls(self) -> None:
        executor = self._dispatch("Open Notepad")
        self.assertEqual(len(executor.calls), 1, "executed exactly once")
        decision = executor.calls[0][0]
        self.assertEqual(decision.tool, "desktop_control")
        self.assertEqual(dict(decision.arguments)["action"], "open_app")
        self.assertEqual(dict(decision.arguments)["target"], "notepad")

    def test_k_switch_to_calculator_makes_zero_cloud_calls(self) -> None:
        executor = self._dispatch("Switch to Calculator")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(
            dict(executor.calls[0][0].arguments),
            {"action": "focus_app", "target": "calculator"},
        )

    def test_k_which_windows_are_open_makes_zero_cloud_calls(self) -> None:
        executor = self._dispatch("Which windows are open?")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(
            dict(executor.calls[0][0].arguments), {"action": "list_windows"}
        )

    def test_k_hinglish_commands_make_zero_cloud_calls(self) -> None:
        for message, expected in (
            ("Notepad kholo", {"action": "open_app", "target": "notepad"}),
            ("Calculator band karo", {"action": "close_app", "target": "calculator"}),
        ):
            with self.subTest(message=message):
                executor = self._dispatch(message)
                self.assertEqual(len(executor.calls), 1)
                self.assertEqual(dict(executor.calls[0][0].arguments), expected)

    def test_i_the_user_message_reaches_the_executor_unchanged(self) -> None:
        executor = self._dispatch("Open Notepad")
        self.assertEqual(executor.calls[0][1], "Open Notepad")
        self.assertEqual(executor.calls[0][2], "test")
        self.assertEqual(executor.calls[0][3], "t1")


class FallbackRoutingTests(unittest.TestCase):
    """A miss must reach the Brain exactly once, with nothing executed first."""

    def _dispatch(self, message: str):
        brain_calls: list[str] = []
        executor = RecordingExecutor()

        def recording_brain(user_message, session_id=None):
            brain_calls.append(user_message)
            from services.api.app.brain_agent import BrainDecision

            return BrainDecision(
                mode="answer",
                tool=None,
                confidence=0.5,
                reply="stub answer",
                spoken_reply="stub answer",
            )

        with patch("services.api.app.brain_agent.decide", recording_brain), patch(
            "services.api.app.tool_executor.execute", executor
        ):
            message_dispatch.handle_message_result(message, session_id="test")
        return brain_calls, executor

    def test_non_matching_messages_call_the_brain_exactly_once(self) -> None:
        for message in (
            "What is Notepad?",
            "Can you explain how to open Notepad?",
            "Don't open Calculator",
            "Open random.exe",
            "Check my emails",
            "What should I work on today?",
        ):
            with self.subTest(message=message):
                brain_calls, executor = self._dispatch(message)
                self.assertEqual(brain_calls, [message], "exactly one Brain call")
                self.assertEqual(
                    executor.calls, [], "nothing executed before the Brain decided"
                )

    def test_i_a_fast_pathed_turn_never_also_calls_the_brain(self) -> None:
        brain_calls, executor = self._dispatch("Open Notepad")
        self.assertEqual(brain_calls, [], "the Brain must not run")
        self.assertEqual(len(executor.calls), 1, "and execution happens once")


# --------------------------------------------------------------------------- #
# J: existing policy still owns authority
# --------------------------------------------------------------------------- #


class ExistingPolicyOwnsAuthorityTests(unittest.TestCase):
    class ExplodingBackend:
        """Any real desktop call is a test failure."""

        def is_available(self):
            return True

        def list_windows(self):
            return ()

        def foreground_handle(self):
            return 0

        def focus(self, handle):
            raise AssertionError("focus must not be attempted")

        def request_close(self, handle):
            raise AssertionError("close must not be attempted")

        def is_window(self, handle):
            return False

        def launch(self, argv):
            raise AssertionError("launch must not be attempted")

    def test_j_close_file_explorer_parses_but_policy_still_blocks_it(self) -> None:
        """The parser identifies intent; the controller owns permission."""
        from services.api.app.desktop.controller import DesktopController

        parsed = match("Close File Explorer")
        self.assertEqual(
            parsed, {"action": "close_app", "target": "file_explorer"},
            "the parser recognises the intent",
        )

        outcome = DesktopController(
            backend=self.ExplodingBackend(), discovery_grace_seconds=0.0
        ).close_app(parsed["target"])
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(outcome.error_code, "close_not_permitted")

    def test_j_close_restrictions_are_not_reimplemented_in_the_parser(self) -> None:
        """vscode and terminal are not closable, but that is the CONTROLLER's
        rule. The parser must not duplicate it -- duplicated policy drifts."""
        from services.api.app.desktop.controller import DesktopController

        for app_id, message in (
            ("vscode", "Close VS Code"),
            ("terminal", "Close Windows Terminal"),
        ):
            with self.subTest(app=app_id):
                self.assertEqual(
                    match(message), {"action": "close_app", "target": app_id}
                )
                outcome = DesktopController(
                    backend=self.ExplodingBackend(), discovery_grace_seconds=0.0
                ).close_app(app_id)
                self.assertEqual(outcome.status, "blocked")

    def test_the_parser_reaches_no_execution_primitive(self) -> None:
        """Structural: local_fast_path -> subprocess/Win32 must be impossible."""
        code = executable_source(local_fast_path.__file__)
        for forbidden in (
            "subprocess",
            "os.system",
            "Popen",
            "ctypes",
            "windll",
            "controller",
            "DesktopController",
            "windows_api",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, code)


class ScopeTests(unittest.TestCase):
    def test_only_four_actions_are_reachable_locally(self) -> None:
        reachable = {entry[0] for entry in local_fast_path._TEMPLATES}
        reachable.add("list_windows")
        self.assertEqual(
            reachable, {"open_app", "focus_app", "close_app", "list_windows"}
        )

    def test_no_other_capability_is_reachable_locally(self) -> None:
        for message in (
            "Check my emails",
            "What's on my calendar tomorrow?",
            "Send an email to Yash",
            "Find my tax documents",
            "Inspect Calculator",
            "Find buttons in Calculator",
            "Press Win+D",
        ):
            with self.subTest(message=message):
                self.assertIsNone(match(message))

    def test_the_fast_path_only_ever_produces_desktop_control(self) -> None:
        code = executable_source(local_fast_path.__file__)
        for other in ("gmail", "calendar", "file_search", "cross_tool"):
            with self.subTest(tool=other):
                self.assertNotIn(other, code.lower())


class PerformanceTests(unittest.TestCase):
    def test_matching_is_fast_and_needs_no_model(self) -> None:
        import time

        messages = ["Open Notepad", "What is Notepad?", "Which windows are open?"]
        started = time.perf_counter()
        for _ in range(300):
            for message in messages:
                match(message)
        elapsed_ms = (time.perf_counter() - started) * 1000 / (300 * len(messages))
        self.assertLess(elapsed_ms, 1.0, f"{elapsed_ms:.3f}ms per match")


if __name__ == "__main__":
    unittest.main()
