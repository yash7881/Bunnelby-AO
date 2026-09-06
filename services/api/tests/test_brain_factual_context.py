"""Desktop execution history must not bias ordinary conversation.

CONTEXT. Live, after a run of desktop commands, "What is Notepad?" came back as
"That phrase is a bit too general... could you clarify?". A fresh session
answered the same question correctly, so neither the provider nor the parser was
at fault: the difference was the CONTEXT the session carried.

`_TOOL_MEMORY_ROUTES` exists precisely to keep tool transcripts out of casual
chat, and `desktop_control` was never added to it when Part 12.1 landed. Worse,
desktop turns record their route as "desktop_control (open_app)" -- with the
action appended -- so a plain membership test misses them even once the name is
present. Both halves are covered here.

No provider call is made by any test in this file.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.api.app import memory_service as ms
from services.api.app.memory_service import MemoryTurn, _base_route, _is_tool_route

DESKTOP_ROUTES = (
    "desktop_control (open_app)",
    "desktop_control (focus_app)",
    "desktop_control (close_app)",
    "desktop_control (list_windows)",
    "desktop_control (inspect_window)",
    "desktop_control (find_control)",
    "desktop_control (safe_shortcut)",
    "desktop_control",
)


def turn(user: str, assistant: str, route: str | None) -> MemoryTurn:
    return MemoryTurn(
        user_id=abs(hash(user)) % 10_000,
        assistant_id=abs(hash(assistant)) % 10_000,
        user=user,
        assistant=assistant,
        route=route,
    )


DESKTOP_HISTORY = [
    turn("Open Notepad", "Notepad is open.", "desktop_control (open_app)"),
    turn(
        "Which windows are open?",
        "7 open window(s): 1. Untitled - Notepad - notepad.exe",
        "desktop_control (list_windows)",
    ),
    turn("Close Calculator", "Calculator closed.", "desktop_control (close_app)"),
]


def context_for(message: str, history: list[MemoryTurn]) -> str:
    with patch.object(ms, "_load_safe_turns", lambda session_id=None: history):
        return ms.build_memory_context(message, session_id="session-under-test")


class RouteClassificationTests(unittest.TestCase):
    def test_every_desktop_route_is_recognised_as_a_tool_route(self) -> None:
        for route in DESKTOP_ROUTES:
            with self.subTest(route=route):
                self.assertTrue(_is_tool_route(route))
                self.assertEqual(_base_route(route), "desktop_control")

    def test_existing_tool_routes_are_unchanged(self) -> None:
        for route in ("gmail", "calendar", "cross_tool", "file_search"):
            with self.subTest(route=route):
                self.assertTrue(_is_tool_route(route))

    def test_conversational_routes_are_not_tool_routes(self) -> None:
        for route in ("brain", "answer", "", None, "memory"):
            with self.subTest(route=route):
                self.assertFalse(_is_tool_route(route))

    def test_normalization_is_case_and_whitespace_insensitive(self) -> None:
        for route in (" GMAIL ", "Desktop_Control (Open_App)", "DESKTOP_CONTROL"):
            with self.subTest(route=route):
                self.assertTrue(_is_tool_route(route))

    def test_normalization_does_not_over_match(self) -> None:
        """A different route that merely starts similarly is still not a tool."""
        self.assertFalse(_is_tool_route("desktop_notes"))
        self.assertFalse(_is_tool_route("gmailish"))


class FactualQuestionContextTests(unittest.TestCase):
    def test_desktop_history_is_excluded_from_a_self_contained_question(self) -> None:
        for question in (
            "What is Notepad?",
            "What is Calculator?",
            "What is File Explorer?",
            "What is Windows Terminal?",
            "How does Alt+Tab work?",
            "Explain what Notepad is.",
        ):
            with self.subTest(question=question):
                context = context_for(question, DESKTOP_HISTORY)
                self.assertNotIn("Open Notepad", context)
                self.assertNotIn("Close Calculator", context)

    def test_window_titles_never_reach_a_casual_question_context(self) -> None:
        """Screen text is attacker-influenceable; it must not be replayed."""
        context = context_for("What is Notepad?", DESKTOP_HISTORY)
        self.assertNotIn("notepad.exe", context)
        self.assertNotIn("7 open window(s)", context)

    def test_ordinary_conversation_is_still_remembered(self) -> None:
        """The fix must not blind Bunnelby to real conversation."""
        history = [
            turn("My name is Parth", "Nice to meet you, Parth.", "brain"),
            turn("I work on Bunnelby", "Understood.", "brain"),
        ]
        context = context_for("What is Notepad?", history)
        self.assertIn("My name is Parth", context)

    def test_gmail_and_calendar_behaviour_is_unchanged(self) -> None:
        history = [
            turn("Check my emails", "You have 3 emails.", "gmail"),
            turn("What's on my calendar?", "Two meetings.", "calendar"),
        ]
        context = context_for("What is Notepad?", history)
        self.assertNotIn("Check my emails", context)
        self.assertNotIn("Two meetings", context)


class ExplicitFollowUpStillSeesToolHistoryTests(unittest.TestCase):
    """Filtering is scoped to unrelated chat, not applied unconditionally."""

    def test_a_direct_follow_up_still_receives_desktop_history(self) -> None:
        for question in ("What about it?", "And that one?", "Then close it"):
            with self.subTest(question=question):
                context = context_for(question, DESKTOP_HISTORY)
                self.assertIn("Open Notepad", context)

    def test_a_temporal_recall_still_receives_desktop_history(self) -> None:
        context = context_for("What did I do last?", DESKTOP_HISTORY)
        self.assertIn("Open Notepad", context)

    def test_replayed_desktop_history_is_wrapped_as_untrusted(self) -> None:
        """When it IS replayed, its provenance travels with it.

        Part 12.1 wraps window titles as untrusted at execution time. Memory
        replay must not launder them back into trusted conversation.
        """
        context = context_for("What about it?", DESKTOP_HISTORY)
        self.assertIn("desktop_control", context.lower())
        self.assertNotIn(
            "Bunnelby: 7 open window(s): 1. Untitled - Notepad - notepad.exe",
            context,
            "the raw reply must not appear unwrapped",
        )


class AmbiguityStillClarifiesTests(unittest.TestCase):
    """Do not globally suppress clarification."""

    def test_bare_ambiguous_inputs_are_not_fast_pathed_into_actions(self) -> None:
        from services.api.app.local_fast_path import try_local_fast_path

        for text in ("Notepad", "Email", "Calendar", "Do it", "Open it"):
            with self.subTest(text=text):
                self.assertIsNone(try_local_fast_path(text))

    def test_write_tools_still_fail_closed_on_low_confidence(self) -> None:
        from services.api.app.brain_agent import _parse_decision

        payload = """{
          "mode": "tool", "tool": "gmail_compose", "confidence": 0.3,
          "arguments": {"recipient_hint": "someone"},
          "reply": "Sending.", "spoken_reply": "Sending."
        }"""
        decision = _parse_decision(payload, "email someone")
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)

    def test_write_tools_still_fail_closed_on_missing_arguments(self) -> None:
        from services.api.app.brain_agent import _parse_decision

        payload = """{
          "mode": "tool", "tool": "calendar_create", "confidence": 0.99,
          "arguments": {},
          "reply": "Booked.", "spoken_reply": "Booked."
        }"""
        decision = _parse_decision(payload, "book a meeting")
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)


class NoHardcodingTests(unittest.TestCase):
    def test_the_fix_names_no_application(self) -> None:
        """No app-specific LOGIC. The docstrings name Notepad to record the
        incident, so this reads code with comments and strings stripped."""
        import io
        import pathlib
        import tokenize

        kept: list[str] = []
        source = pathlib.Path(ms.__file__).read_text(encoding="utf-8")
        with io.StringIO(source) as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                kept.append(token.string)
        code = " ".join(kept).lower()
        for name in ("notepad", "calculator", "explorer", "chrome"):
            with self.subTest(name=name):
                self.assertNotIn(name, code)


class DefinitionalQuestionScopeTests(unittest.TestCase):
    """A cue word inside a definitional question is not a reference to data."""

    def test_definitional_questions_do_not_pull_tool_history(self) -> None:
        for question in (
            "What is File Explorer?",
            "What is Gmail?",
            "What is Google Calendar?",
            "Explain the difference between Gmail and Google Calendar",
            "What does a calendar event mean?",
        ):
            with self.subTest(question=question):
                self.assertTrue(ms._is_self_contained_definitional_question(question))

    def test_questions_about_the_users_own_data_still_pull_history(self) -> None:
        for question in (
            "What is on my calendar?",
            "What are my recent emails?",
            "What is my inbox like?",
        ):
            with self.subTest(question=question):
                self.assertFalse(ms._is_self_contained_definitional_question(question))

    def test_a_bare_pronoun_question_keeps_its_context(self) -> None:
        for question in ("What is it?", "What is that?", "What are those?"):
            with self.subTest(question=question):
                self.assertFalse(ms._is_self_contained_definitional_question(question))

    def test_commands_are_not_definitional(self) -> None:
        for text in ("Check my emails", "Open Notepad", "Send an email to Yash"):
            with self.subTest(text=text):
                self.assertFalse(ms._is_self_contained_definitional_question(text))

    def test_file_explorer_question_excludes_desktop_history_end_to_end(self) -> None:
        context = context_for("What is File Explorer?", DESKTOP_HISTORY)
        self.assertNotIn("Open Notepad", context)


if __name__ == "__main__":
    unittest.main()
