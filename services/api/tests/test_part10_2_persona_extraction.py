from __future__ import annotations

import ast
import pathlib
import unittest
from datetime import datetime
from unittest.mock import patch

from services.api.app import brain_agent, orchestrator, persona

APP_DIR = pathlib.Path(__file__).resolve().parents[1] / "app"


def local_imports(module_name: str) -> set[str]:
    """Top-level package-relative imports declared by one app module."""
    tree = ast.parse((APP_DIR / f"{module_name}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found.update(alias.name for alias in node.names)
    return found


class PersonaExtractionTests(unittest.TestCase):
    """Phase C: the Brain must no longer reach into the legacy router."""

    def test_brain_agent_does_not_import_the_legacy_router(self) -> None:
        self.assertNotIn(
            "orchestrator",
            local_imports("brain_agent"),
            "brain_agent importing orchestrator is the Part 10.2 import cycle.",
        )

    def test_brain_agent_imports_persona(self) -> None:
        self.assertIn("persona", local_imports("brain_agent"))

    def test_persona_has_no_local_dependencies(self) -> None:
        self.assertEqual(
            local_imports("persona"),
            set(),
            "persona must stay a leaf module so it can never re-introduce a cycle.",
        )

    def test_orchestrator_still_re_exports_every_moved_symbol(self) -> None:
        # Existing tests and any external caller reference these through
        # orchestrator; the re-export keeps those call sites working.
        for name in (
            "AO_CHAT_SYSTEM_INSTRUCTION",
            "_SIMPLE_GREETING_PATTERN",
            "_COMPLEX_GENERAL_CHAT_PATTERN",
            "_time_appropriate_greeting",
            "_general_chat_inference_profile",
        ):
            self.assertIs(
                getattr(orchestrator, name),
                getattr(persona, name),
                f"orchestrator.{name} must be the same object as persona.{name}",
            )

    def test_public_and_legacy_aliases_are_the_same_object(self) -> None:
        self.assertIs(persona.SIMPLE_GREETING_PATTERN, persona._SIMPLE_GREETING_PATTERN)
        self.assertIs(
            persona.COMPLEX_GENERAL_CHAT_PATTERN, persona._COMPLEX_GENERAL_CHAT_PATTERN
        )
        self.assertIs(persona.time_appropriate_greeting, persona._time_appropriate_greeting)
        self.assertIs(
            persona.general_chat_inference_profile, persona._general_chat_inference_profile
        )


class PersonaBehaviorUnchangedTests(unittest.TestCase):
    """The extraction must be behavior-neutral, not merely import-neutral."""

    def test_system_instruction_retains_its_load_bearing_policy_clauses(self) -> None:
        instruction = persona.AO_CHAT_SYSTEM_INSTRUCTION
        for clause in (
            "You are Bunnelby, a professional personal desktop AI assistant.",
            '{"reply":"complete response for the screen","spoken_reply":"concise response to speak"}',
            "NEVER proactively offer Gmail",
            "Do not expose internal routing labels",
            "spoken_reply must contain no Markdown",
        ):
            self.assertIn(clause, instruction)

    def test_system_instruction_is_stripped_and_non_trivial(self) -> None:
        instruction = persona.AO_CHAT_SYSTEM_INSTRUCTION
        self.assertEqual(instruction, instruction.strip())
        self.assertGreater(len(instruction), 4000)

    def test_brain_uses_the_persona_instruction_as_its_base(self) -> None:
        self.assertTrue(
            brain_agent.BRAIN_SYSTEM_INSTRUCTION.startswith(
                persona.AO_CHAT_SYSTEM_INSTRUCTION
            ),
            "BRAIN_SYSTEM_INSTRUCTION must still be the persona plus routing policy.",
        )

    def test_simple_greetings_match(self) -> None:
        for greeting in (
            "hi",
            "Hello",
            "hey",
            "hey bunnelby",
            "hello ao",
            "good morning",
            "Good Evening!",
            "  hi!  ",
        ):
            self.assertIsNotNone(
                persona.SIMPLE_GREETING_PATTERN.match(greeting), greeting
            )

    def test_non_greetings_do_not_match(self) -> None:
        for text in (
            "hi, can you check my email",
            "hello world program",
            "good morning meeting at 9",
            "what is RAG",
        ):
            self.assertIsNone(persona.SIMPLE_GREETING_PATTERN.match(text), text)

    def test_time_appropriate_greeting_covers_every_daypart(self) -> None:
        cases = {
            7: "Good morning, sir. How can I help?",
            14: "Good afternoon, sir. How can I help?",
            19: "Good evening, sir. How can I help?",
            2: "Hello, sir. How can I help?",
        }
        for hour, expected in cases.items():
            # _time_appropriate_greeting() calls .astimezone(), so the frozen
            # moment must already carry the local offset or it shifts dayparts.
            moment = datetime(2026, 9, 3, hour, 0).astimezone()

            class _FrozenDatetime(datetime):
                @classmethod
                def now(cls, tz=None):  # noqa: ANN001
                    return moment

            with patch.object(persona, "datetime", _FrozenDatetime):
                self.assertEqual(persona.time_appropriate_greeting(), expected, hour)

    def test_inference_profile_selection(self) -> None:
        self.assertEqual(persona.general_chat_inference_profile("my brother is lazy"), "fast")
        self.assertEqual(persona.general_chat_inference_profile("hello"), "fast")
        self.assertEqual(
            persona.general_chat_inference_profile("analyze this architecture"), "balanced"
        )
        self.assertEqual(
            persona.general_chat_inference_profile("compare the trade-offs"), "balanced"
        )
        self.assertEqual(persona.general_chat_inference_profile("x" * 700), "balanced")


class BrainFastPathsUnchangedTests(unittest.TestCase):
    """The two zero-cost Brain fast paths must still bypass the model entirely."""

    def _no_llm(self):
        def _fail(**_: object) -> object:
            raise AssertionError("a fast path must not call any model provider")

        return patch.multiple(
            brain_agent, generate_fast_text=_fail, generate_text=_fail
        )

    def test_greeting_fast_path_answers_locally(self) -> None:
        with self._no_llm():
            decision = brain_agent.decide("hello")
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.reason, "simple greeting fast path")
        self.assertIn("How can I help?", decision.reply)

    def test_empty_message_answers_locally(self) -> None:
        with self._no_llm():
            decision = brain_agent.decide("   ")
        self.assertEqual(decision.mode, "answer")
        self.assertEqual(decision.reason, "empty message")

    def test_identity_fast_path_answers_locally(self) -> None:
        with self._no_llm():
            decision = brain_agent.decide("who are you")
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.reason, "local identity fast path")


if __name__ == "__main__":
    unittest.main()
