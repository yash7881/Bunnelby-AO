from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import brain_agent, personal_facts
from services.api.app.database import Base

# Personal Facts Memory V1.
#
# Covers: deterministic extraction (no LLM), never-guess behavior, upsert
# persistence, cross-"session"/cross-"restart" durability (simulated by
# reopening a fresh SQLAlchemy Session against the same on-disk SQLite file,
# exactly as a real app restart would), and retrieval into brain_agent's
# Brain context so a compound question like "what is my and my father's
# name?" has the answer available without re-reading global chat history.


class PersonalFactExtractionTests(unittest.TestCase):
    """Deterministic pattern matching only -- no LLM, no inference."""

    def test_fathers_name_with_apostrophe(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("My father's name is Rajesh."),
            ("father", "Rajesh"),
        )

    def test_fathers_name_without_apostrophe_voice_transcribed(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("my fathers name is Rajesh"),
            ("father", "Rajesh"),
        )

    def test_dad_is_called_variant(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("my dad is called Suresh"),
            ("father", "Suresh"),
        )

    def test_mothers_name(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("My mom's name is Sunita"),
            ("mother", "Sunita"),
        )

    def test_two_word_name_stops_at_punctuation(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("My son's name is John Smith."),
            ("son", "John Smith"),
        )

    def test_trailing_clause_does_not_get_captured(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact(
                "My father's name is Rajesh and he lives in Mumbai"
            ),
            ("father", "Rajesh"),
        )

    def test_self_name(self) -> None:
        self.assertEqual(
            personal_facts.extract_personal_fact("My name is Parth"),
            ("self", "Parth"),
        )

    def test_ordinary_sentence_about_a_relation_is_not_a_fact(self) -> None:
        """'my brother is lazy' must never be mistaken for a naming statement."""
        self.assertIsNone(personal_facts.extract_personal_fact("my brother is too lazy"))

    def test_unrelated_message_extracts_nothing(self) -> None:
        self.assertIsNone(personal_facts.extract_personal_fact("What's the weather today?"))

    def test_negated_statement_is_never_guessed(self) -> None:
        """Never infer or hallucinate: an uncertain/negated statement stays unsaved."""
        self.assertIsNone(
            personal_facts.extract_personal_fact("My father's name is not sure right now")
        )

    def test_empty_message(self) -> None:
        self.assertIsNone(personal_facts.extract_personal_fact(""))


class PersonalFactPersistenceTests(unittest.TestCase):
    """Storage layer: upsert, bounded read, and restart/new-session durability."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "facts.db"
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session_patch = patch.object(personal_facts, "SessionLocal", self.Session)
        self.session_patch.start()

    def tearDown(self) -> None:
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def test_save_then_load_roundtrip(self) -> None:
        personal_facts.save_personal_fact("father", "Rajesh", session_id="s1")
        self.assertEqual(personal_facts.load_personal_facts(), {"father": "Rajesh"})

    def test_restating_a_fact_updates_in_place_not_duplicates(self) -> None:
        personal_facts.save_personal_fact("father", "Rajesh", session_id="s1")
        personal_facts.save_personal_fact("father", "Ramesh", session_id="s2")
        facts = personal_facts.load_personal_facts()
        self.assertEqual(facts, {"father": "Ramesh"})

    def test_survives_a_simulated_app_restart(self) -> None:
        """A restart just means a brand-new SQLAlchemy Session against the
        same on-disk file with no shared process memory -- exactly what a
        real app restart looks like for a SQLite-backed store."""
        personal_facts.try_save_stated_fact("My father's name is Rajesh")

        fresh_session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        with patch.object(personal_facts, "SessionLocal", fresh_session_factory):
            facts_after_restart = personal_facts.load_personal_facts()

        self.assertEqual(facts_after_restart.get("father"), "Rajesh")

    def test_survives_a_new_session_id(self) -> None:
        """Facts are global, not scoped to the session that stated them."""
        personal_facts.try_save_stated_fact(
            "My father's name is Rajesh", session_id="session-A"
        )
        # A brand new conversation/session should still see the fact.
        context = personal_facts.build_personal_facts_context()
        self.assertIn("Rajesh", context)

    def test_non_matching_message_never_writes_to_storage(self) -> None:
        result = personal_facts.try_save_stated_fact("What's the weather today?")
        self.assertIsNone(result)
        self.assertEqual(personal_facts.load_personal_facts(), {})

    def test_context_is_empty_string_when_no_facts_known(self) -> None:
        self.assertEqual(personal_facts.build_personal_facts_context(), "")

    def test_context_renders_known_facts_with_readable_labels(self) -> None:
        personal_facts.save_personal_fact("father", "Rajesh")
        personal_facts.save_personal_fact("mother", "Sunita")
        context = personal_facts.build_personal_facts_context()
        self.assertIn("your father's name: Rajesh", context)
        self.assertIn("your mother's name: Sunita", context)

    def test_fact_saved_reply_mentions_the_value(self) -> None:
        reply = personal_facts.fact_saved_reply(("father", "Rajesh"))
        self.assertIn("Rajesh", reply)
        self.assertIn("father", reply)


def _fake_llm_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(payload), provider="gemini", model="fixture")


class BrainAgentPersonalFactsIntegrationTests(unittest.TestCase):
    """End-to-end through brain_agent.decide(): statement -> save -> retrieval."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "facts.db"
        self.engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session_patch = patch.object(personal_facts, "SessionLocal", self.Session)
        self.session_patch.start()
        self.memory_patch = patch.object(
            brain_agent, "build_memory_context", return_value="MEMORY"
        )
        self.memory_patch.start()

    def tearDown(self) -> None:
        self.memory_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def test_stating_a_fact_saves_it_and_short_circuits_without_calling_the_llm(self) -> None:
        with patch.object(brain_agent, "generate_text") as mock_llm, patch.object(
            brain_agent, "generate_fast_text"
        ) as mock_fast_llm:
            decision = brain_agent.decide(
                "My father's name is Rajesh", session_id="session-A"
            )

        mock_llm.assert_not_called()
        mock_fast_llm.assert_not_called()
        self.assertEqual(decision.mode, "answer")
        self.assertIn("Rajesh", decision.reply)
        self.assertEqual(
            personal_facts.load_personal_facts().get("father"), "Rajesh"
        )

    def test_a_later_question_in_a_new_session_sees_the_fact_in_brain_context(self) -> None:
        """This is the exact bug scenario: a fact stated in one session must be
        retrievable by a later, unrelated session -- not replayed from chat
        history, but injected as a standing fact in the Brain's context."""
        brain_agent.decide("My father's name is Rajesh", session_id="session-A")

        captured_user_content: dict[str, str] = {}

        def _capture(*, system_instruction, user_content, response_schema):
            captured_user_content["value"] = user_content
            payload = {
                "mode": "answer",
                "tool": None,
                "confidence": 0.9,
                "arguments": {},
                "reply": "Your name is Parth and your father's name is Rajesh.",
                "spoken_reply": "Your name is Parth and your father's name is Rajesh.",
                "reason": "answered from persisted personal facts",
            }
            return _fake_llm_result(payload)

        with patch.object(brain_agent, "generate_text", side_effect=_capture), patch.object(
            brain_agent, "generate_fast_text", side_effect=_capture
        ):
            decision = brain_agent.decide(
                "what is my and my father's name?", session_id="session-B-brand-new"
            )

        self.assertIn("PERSISTED PERSONAL FACTS", captured_user_content["value"])
        self.assertIn("Rajesh", captured_user_content["value"])
        self.assertEqual(decision.mode, "answer")

    def test_no_facts_yet_means_no_facts_section_is_injected(self) -> None:
        captured_user_content: dict[str, str] = {}

        def _capture(*, system_instruction, user_content, response_schema):
            captured_user_content["value"] = user_content
            payload = {
                "mode": "answer",
                "tool": None,
                "confidence": 0.9,
                "arguments": {},
                "reply": "I don't have that on file yet.",
                "spoken_reply": "I don't have that on file yet.",
                "reason": "no fact known",
            }
            return _fake_llm_result(payload)

        with patch.object(brain_agent, "generate_text", side_effect=_capture), patch.object(
            brain_agent, "generate_fast_text", side_effect=_capture
        ):
            brain_agent.decide("what is my father's name?", session_id="session-C")

        self.assertNotIn("PERSISTED PERSONAL FACTS", captured_user_content["value"])


if __name__ == "__main__":
    unittest.main()
