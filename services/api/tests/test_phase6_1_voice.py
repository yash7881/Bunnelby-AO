from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from services.api.app import llm_service, main, model_gateway, orchestrator
from services.api.app.acknowledgments import (
    detect_spoken_language,
    normalize_spoken_text,
    select_spoken_response,
)
from services.api.app.llm_service import LLMResult
from services.api.app.llm_service import LLMUnavailableError
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.schemas import ChatRequest
from services.api.app.schemas import TTSRequest
from services.api.app.tts_service import VoiceModelMissingError, length_scale
from fastapi import HTTPException


class FakeSession:
    def __init__(self) -> None:
        self.rows: list[object] = []
        self.committed = False

    def add(self, row: object) -> None:
        self.rows.append(row)

    def commit(self) -> None:
        self.committed = True


class Phase61VoicePolicyTests(unittest.TestCase):
    def test_general_chat_uses_one_generation_for_both_outputs(self) -> None:
        payload = json.dumps(
            {
                "reply": "RAG retrieves external context before a model answers.\n\nIt improves grounding.",
                "spoken_reply": "RAG retrieves relevant context before answering, which improves grounding.",
            }
        )
        with (
            patch.object(orchestrator, "build_memory_context", return_value="LOCAL CONTEXT"),
            patch.object(
                orchestrator,
                "generate_fast_text",
                return_value=LLMResult(text=payload, provider="groq", model="test"),
            ) as generate,
        ):
            result = orchestrator.general_chat_handler("What is RAG?")

        self.assertEqual(generate.call_count, 1)
        self.assertIn(
            "TRUSTED SPOKEN OUTPUT LANGUAGE FOR THIS TURN",
            generate.call_args.kwargs["user_content"],
        )
        self.assertIn("English", generate.call_args.kwargs["user_content"])
        self.assertIn("improves grounding", result.reply)
        self.assertEqual(
            result.spoken_reply,
            "RAG retrieves relevant context before answering, which improves grounding.",
        )

    def test_malformed_model_envelope_preserves_screen_text(self) -> None:
        raw = "A vector database stores and searches embeddings. It supports semantic retrieval."
        result = orchestrator._parse_conversational_output(raw, "What is a vector database?")
        self.assertEqual(result.reply, raw)
        self.assertIn("semantic retrieval", result.spoken_reply or "")

    def test_model_language_mismatch_falls_back_without_another_call(self) -> None:
        payload = json.dumps(
            {
                "reply": "RAG retrieves relevant information before answering. It improves grounding.",
                "spoken_reply": "RAG yaani ek technique hai jisme AI pehle jaankari search karta hai.",
            }
        )
        result = orchestrator._parse_conversational_output(payload, "What is RAG?")
        self.assertIn("improves grounding", result.spoken_reply or "")
        self.assertNotIn("yaani", result.spoken_reply or "")

    def test_hinglish_generation_requests_devanagari_in_same_call(self) -> None:
        payload = json.dumps(
            {
                "reply": "RAG external context use karta hai.",
                "spoken_reply": "आर ए जी जवाब देने से पहले संबंधित जानकारी ढूँढता है।",
            },
            ensure_ascii=False,
        )
        with (
            patch.object(orchestrator, "build_memory_context", return_value="LOCAL CONTEXT"),
            patch.object(
                orchestrator,
                "generate_fast_text",
                return_value=LLMResult(text=payload, provider="gemini", model="test"),
            ) as generate,
        ):
            result = orchestrator.general_chat_handler("RAG kya hota hai?")

        self.assertEqual(generate.call_count, 1)
        self.assertIn("Hindi in natural Devanagari", generate.call_args.kwargs["user_content"])
        self.assertRegex(result.spoken_reply or "", r"[\u0900-\u097f]")

    def test_hinglish_uses_result_aware_devanagari_gmail_speech(self) -> None:
        spoken = select_spoken_response(
            "Mere unread emails check karo.",
            "gmail_summary",
            metadata={"email_count": 5, "unread_only": True},
        )
        self.assertEqual(spoken.language, "hi")
        self.assertIn("पाँच", spoken.text)
        self.assertRegex(spoken.text, r"[\u0900-\u097f]")

    def test_spoken_text_strips_screen_only_syntax_and_limits_words(self) -> None:
        text = "# Result\n- Read [the docs](https://example.com) `now`. " + "word " * 80
        cleaned = normalize_spoken_text(text, "en")
        self.assertNotIn("http", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertLessEqual(len(cleaned.split()), 65)

    def test_language_is_recalculated_per_turn(self) -> None:
        self.assertEqual(detect_spoken_language("RAG kya hota hai?"), "hi")
        self.assertEqual(detect_spoken_language("What is a vector database?"), "en")

    def test_roman_hindi_never_sends_english_fallback_to_rohan(self) -> None:
        spoken = select_spoken_response(
            "RAG kya hota hai?",
            "general_answer",
            preferred_text="RAG retrieves relevant information before answering.",
        )
        self.assertEqual(spoken.language, "hi")
        self.assertRegex(spoken.text, r"[\u0900-\u097f]")
        self.assertNotIn("retrieves", spoken.text)

    def test_chat_returns_new_field_and_prompt6_alias(self) -> None:
        fake_db = FakeSession()
        result = OrchestratorResult(
            reply="Detailed screen answer.",
            action_type="general_answer",
            memory_content="Detailed screen answer.",
            spoken_reply="Useful spoken answer.",
        )
        with patch.object(main, "handle_message_result", return_value=result):
            response = main.chat(ChatRequest(message="Explain this"), fake_db)  # type: ignore[arg-type]

        self.assertEqual(response.spoken_reply, "Useful spoken answer.")
        self.assertEqual(response.spoken_ack, response.spoken_reply)
        self.assertTrue(fake_db.committed)

    def test_gmail_handler_exposes_confirmed_count_not_inferred_priority(self) -> None:
        emails = [
            {"sender": "A", "subject": "One", "snippet": "", "timestamp": ""},
            {"sender": "B", "subject": "Two", "snippet": "", "timestamp": ""},
        ]
        with (
            patch.object(orchestrator, "get_unread_emails", return_value=emails),
            patch.object(orchestrator, "summarize_with_graceful_fallback", return_value="Summary"),
        ):
            result = orchestrator.gmail_handler("Check my unread emails")

        self.assertEqual(result.reply, "Summary")
        self.assertEqual(result.spoken_metadata, {"email_count": 2, "unread_only": True})
        self.assertIsNone(result.spoken_reply)

    def test_unsupported_new_gmail_message_remains_blocked(self) -> None:
        result = orchestrator.gmail_handler("Send an email to Rahul")
        self.assertIn("replies to existing Gmail threads", result.reply)
        self.assertIn("Nothing was changed", result.spoken_reply or "")

    def test_initial_piper_cadence_defaults_are_language_specific(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIPER_EN_LENGTH_SCALE", None)
            os.environ.pop("PIPER_HI_LENGTH_SCALE", None)
            self.assertEqual(length_scale("en"), 1.11)
            self.assertEqual(length_scale("hi"), 1.12)

    def test_gemini_failure_preserves_groq_output_contract_without_summary_call(self) -> None:
        llm_service.clear_gemini_cooldown()
        self.addCleanup(llm_service.clear_gemini_cooldown)
        fallback = LLMResult(
            text='{"reply":"Screen","spoken_reply":"Speech"}',
            provider="groq",
            model="fallback-test",
        )
        # Part 10.2 Phase E: the failover contract is unchanged, but it is now
        # enforced by the Model Gateway, so the provider primitives are the seam.
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test", "GROQ_API_KEY": "test"}),
            patch.object(
                model_gateway,
                "_gemini_call",
                side_effect=LLMUnavailableError("forced primary failure"),
            ) as primary,
            patch.object(model_gateway, "_groq_call", return_value=fallback) as secondary,
        ):
            result = llm_service.generate_text(system_instruction="system", user_content="user")

        self.assertEqual(result.provider, "groq")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(secondary.call_count, 1)

    def test_missing_piper_voice_is_isolated_from_chat(self) -> None:
        with patch.object(
            main,
            "synthesize_speech",
            side_effect=VoiceModelMissingError("missing test voice"),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.text_to_speech(TTSRequest(text="Test speech", language="en"))
        self.assertEqual(raised.exception.status_code, 503)

        fake_db = FakeSession()
        result = OrchestratorResult(
            reply="Text remains available.",
            action_type="general_answer",
            memory_content="Text remains available.",
            spoken_reply="Text remains available.",
        )
        with patch.object(main, "handle_message_result", return_value=result):
            response = main.chat(ChatRequest(message="Continue with text"), fake_db)  # type: ignore[arg-type]
        self.assertEqual(response.reply, "Text remains available.")


if __name__ == "__main__":
    unittest.main()
