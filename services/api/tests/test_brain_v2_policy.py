from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import (
    brain_agent,
    intelligence_dispatch,
    memory_service,
    message_dispatch,
    orchestrator,
    tool_executor,
)
from services.api.app.database import Base
from services.api.app.models import Message


class BrainV2RoutingTests(unittest.TestCase):
    def test_casual_family_statement_is_general_chat(self) -> None:
        samples = (
            "my brother is too lazy",
            "mera bhai bahut lazy hai",
            "my brother is very lazy today",
            "bhai aaj kuch kaam nahi kar raha",
        )

        for sample in samples:
            decision = orchestrator._local_pre_route(sample)
            self.assertIsNotNone(decision)
            self.assertEqual(
                decision.intent,
                "general_chat",
                sample,
            )

    def test_explicit_tools_still_route_to_tools(self) -> None:
        self.assertEqual(
            orchestrator._local_pre_route(
                "check my unread emails"
            ).intent,
            "gmail",
        )

        self.assertEqual(
            orchestrator._local_pre_route(
                "show my calendar tomorrow"
            ).intent,
            "calendar",
        )

    def test_simple_chat_uses_fast_profile(self) -> None:
        self.assertEqual(
            orchestrator._general_chat_inference_profile(
                "my brother is too lazy"
            ),
            "fast",
        )

    def test_general_chat_system_policy_forbids_unrelated_tool_offers(self) -> None:
        self.assertIn(
            "NEVER proactively offer Gmail",
            orchestrator.AO_CHAT_SYSTEM_INSTRUCTION,
        )
        self.assertIn(
            "A casual personal statement is conversation",
            orchestrator.AO_CHAT_SYSTEM_INSTRUCTION,
        )
        self.assertIn(
            "Do not infer negative traits about the user",
            orchestrator.AO_CHAT_SYSTEM_INSTRUCTION,
        )


    def test_complex_work_keeps_balanced_profile(self) -> None:
        self.assertEqual(
            orchestrator._general_chat_inference_profile(
                "Analyze this architecture and design a production security plan."
            ),
            "balanced",
        )


class BrainV2MemoryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

        db_path = Path(self.tempdir.name) / "memory.db"

        self.engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )

        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        self.session_patch = patch.object(
            memory_service,
            "SessionLocal",
            self.Session,
        )
        self.session_patch.start()

        self.profile_patch = patch.object(
            memory_service,
            "load_user_profile",
            return_value=memory_service.UserProfile(
                profile_id="test",
                preferred_name="Parth",
                assistant_name="Bunnelby",
            ),
        )
        self.profile_patch.start()

    def tearDown(self) -> None:
        self.profile_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def _add_turn(
        self,
        user: str,
        assistant: str,
    ) -> None:
        with self.Session() as db:
            db.add(
                Message(
                    role="user",
                    content=user,
                )
            )
            db.add(
                Message(
                    role="assistant",
                    content=assistant,
                )
            )
            db.commit()

    def test_unrelated_tool_history_is_not_injected(self) -> None:
        self._add_turn(
            "Check my unread emails",
            "You have two unread messages about invoices.\n"
            "Route: gmail\n"
            "Why: verified Gmail lookup",
        )

        self._add_turn(
            "My brother likes cricket",
            "Noted. Cricket sounds like one of his interests.",
        )

        context = memory_service.build_memory_context(
            "my brother is too lazy"
        )

        self.assertNotIn(
            "two unread messages about invoices",
            context,
        )

        self.assertIn(
            "My brother likes cricket",
            context,
        )

    def test_explicit_temporal_recall_can_use_tool_history(self) -> None:
        self._add_turn(
            "Check my unread emails",
            "You have two unread messages.\n"
            "Route: gmail\n"
            "Why: verified Gmail lookup",
        )

        context = memory_service.build_memory_context(
            "hamne last kya kaam kiya?"
        )

        self.assertIn(
            "two unread messages",
            context,
        )


def _fake_llm_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(payload), provider="gemini", model="fixture")


class BrainAgentDecisionTests(unittest.TestCase):
    """Covers brain_agent.decide() with the underlying LLM call mocked; no live network calls."""

    def setUp(self) -> None:
        self.memory_patch = patch.object(
            brain_agent, "build_memory_context", return_value="MEMORY"
        )
        self.memory_patch.start()

    def tearDown(self) -> None:
        self.memory_patch.stop()

    def _decide_with(self, message: str, payload: dict) -> brain_agent.BrainDecision:
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ) as fast, patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ):
            decision = brain_agent.decide(message)
        return decision

    def test_general_chat_english_knowledge_question_is_answer(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.95,
            "arguments": {},
            "reply": "Gmail is Google's email service.",
            "spoken_reply": "Gmail is Google's email service.",
            "reason": "knowledge question",
        }
        decision = self._decide_with("What is Gmail?", payload)
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_hinglish_meta_question_is_answer(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "Calendar database events store karta hai.",
            "spoken_reply": "कैलेंडर डेटाबेस इवेंट्स स्टोर करता है।",
            "reason": "hinglish knowledge question",
        }
        decision = self._decide_with("calendar database kya hota hai", payload)
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_casual_statement_mentioning_meeting_is_answer(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "Sorry to hear that.",
            "spoken_reply": "Sorry to hear that.",
            "reason": "casual personal statement",
        }
        decision = self._decide_with("My meeting with Rahul was terrible", payload)
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_bare_ambiguous_word_is_clarify(self) -> None:
        payload = {
            "mode": "clarify",
            "tool": None,
            "confidence": 0.3,
            "arguments": {},
            "reply": "What would you like to do with email?",
            "spoken_reply": "What would you like to do with email?",
            "reason": "bare ambiguous word",
        }
        decision = self._decide_with("email", payload)
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)

    def test_explicit_gmail_read_is_tool(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "gmail_read",
            "confidence": 0.95,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit read",
        }
        decision = self._decide_with("Check my latest emails", payload)
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "gmail_read")

    def test_explicit_calendar_read_is_tool(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "calendar_read",
            "confidence": 0.95,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit read",
        }
        decision = self._decide_with("What's on my calendar tomorrow?", payload)
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "calendar_read")

    def test_explicit_gmail_write_with_high_confidence_and_args_is_tool(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "gmail_compose",
            "confidence": 0.9,
            "arguments": {"recipient_hint": "test@example.com", "body_hint": "hello"},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit compose",
        }
        decision = self._decide_with(
            "Send an email to test@example.com saying hello", payload
        )
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "gmail_compose")

    def test_gmail_write_downgrades_to_clarify_on_low_confidence(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "gmail_compose",
            "confidence": 0.4,
            "arguments": {"recipient_hint": "someone"},
            "reply": "",
            "spoken_reply": "",
            "reason": "uncertain compose",
        }
        decision = self._decide_with("maybe email someone", payload)
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)

    def test_calendar_write_downgrades_to_clarify_on_missing_arguments(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "calendar_create",
            "confidence": 0.9,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "missing title",
        }
        decision = self._decide_with("schedule something", payload)
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)

    def test_explicit_calendar_write_with_confidence_and_args_is_tool(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "calendar_create",
            "confidence": 0.85,
            "arguments": {"title": "Team sync", "start_hint": "tomorrow 9pm"},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit create",
        }
        decision = self._decide_with("Schedule a meeting tomorrow at 9 PM", payload)
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "calendar_create")

    def test_garbled_voice_input_is_clarify(self) -> None:
        payload = {
            "mode": "clarify",
            "tool": None,
            "confidence": 0.2,
            "arguments": {},
            "reply": "I didn't catch that clearly. Could you repeat it?",
            "spoken_reply": "I didn't catch that clearly. Could you repeat it?",
            "reason": "garbled transcript",
        }
        decision = self._decide_with("uhh yeah so like the thing um send", payload)
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)


class BrainDispatchIntegrationTests(unittest.TestCase):
    """Covers message_dispatch.handle_message_result routing through brain_agent + tool_executor."""

    def test_general_chat_never_touches_gmail_or_calendar_execution(self) -> None:
        decision = brain_agent.BrainDecision(
            mode="answer",
            tool=None,
            confidence=0.9,
            reply="Gmail is Google's email service.",
            spoken_reply="Gmail is Google's email service.",
            reason="knowledge question",
        )
        with patch.object(brain_agent, "decide", return_value=decision) as mocked_decide, patch.object(
            tool_executor, "execute"
        ) as mocked_execute:
            result = message_dispatch.handle_message_result("What is Gmail?")

        mocked_decide.assert_called_once()
        mocked_execute.assert_not_called()
        self.assertEqual(result.action_type, "general_answer")
        self.assertIn("Gmail is Google's email service.", result.reply)

    def test_ambiguous_bare_word_clarifies_without_tool_execution(self) -> None:
        decision = brain_agent.BrainDecision(
            mode="clarify",
            tool=None,
            confidence=0.3,
            reply="What would you like to do with your calendar?",
            spoken_reply="What would you like to do with your calendar?",
            reason="bare ambiguous word",
        )
        with patch.object(brain_agent, "decide", return_value=decision), patch.object(
            tool_executor, "execute"
        ) as mocked_execute:
            result = message_dispatch.handle_message_result("calendar")

        mocked_execute.assert_not_called()
        self.assertEqual(result.action_type, "clarification_required")

    def test_gmail_write_reaches_approval_required_without_direct_send(self) -> None:
        decision = brain_agent.BrainDecision(
            mode="tool",
            tool="gmail_compose",
            confidence=0.9,
            arguments={"recipient_hint": "test@example.com"},
            reply="",
            spoken_reply="",
            reason="explicit compose",
        )
        fake_draft = {
            "mode": "compose",
            "thread_id": "",
            "source_message_id": "",
            "source_rfc_message_id": "",
            "references": "",
            "to": "test@example.com",
            "recipient_display": "test@example.com",
            "subject": "Hello",
            "body": "Hello there.",
            "instruction": "Send an email to test@example.com saying hello",
            "provider": "groq",
            "status": "draft",
        }
        fake_approval = SimpleNamespace(id=1)
        with patch.object(brain_agent, "decide", return_value=decision), patch.object(
            message_dispatch, "draft_new_email_from_request", return_value=fake_draft
        ) as draft_fn, patch.object(
            message_dispatch, "create_gmail_compose_approval", return_value=fake_approval
        ) as approval_fn, patch.object(
            message_dispatch, "approval_public_dict", return_value={"id": 1}
        ):
            result = message_dispatch.handle_message_result(
                "Send an email to test@example.com saying hello"
            )

        draft_fn.assert_called_once()
        approval_fn.assert_called_once()
        self.assertEqual(result.action_type, "approval_required")

    def test_calendar_write_reaches_approval_required_without_direct_create(self) -> None:
        decision = brain_agent.BrainDecision(
            mode="tool",
            tool="calendar_create",
            confidence=0.9,
            arguments={"title": "Team sync"},
            reply="",
            spoken_reply="",
            reason="explicit create",
        )
        from datetime import datetime, timedelta

        start = datetime.now().astimezone() + timedelta(days=1)
        fake_request = SimpleNamespace(
            action="create_event", assumed_duration=False, start=start, end=start + timedelta(hours=1)
        )
        fake_proposal = {"title": "Team sync"}
        fake_approval = SimpleNamespace(id=2)
        with patch.object(brain_agent, "decide", return_value=decision), patch.object(
            message_dispatch, "is_agenda_request", return_value=False
        ), patch.object(
            message_dispatch, "parse_calendar_request", return_value=fake_request
        ), patch.object(
            message_dispatch, "check_free_busy", return_value=[]
        ), patch.object(
            message_dispatch, "calendar_event_proposal", return_value=fake_proposal
        ), patch.object(
            message_dispatch, "create_calendar_event_approval", return_value=fake_approval
        ) as approval_fn, patch.object(
            message_dispatch, "approval_public_dict", return_value={"id": 2}
        ):
            result = message_dispatch.handle_message_result(
                "Schedule a meeting tomorrow at 9 PM"
            )

        approval_fn.assert_called_once()
        self.assertEqual(result.action_type, "approval_required")


class LiveBugRecencyResolutionTests(unittest.TestCase):
    """End-to-end regression coverage for the live bug: a follow-up such as "explain it
    with a real life example" immediately after a RAG explanation must resolve "it" against
    the immediately preceding turn and answer directly, instead of asking the user to
    clarify because multiple recently-discussed topics look equally salient.

    Root cause: memory_service.build_memory_context() rendered every recent turn as an
    undifferentiated flat "RECENT CONVERSATION AND TOOL RESULTS" block with no signal for
    which turn was the immediately preceding one, and neither the memory-context text nor
    brain_agent.BRAIN_SYSTEM_INSTRUCTION told the LLM to prioritize that turn for anaphora
    resolution. The fix splits recent turns into an explicitly labeled "MOST RECENT TURN"
    section (the primary anaphora-resolution candidate) versus "EARLIER RECENT
    CONVERSATION"/"RELEVANT OLDER LOCAL MEMORY" (lower priority), and adds explicit
    reference-resolution-priority instructions in both the memory context and the brain
    system instruction.

    These tests drive real SQLite-backed memory (same setup pattern as
    BrainV2MemoryIsolationTests) with only the underlying LLM call mocked, so they prove
    both (a) the constructed prompt structurally marks recency, and (b) routing/mode
    outcomes for these follow-ups are sane and never spuriously invoke a Gmail/Calendar tool.
    """

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "memory.db"

        self.engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        self.session_patch = patch.object(memory_service, "SessionLocal", self.Session)
        self.session_patch.start()

        self.profile_patch = patch.object(
            memory_service,
            "load_user_profile",
            return_value=memory_service.UserProfile(
                profile_id="test",
                preferred_name="Parth",
                assistant_name="Bunnelby",
            ),
        )
        self.profile_patch.start()

    def tearDown(self) -> None:
        self.profile_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def _add_turn(self, user: str, assistant: str) -> None:
        with self.Session() as db:
            db.add(Message(role="user", content=user))
            db.add(Message(role="assistant", content=assistant))
            db.commit()

    def _decide_capturing_prompt(
        self, message: str, payload: dict
    ) -> tuple[brain_agent.BrainDecision, dict]:
        captured: dict = {}

        def _fake_generate(system_instruction: str, user_content: str):
            captured["system_instruction"] = system_instruction
            captured["user_content"] = user_content
            return _fake_llm_result(payload)

        with patch.object(
            brain_agent, "generate_fast_text", side_effect=_fake_generate
        ), patch.object(brain_agent, "generate_text", side_effect=_fake_generate):
            decision = brain_agent.decide(message)
        return decision, captured

    def test_pronoun_it_resolves_to_immediately_preceding_rag_turn(self) -> None:
        self._add_turn(
            "What is RAG in simple terms?",
            "RAG, or Retrieval-Augmented Generation, retrieves relevant documents from a "
            "vector database before generating an answer, so the model grounds its reply in "
            "real information instead of relying only on what it memorized during training.",
        )
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.92,
            "arguments": {},
            "reply": "For example, a support chatbot uses RAG to pull your order history "
            "from a knowledge base before answering a question about your refund.",
            "spoken_reply": "For example, a support chatbot uses RAG to look up your order "
            "before answering, sir.",
            "reason": "resolved 'it' to the immediately preceding RAG topic",
        }

        decision, captured = self._decide_capturing_prompt(
            "explain it with a real life example", payload
        )

        user_content = captured["user_content"]
        self.assertIn("MOST RECENT TURN", user_content)
        self.assertIn("Reference resolution priority", user_content)
        # The RAG turn must appear inside the MOST RECENT TURN section, not merely
        # somewhere undifferentiated in the prompt.
        most_recent_idx = user_content.index("MOST RECENT TURN")
        rag_idx = user_content.index("What is RAG in simple terms?")
        self.assertGreater(rag_idx, most_recent_idx)

        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_pronoun_it_resolves_to_immediately_preceding_vector_db_turn(self) -> None:
        self._add_turn(
            "What is a vector database?",
            "A vector database stores numeric embeddings of data and lets you search by "
            "semantic similarity rather than exact keyword match.",
        )
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "It's useful in RAG because it lets the system quickly retrieve the "
            "most semantically relevant chunks of text to ground the generated answer.",
            "spoken_reply": "It helps RAG quickly find the most relevant text to ground its "
            "answer, sir.",
            "reason": "resolved 'it' to the immediately preceding vector database topic",
        }

        decision, captured = self._decide_capturing_prompt(
            "Why is it useful in RAG?", payload
        )

        user_content = captured["user_content"]
        self.assertIn("MOST RECENT TURN", user_content)
        most_recent_idx = user_content.index("MOST RECENT TURN")
        vector_db_idx = user_content.index("What is a vector database?")
        self.assertGreater(vector_db_idx, most_recent_idx)

        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_that_resolves_to_immediately_preceding_fastapi_turn(self) -> None:
        self._add_turn(
            "What is FastAPI?",
            "FastAPI is a modern Python web framework for building APIs quickly, built on "
            "top of Starlette and Pydantic with native async support.",
        )
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "Compared with Flask, FastAPI has native async support and automatic "
            "request validation, while Flask is simpler but requires extensions for those.",
            "spoken_reply": "FastAPI adds native async and automatic validation compared "
            "with Flask, sir.",
            "reason": "resolved 'that' to the immediately preceding FastAPI topic",
        }

        decision, captured = self._decide_capturing_prompt(
            "Compare that with Flask.", payload
        )

        user_content = captured["user_content"]
        self.assertIn("MOST RECENT TURN", user_content)
        most_recent_idx = user_content.index("MOST RECENT TURN")
        fastapi_idx = user_content.index("What is FastAPI?")
        self.assertGreater(fastapi_idx, most_recent_idx)

        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_genuinely_multi_topic_preceding_turn_may_still_clarify_without_crashing(
        self,
    ) -> None:
        # This is the intentional non-regression case: when the immediately preceding turn
        # itself covered two distinct topics, clarification remains an acceptable outcome
        # (per design) -- the assertion here only guards against a crash or a spurious
        # tool invocation, not against clarification itself.
        self._add_turn(
            "Explain RAG and vector databases.",
            "RAG combines retrieval with generation. A vector database is a separate "
            "storage system that supports semantic similarity search over embeddings.",
        )
        payload = {
            "mode": "clarify",
            "tool": None,
            "confidence": 0.4,
            "arguments": {},
            "reply": "Would you like more detail on RAG, on vector databases, or both?",
            "spoken_reply": "Do you want more on RAG, vector databases, or both, sir?",
            "reason": "immediately preceding turn covered two distinct topics",
        }

        decision, captured = self._decide_capturing_prompt("Explain it more.", payload)

        self.assertIn("MOST RECENT TURN", captured["user_content"])
        self.assertIn(decision.mode, ("clarify", "answer"))
        self.assertIsNone(decision.tool)

    def test_follow_up_never_invokes_gmail_or_calendar_tool_execution(self) -> None:
        self._add_turn(
            "What is RAG in simple terms?",
            "RAG, or Retrieval-Augmented Generation, retrieves relevant documents from a "
            "vector database before generating an answer.",
        )
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.92,
            "arguments": {},
            "reply": "For example, a support chatbot uses RAG to look up your order before "
            "answering a refund question.",
            "spoken_reply": "For example, a support chatbot uses RAG to look up your order, "
            "sir.",
            "reason": "resolved 'it' to the immediately preceding RAG topic",
        }

        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "gmail_handler"
        ) as gmail_handler, patch.object(
            tool_executor, "execute"
        ) as tool_execute, patch.object(
            message_dispatch, "_calendar_result"
        ) as calendar_result, patch.object(
            message_dispatch, "_calendar_agenda_result"
        ) as calendar_agenda_result, patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as cross_tool_fast_request:
            result = message_dispatch.handle_message_result(
                "explain it with a real life example"
            )

        gmail_handler.assert_not_called()
        tool_execute.assert_not_called()
        calendar_result.assert_not_called()
        calendar_agenda_result.assert_not_called()
        cross_tool_fast_request.assert_not_called()
        self.assertEqual(result.action_type, "general_answer")


class CollectionReferenceResolutionTests(unittest.TestCase):
    """Regression coverage for the live bug: after a tool call (or a plain-chat answer)
    returns a SET of multiple items in one turn -- several emails, several meetings, several
    enumerated options -- a follow-up like "which one" / "which is most important" / "the
    first one" must resolve against that already-returned set (mode="answer") instead of
    spuriously asking for clarification or re-invoking the same tool.

    Root cause (proven via a standalone reproduction against memory_service.build_memory_context
    before this fix): the immediately preceding Gmail/Calendar turn was being silently dropped
    from memory context entirely for phrasings such as "Which one looks most important and
    why?", because build_memory_context()'s tool-memory eligibility gate
    (_FOLLOW_UP_REFERENCE_PATTERN / _TEMPORAL_RECALL_PATTERN / _TOOL_MEMORY_CUE_PATTERN) did not
    recognize "which one" style phrasing as a follow-up reference, so allow_tool_memory was
    False and the gmail/calendar-routed MOST RECENT TURN was filtered out of eligible_turns
    before the MOST RECENT TURN section was even built -- the brain had nothing concrete to
    reason over. Separately, MAX_MESSAGE_CHARS (1600) uniformly clipped every turn including
    the most recent one, which could truncate a long itemized Gmail/Calendar list. The fix adds
    a domain-agnostic _COLLECTION_FOLLOW_UP_PATTERN so such phrasings count as a follow-up
    reference (keeping tool history eligible), widens the most-recent-turn clip budget via
    MOST_RECENT_TURN_MAX_CHARS, and adds a SET/COLLECTION REFERENCE RESOLUTION rule to
    BRAIN_SYSTEM_INSTRUCTION instructing mode="answer" (never "tool", never "clarify") for
    these follow-ups unless the user explicitly asks to refresh/re-check/re-fetch.

    These tests drive real SQLite-backed memory (same pattern as LiveBugRecencyResolutionTests
    above) with only the underlying LLM call mocked.
    """

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "memory.db"

        self.engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        self.session_patch = patch.object(memory_service, "SessionLocal", self.Session)
        self.session_patch.start()

        self.profile_patch = patch.object(
            memory_service,
            "load_user_profile",
            return_value=memory_service.UserProfile(
                profile_id="test",
                preferred_name="Parth",
                assistant_name="Bunnelby",
            ),
        )
        self.profile_patch.start()

    def tearDown(self) -> None:
        self.profile_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def _add_turn(self, user: str, assistant: str) -> None:
        with self.Session() as db:
            db.add(Message(role="user", content=user))
            db.add(Message(role="assistant", content=assistant))
            db.commit()

    def _decide_capturing_prompt(
        self, message: str, payload: dict
    ) -> tuple[brain_agent.BrainDecision, dict]:
        captured: dict = {}

        def _fake_generate(system_instruction: str, user_content: str):
            captured["system_instruction"] = system_instruction
            captured["user_content"] = user_content
            return _fake_llm_result(payload)

        with patch.object(
            brain_agent, "generate_fast_text", side_effect=_fake_generate
        ), patch.object(brain_agent, "generate_text", side_effect=_fake_generate):
            decision = brain_agent.decide(message)
        return decision, captured

    def _assert_no_tool_reexecution(self, decision: brain_agent.BrainDecision, message: str) -> None:
        """Dispatch `decision` through message_dispatch and prove no tool re-fetch happens."""
        with patch.object(brain_agent, "decide", return_value=decision), patch.object(
            tool_executor, "gmail_handler"
        ) as gmail_handler, patch.object(
            tool_executor, "execute"
        ) as tool_execute, patch.object(
            message_dispatch, "_calendar_result"
        ) as calendar_result, patch.object(
            message_dispatch, "_calendar_agenda_result"
        ) as calendar_agenda_result, patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as cross_tool_fast_request:
            result = message_dispatch.handle_message_result(message)

        gmail_handler.assert_not_called()
        tool_execute.assert_not_called()
        calendar_result.assert_not_called()
        calendar_agenda_result.assert_not_called()
        cross_tool_fast_request.assert_not_called()
        self.assertEqual(result.action_type, "general_answer")

    def test_gmail_collection_which_one_is_most_important_resolves_without_clarify(self) -> None:
        gmail_reply = (
            "Needs attention\n"
            "- Devfolio -- Your hackathon submission is under review\n"
            "- Emergent -- New feature announcement for your workspace\n"
            "- MENA AI/Data Community -- Weekly meetup digest\n"
            "- Meetup -- RSVP reminder for local AI meetup\n"
            "- Google -- Security alert on your account\n"
            "- Adobe -- Your Creative Cloud subscription renewal\n"
        )
        self._add_turn(
            "Now check my latest emails.",
            f"{gmail_reply}\nRoute: gmail\nWhy: brain_agent selected gmail_read",
        )

        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": (
                "The Google security alert looks most important -- account security issues "
                "should be checked first, ahead of the newsletters and RSVP reminders."
            ),
            "spoken_reply": "The Google security alert looks most important, sir.",
            "reason": "resolved against the single Gmail result set from the most recent turn",
        }

        decision, captured = self._decide_capturing_prompt(
            "Which one looks most important and why?", payload
        )

        user_content = captured["user_content"]
        self.assertIn("MOST RECENT TURN", user_content)
        self.assertIn("Devfolio", user_content)
        self.assertIn("Adobe", user_content)
        most_recent_idx = user_content.index("MOST RECENT TURN")
        devfolio_idx = user_content.index("Devfolio")
        self.assertGreater(devfolio_idx, most_recent_idx)

        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

        self._assert_no_tool_reexecution(decision, "Which one looks most important and why?")

    def test_gmail_collection_which_one_needs_attention_first_resolves_without_clarify(
        self,
    ) -> None:
        gmail_reply = (
            "Needs attention\n"
            "- Devfolio -- Your hackathon submission is under review\n"
            "- Emergent -- New feature announcement for your workspace\n"
            "- MENA AI/Data Community -- Weekly meetup digest\n"
            "- Meetup -- RSVP reminder for local AI meetup\n"
            "- Google -- Security alert on your account\n"
            "- Adobe -- Your Creative Cloud subscription renewal\n"
        )
        self._add_turn(
            "Now check my latest emails.",
            f"{gmail_reply}\nRoute: gmail\nWhy: brain_agent selected gmail_read",
        )

        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.88,
            "arguments": {},
            "reply": "The Google security alert needs your attention first.",
            "spoken_reply": "The Google security alert needs your attention first, sir.",
            "reason": "resolved against the single Gmail result set from the most recent turn",
        }

        decision, captured = self._decide_capturing_prompt(
            "Which one needs my attention first?", payload
        )

        self.assertIn("Devfolio", captured["user_content"])
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

        self._assert_no_tool_reexecution(decision, "Which one needs my attention first?")

    def test_calendar_collection_which_one_is_first_resolves_without_clarify(self) -> None:
        calendar_reply = (
            "You have 3 events today:\n"
            "- 9:00 AM - Standup with engineering\n"
            "- 1:00 PM - Client review call with Acme Corp\n"
            "- 5:30 PM - 1:1 with manager\n"
        )
        self._add_turn(
            "What's on my calendar today?",
            f"{calendar_reply}\nRoute: calendar\nWhy: verified Google Calendar agenda lookup",
        )

        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "The Standup with engineering at 9:00 AM is first.",
            "spoken_reply": "The 9 AM standup with engineering is first, sir.",
            "reason": "resolved against the single Calendar result set from the most recent turn",
        }

        decision, captured = self._decide_capturing_prompt("Which one is first?", payload)

        user_content = captured["user_content"]
        self.assertIn("Standup with engineering", user_content)
        self.assertIn("Client review call with Acme Corp", user_content)
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

        self._assert_no_tool_reexecution(decision, "Which one is first?")

    def test_generic_enumerated_options_which_one_would_you_choose_resolves(self) -> None:
        # This turn never involved a tool at all -- Bunnelby itself enumerated options in a
        # plain-conversation reply. Proves the fix generalizes beyond Gmail/Calendar.
        self._add_turn(
            "What laptop should I get for machine learning work?",
            "Here are three solid options: the Dell XPS 15 with an RTX 4060, the MacBook Pro "
            "14-inch with M3 Pro, or the Lenovo Legion 5 Pro with an RTX 4070.",
        )

        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.85,
            "arguments": {},
            "reply": (
                "I'd choose the Lenovo Legion 5 Pro with the RTX 4070 -- it gives you the "
                "most GPU headroom for local model training among the three options."
            ),
            "spoken_reply": "I'd go with the Lenovo Legion 5 Pro, sir.",
            "reason": "resolved against the enumerated laptop options from the most recent turn",
        }

        decision, captured = self._decide_capturing_prompt(
            "Which one would you choose?", payload
        )

        user_content = captured["user_content"]
        self.assertIn("Lenovo Legion 5 Pro", user_content)
        self.assertIn("MacBook Pro", user_content)
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

        self._assert_no_tool_reexecution(decision, "Which one would you choose?")


class LiveBugCrossToolGateRegressionTests(unittest.TestCase):
    """End-to-end regression coverage for the live bug: a conceptual Gmail+Calendar
    question must never execute a real Gmail or Calendar read.

    Root cause was intelligence_dispatch.py's pre-brain `is_cross_tool_request()` keyword
    gate, which ran BEFORE brain_agent.decide() and executed a real combined Gmail+Calendar
    read whenever both sets of keywords appeared, regardless of intent. These tests drive
    the real top-level entry point (intelligence_dispatch.handle_message_result, same as
    main.py's route) with only the underlying LLM call mocked -- following the same mocking
    pattern as BrainAgentDecisionTests above -- and assert no tool execution function is
    ever invoked for a conceptual/comparative question.
    """

    def setUp(self) -> None:
        self.memory_patch = patch.object(
            brain_agent, "build_memory_context", return_value="MEMORY"
        )
        self.memory_patch.start()

    def tearDown(self) -> None:
        self.memory_patch.stop()

    def _run_with_decision(self, message: str, payload: dict):
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "gmail_handler"
        ) as gmail_handler, patch.object(
            message_dispatch, "_calendar_result"
        ) as calendar_result, patch.object(
            message_dispatch, "_calendar_agenda_result"
        ) as calendar_agenda_result, patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as cross_tool_fast_request:
            result = intelligence_dispatch.handle_message_result(message)
        return result, {
            "gmail_handler": gmail_handler,
            "calendar_result": calendar_result,
            "calendar_agenda_result": calendar_agenda_result,
            "cross_tool_fast_request": cross_tool_fast_request,
        }

    def _assert_no_tool_executed(self, mocks: dict) -> None:
        mocks["gmail_handler"].assert_not_called()
        mocks["calendar_result"].assert_not_called()
        mocks["calendar_agenda_result"].assert_not_called()
        mocks["cross_tool_fast_request"].assert_not_called()

    def test_explain_difference_between_gmail_and_calendar_is_answer_with_no_tools(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.95,
            "arguments": {},
            "reply": (
                "Gmail is Google's email service for sending and receiving messages. "
                "Google Calendar is a separate scheduling service for events and reminders."
            ),
            "spoken_reply": "Gmail handles email, sir, while Calendar handles your schedule.",
            "reason": "conceptual comparison question",
        }
        result, mocks = self._run_with_decision(
            "Explain the difference between Gmail and Google Calendar.", payload
        )
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")
        self.assertNotIn("I checked your inbox", result.reply)

    def test_compare_email_and_calendar_systems_is_general_answer(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "Email systems store messages; calendar systems store scheduled events.",
            "spoken_reply": "Email stores messages, and calendars store scheduled events.",
            "reason": "conceptual comparison question",
        }
        result, mocks = self._run_with_decision("Compare email and calendar systems.", payload)
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")

    def test_can_gmail_and_calendar_integrate_is_general_answer(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "Yes, Gmail and Google Calendar can integrate, for example via event invites.",
            "spoken_reply": "Yes, they can integrate, sir, for example through event invites.",
            "reason": "conceptual integration question",
        }
        result, mocks = self._run_with_decision(
            "Can Gmail and Calendar integrate with each other?", payload
        )
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")

    def test_i_use_gmail_and_calendar_every_day_is_conversational(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.85,
            "arguments": {},
            "reply": "Good to know, sir.",
            "spoken_reply": "Good to know, sir.",
            "reason": "casual personal statement",
        }
        result, mocks = self._run_with_decision("I use Gmail and Calendar every day.", payload)
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")

    def test_what_is_a_gmail_address_is_conversational(self) -> None:
        # Routing regression guard for the spoken-email recipient precedence fix: a
        # conceptual question about Gmail addresses must stay conversational and must
        # never invoke the gmail_compose/gmail_read execution path.
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.9,
            "arguments": {},
            "reply": "A Gmail address is the unique example@gmail.com identifier for your Gmail account.",
            "spoken_reply": "A Gmail address is your unique identifier for your Gmail account, sir.",
            "reason": "conceptual question about Gmail addresses",
        }
        result, mocks = self._run_with_decision("What is a Gmail address?", payload)
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")

    def test_gmail_emails_mention_calendar_meetings_is_conversational(self) -> None:
        payload = {
            "mode": "answer",
            "tool": None,
            "confidence": 0.85,
            "arguments": {},
            "reply": "Noted.",
            "spoken_reply": "Noted, sir.",
            "reason": "casual observation, keyword overlap only",
        }
        result, mocks = self._run_with_decision(
            "My Gmail emails mention calendar meetings.", payload
        )
        self._assert_no_tool_executed(mocks)
        self.assertEqual(result.action_type, "general_answer")

    def test_explicit_combined_read_request_invokes_cross_tool_read(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "cross_tool_read",
            "confidence": 0.95,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit combined Gmail and Calendar read",
        }
        fake_result = SimpleNamespace(
            reply="Combined answer.",
            spoken_reply="Combined spoken answer, sir.",
            plan=SimpleNamespace(source="local_fastpath"),
            steps=(
                SimpleNamespace(status="success"),
                SimpleNamespace(status="success"),
            ),
            timings_ms={"cross_tool_total_ms": 12.3},
        )
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "handle_cross_tool_fast_request", return_value=fake_result
        ) as cross_tool_fast_request, patch.object(
            tool_executor, "gmail_handler"
        ) as gmail_handler, patch.object(
            message_dispatch, "_calendar_result"
        ) as calendar_result:
            result = intelligence_dispatch.handle_message_result(
                "Check my latest emails and what's on my calendar tomorrow."
            )

        cross_tool_fast_request.assert_called_once()
        gmail_handler.assert_not_called()
        calendar_result.assert_not_called()
        self.assertEqual(result.action_type, "task_complete")
        self.assertEqual(result.reply, "Combined answer.")
        self.assertTrue(result.spoken_metadata["cross_tool"])

    def test_read_inbox_and_check_todays_calendar_invokes_cross_tool_read(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "cross_tool_read",
            "confidence": 0.9,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit combined Gmail and Calendar read",
        }
        fake_result = SimpleNamespace(
            reply="Combined answer.",
            spoken_reply="Combined spoken answer, sir.",
            plan=SimpleNamespace(source="local_fastpath"),
            steps=(SimpleNamespace(status="success"),),
            timings_ms={},
        )
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "handle_cross_tool_fast_request", return_value=fake_result
        ) as cross_tool_fast_request:
            result = intelligence_dispatch.handle_message_result(
                "Read my inbox and check today's calendar."
            )

        cross_tool_fast_request.assert_called_once()
        self.assertEqual(result.action_type, "task_complete")

    def test_check_my_latest_emails_still_routes_to_gmail_read(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "gmail_read",
            "confidence": 0.95,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit read",
        }
        fake_handler_result = SimpleNamespace(
            reply="You have two unread messages.",
            action_type_override=None,
            spoken_reply="You have two unread messages, sir.",
            spoken_metadata=None,
            approval=None,
        )
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "gmail_handler", return_value=fake_handler_result
        ) as gmail_handler, patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as cross_tool_fast_request:
            result = intelligence_dispatch.handle_message_result("Check my latest emails.")

        gmail_handler.assert_called_once()
        cross_tool_fast_request.assert_not_called()
        self.assertEqual(result.action_type, "gmail_summary")

    def test_whats_on_my_calendar_tomorrow_still_routes_to_calendar_read(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "calendar_read",
            "confidence": 0.95,
            "arguments": {},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit read",
        }
        fake_result = SimpleNamespace(
            reply="You have nothing scheduled tomorrow.",
            action_type="calendar_read",
            memory_content="",
            spoken_reply="Nothing scheduled tomorrow, sir.",
            spoken_metadata=None,
            approval=None,
        )
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            tool_executor, "is_agenda_request", return_value=False
        ), patch.object(
            message_dispatch, "_calendar_result", return_value=fake_result
        ) as calendar_result, patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as cross_tool_fast_request:
            result = intelligence_dispatch.handle_message_result(
                "What's on my calendar tomorrow?"
            )

        calendar_result.assert_called_once()
        cross_tool_fast_request.assert_not_called()
        self.assertEqual(result.action_type, "calendar_read")

    def test_send_email_write_stops_at_approval_required(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "gmail_compose",
            "confidence": 0.9,
            "arguments": {"recipient_hint": "test@example.com", "body_hint": "hello"},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit compose",
        }
        fake_draft = {
            "mode": "compose",
            "thread_id": "",
            "source_message_id": "",
            "source_rfc_message_id": "",
            "references": "",
            "to": "test@example.com",
            "recipient_display": "test@example.com",
            "subject": "Hello",
            "body": "Hello there.",
            "instruction": "Send test@example.com an email saying hello",
            "provider": "groq",
            "status": "draft",
        }
        fake_approval = SimpleNamespace(id=1)
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            message_dispatch, "draft_new_email_from_request", return_value=fake_draft
        ) as draft_fn, patch.object(
            message_dispatch, "create_gmail_compose_approval", return_value=fake_approval
        ) as approval_fn, patch.object(
            message_dispatch, "approval_public_dict", return_value={"id": 1}
        ):
            result = intelligence_dispatch.handle_message_result(
                "Send test@example.com an email saying hello."
            )

        draft_fn.assert_called_once()
        approval_fn.assert_called_once()
        self.assertEqual(result.action_type, "approval_required")

    def test_schedule_meeting_write_stops_at_approval_required(self) -> None:
        payload = {
            "mode": "tool",
            "tool": "calendar_create",
            "confidence": 0.9,
            "arguments": {"title": "Team sync"},
            "reply": "",
            "spoken_reply": "",
            "reason": "explicit create",
        }
        from datetime import datetime, timedelta

        start = datetime.now().astimezone() + timedelta(days=1)
        fake_request = SimpleNamespace(
            action="create_event", assumed_duration=False, start=start, end=start + timedelta(hours=1)
        )
        fake_proposal = {"title": "Team sync"}
        fake_approval = SimpleNamespace(id=2)
        with patch.object(
            brain_agent, "generate_fast_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            brain_agent, "generate_text", return_value=_fake_llm_result(payload)
        ), patch.object(
            message_dispatch, "is_agenda_request", return_value=False
        ), patch.object(
            message_dispatch, "parse_calendar_request", return_value=fake_request
        ), patch.object(
            message_dispatch, "check_free_busy", return_value=[]
        ), patch.object(
            message_dispatch, "calendar_event_proposal", return_value=fake_proposal
        ), patch.object(
            message_dispatch, "create_calendar_event_approval", return_value=fake_approval
        ) as approval_fn, patch.object(
            message_dispatch, "approval_public_dict", return_value={"id": 2}
        ):
            result = intelligence_dispatch.handle_message_result(
                "Schedule a meeting tomorrow at 9 PM."
            )

        approval_fn.assert_called_once()
        self.assertEqual(result.action_type, "approval_required")


if __name__ == "__main__":
    unittest.main()
