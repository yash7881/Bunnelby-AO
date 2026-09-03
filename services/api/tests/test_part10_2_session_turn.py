from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import (
    brain_agent,
    intelligence_dispatch,
    main,
    memory_service,
    message_dispatch,
    session_service,
)
from services.api.app.database import Base
from services.api.app.models import Message
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.schemas import ChatRequest
from services.api.app.session_service import (
    LEGACY_SESSION_ID,
    TurnContext,
    is_valid_identifier,
    new_result_set_id,
    new_session_id,
    new_turn_id,
    resolve_session_id,
    resolve_turn_context,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class SessionIdentityTests(unittest.TestCase):
    def test_generated_ids_are_prefixed_and_unique(self) -> None:
        sessions = {new_session_id() for _ in range(200)}
        turns = {new_turn_id() for _ in range(200)}
        self.assertEqual(len(sessions), 200)
        self.assertEqual(len(turns), 200)
        self.assertTrue(all(value.startswith("sess-") for value in sessions))
        self.assertTrue(all(value.startswith("turn-") for value in turns))

    def test_generated_ids_pass_their_own_validator(self) -> None:
        self.assertTrue(is_valid_identifier(new_session_id()))
        self.assertTrue(is_valid_identifier(new_turn_id()))
        self.assertTrue(is_valid_identifier(new_result_set_id("gmail read")))

    def test_result_set_ids_carry_a_readable_source_slug(self) -> None:
        self.assertTrue(new_result_set_id("gmail_read").startswith("rs-gmail-read-"))
        self.assertTrue(new_result_set_id("Calendar Agenda").startswith("rs-calendar-agenda-"))
        self.assertTrue(new_result_set_id("!!!").startswith("rs-unknown-"))

    def test_validator_rejects_unsafe_or_oversized_values(self) -> None:
        for bad in (None, 42, "", "   ", "a" * 129, "has space", "semi;colon", "-leading", "../x"):
            self.assertFalse(is_valid_identifier(bad), repr(bad))

    def test_resolve_session_id_accepts_a_valid_caller_value(self) -> None:
        self.assertEqual(resolve_session_id("sess-abc123"), "sess-abc123")
        self.assertEqual(resolve_session_id("  sess-abc123  "), "sess-abc123")

    def test_resolve_session_id_mints_one_instead_of_failing(self) -> None:
        # Backward compatibility: a pre-10.2 caller must not crash the turn.
        for omitted in (None, "", "   ", "bad value", 7):
            resolved = resolve_session_id(omitted)
            self.assertTrue(resolved.startswith("sess-"), repr(omitted))

    def test_resolve_turn_context_returns_both_ids(self) -> None:
        context = resolve_turn_context()
        self.assertIsInstance(context, TurnContext)
        self.assertTrue(context.session_id.startswith("sess-"))
        self.assertTrue(context.turn_id.startswith("turn-"))

    def test_legacy_sentinel_matches_the_migration(self) -> None:
        migration = (
            REPO_ROOT
            / "database/migrations/versions/0004_add_message_session_turn.py"
        ).read_text(encoding="utf-8")
        self.assertIn(f'LEGACY_SESSION_ID = "{LEGACY_SESSION_ID}"', migration)


class MessageSchemaTests(unittest.TestCase):
    def test_message_model_carries_session_and_turn(self) -> None:
        columns = {column.name for column in Message.__table__.columns}
        self.assertIn("session_id", columns)
        self.assertIn("turn_id", columns)

    def test_both_columns_are_nullable_for_a_metadata_only_migration(self) -> None:
        self.assertTrue(Message.__table__.c.session_id.nullable)
        self.assertTrue(Message.__table__.c.turn_id.nullable)

    def test_chat_request_session_id_is_optional(self) -> None:
        self.assertIsNone(ChatRequest(message="hi").session_id)
        self.assertEqual(
            ChatRequest(message="hi", session_id="sess-x1").session_id, "sess-x1"
        )


MOST_RECENT_TURN_HEADER = "\nMOST RECENT TURN (the immediately preceding exchange"


class SessionScopedMemoryTests(unittest.TestCase):
    """The core Phase D guarantee: a new session inherits nothing."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/memory.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session_patch = patch.object(memory_service, "SessionLocal", self.Session)
        self.session_patch.start()

    def tearDown(self) -> None:
        self.session_patch.stop()
        self.engine.dispose()
        self.tmp.cleanup()

    def _turn(self, session_id: str, turn_id: str, user: str, assistant: str) -> None:
        with self.Session() as db:
            db.add(Message(role="user", content=user, session_id=session_id, turn_id=turn_id))
            db.add(
                Message(
                    role="assistant",
                    content=assistant,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            )
            db.commit()

    def test_previous_session_is_not_the_current_active_topic(self) -> None:
        self._turn("sess-old", "turn-old", "Explain Qdrant", "Qdrant is a vector database.")
        self._turn("sess-new", "turn-new", "Explain FastAPI", "FastAPI is a Python framework.")

        context = memory_service.build_memory_context("explain it more", session_id="sess-new")

        self.assertIn("FastAPI", context)
        self.assertNotIn("Qdrant", context)
        most_recent = context.split(MOST_RECENT_TURN_HEADER, 1)[1]
        self.assertIn("FastAPI", most_recent)

    def test_a_brand_new_session_sees_no_prior_conversation_at_all(self) -> None:
        self._turn("sess-old", "turn-old", "Explain Qdrant", "Qdrant is a vector database.")

        context = memory_service.build_memory_context("hello", session_id=new_session_id())

        self.assertNotIn("Qdrant", context)
        # No turn section is emitted at all for an empty session. (The static
        # memory-rules prose mentions the phrase, so match the real header.)
        self.assertNotIn(MOST_RECENT_TURN_HEADER, context)

    def test_same_session_still_keeps_continuity(self) -> None:
        self._turn("sess-a", "turn-1", "Explain RAG", "RAG retrieves then generates.")
        self._turn("sess-a", "turn-2", "give an example", "For example, a support bot.")

        context = memory_service.build_memory_context("explain it more", session_id="sess-a")

        self.assertIn("RAG", context)
        self.assertIn("support bot", context)

    def test_legacy_rows_are_reachable_only_through_their_own_session(self) -> None:
        self._turn(LEGACY_SESSION_ID, "legacy-turn-1", "old question", "old answer")

        scoped = memory_service.build_memory_context("anything", session_id="sess-fresh")
        self.assertNotIn("old answer", scoped)

        legacy = memory_service.build_memory_context("anything", session_id=LEGACY_SESSION_ID)
        self.assertIn("old answer", legacy)

    def test_omitting_session_id_preserves_the_pre_10_2_global_scan(self) -> None:
        self._turn("sess-old", "turn-old", "Explain Qdrant", "Qdrant is a vector database.")
        context = memory_service.build_memory_context("explain it more")
        self.assertIn("Qdrant", context)

    def test_tool_history_from_another_session_cannot_leak(self) -> None:
        with self.Session() as db:
            db.add(
                Message(
                    role="user",
                    content="check my latest emails",
                    session_id="sess-old",
                    turn_id="turn-old",
                )
            )
            db.add(
                Message(
                    role="assistant",
                    content="Email from Rahul about invoices.\nRoute: gmail\nWhy: read",
                    session_id="sess-old",
                    turn_id="turn-old",
                )
            )
            db.commit()

        context = memory_service.build_memory_context(
            "which one is most important", session_id="sess-new"
        )
        self.assertNotIn("Rahul", context)


class SessionThreadingTests(unittest.TestCase):
    """session_id must survive every hop from /chat down to memory."""

    def test_brain_forwards_session_id_to_memory(self) -> None:
        seen: dict[str, object] = {}

        def fake_context(message: str, session_id: str | None = None) -> str:
            seen["session_id"] = session_id
            return "CONTEXT"

        payload = json.dumps(
            {"mode": "answer", "tool": None, "confidence": 0.9, "reply": "ok", "spoken_reply": "ok"}
        )
        result = type("R", (), {"text": payload, "provider": "gemini", "model": "m"})()

        with patch.object(brain_agent, "build_memory_context", fake_context), patch.object(
            brain_agent, "generate_fast_text", return_value=result
        ), patch.object(brain_agent, "generate_text", return_value=result):
            brain_agent.decide("what is a vector database", session_id="sess-threaded")

        self.assertEqual(seen["session_id"], "sess-threaded")

    def test_message_dispatch_forwards_session_id(self) -> None:
        seen: dict[str, object] = {}

        def fake_decide(message: str, session_id: str | None = None):
            seen["session_id"] = session_id
            return brain_agent.BrainDecision(
                mode="answer", tool=None, confidence=1.0, reply="ok", spoken_reply="ok"
            )

        with patch.object(brain_agent, "decide", fake_decide):
            message_dispatch.handle_message_result("hi there", session_id="sess-md")
        self.assertEqual(seen["session_id"], "sess-md")

    def test_intelligence_dispatch_forwards_session_id(self) -> None:
        seen: dict[str, object] = {}

        def fake_dispatch(message: str, session_id: str | None = None, turn_id=None):
            seen["session_id"] = session_id
            seen["turn_id"] = turn_id
            return OrchestratorResult(
                reply="ok", action_type="general_answer", memory_content="ok"
            )

        with patch.object(intelligence_dispatch, "_legacy_dispatch", fake_dispatch):
            intelligence_dispatch.handle_message_result("hi", session_id="sess-id")
        self.assertEqual(seen["session_id"], "sess-id")

    def test_every_hop_stays_backward_compatible_without_a_session(self) -> None:
        with patch.object(
            brain_agent,
            "decide",
            return_value=brain_agent.BrainDecision(
                mode="answer", tool=None, confidence=1.0, reply="ok", spoken_reply="ok"
            ),
        ):
            self.assertEqual(
                message_dispatch.handle_message_result("hi").action_type, "general_answer"
            )
            self.assertEqual(
                intelligence_dispatch.handle_message_result("hi").action_type,
                "general_answer",
            )


class ChatEndpointSessionTests(unittest.TestCase):
    """/chat must persist both rows of a turn under one session and one turn id."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/chat.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # Never let an endpoint test touch the real ao.db.
        self.db_patch = patch.object(main, "SessionLocal", self.Session)
        self.db_patch.start()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.engine.dispose()
        self.tmp.cleanup()

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(main.app)

    def _result(self) -> OrchestratorResult:
        return OrchestratorResult(
            reply="answer", action_type="general_answer", memory_content="answer"
        )

    def test_response_echoes_a_minted_session_and_turn(self) -> None:
        with patch.object(main, "handle_message_result", return_value=self._result()):
            body = self._client().post("/chat", json={"message": "hello"}).json()
        self.assertTrue(body["session_id"].startswith("sess-"))
        self.assertTrue(body["turn_id"].startswith("turn-"))

    def test_caller_supplied_session_is_reused_across_turns(self) -> None:
        with patch.object(main, "handle_message_result", return_value=self._result()):
            client = self._client()
            first = client.post("/chat", json={"message": "one"}).json()
            second = client.post(
                "/chat", json={"message": "two", "session_id": first["session_id"]}
            ).json()
        self.assertEqual(second["session_id"], first["session_id"])
        self.assertNotEqual(second["turn_id"], first["turn_id"])

    def test_both_rows_of_a_turn_share_session_and_turn_ids(self) -> None:
        with patch.object(main, "handle_message_result", return_value=self._result()):
            body = self._client().post("/chat", json={"message": "hello"}).json()

        with self.Session() as db:
            rows = db.query(Message).order_by(Message.id).all()
            self.assertEqual(len(rows), 2)
            self.assertEqual({row.session_id for row in rows}, {body["session_id"]})
            self.assertEqual({row.turn_id for row in rows}, {body["turn_id"]})
            self.assertEqual([row.role for row in rows], ["user", "assistant"])

    def test_dispatch_receives_the_resolved_session_id(self) -> None:
        with patch.object(
            main, "handle_message_result", return_value=self._result()
        ) as mocked:
            body = self._client().post("/chat", json={"message": "hello"}).json()
        self.assertEqual(mocked.call_args.kwargs["session_id"], body["session_id"])

    def test_omitted_session_id_is_still_accepted(self) -> None:
        with patch.object(main, "handle_message_result", return_value=self._result()):
            response = self._client().post("/chat", json={"message": "hello"})
        self.assertEqual(response.status_code, 200)

    def test_two_omitted_session_requests_are_isolated_from_each_other(self) -> None:
        with patch.object(main, "handle_message_result", return_value=self._result()):
            client = self._client()
            first = client.post("/chat", json={"message": "one"}).json()
            second = client.post("/chat", json={"message": "two"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])


class VoiceRuntimeSessionTests(unittest.TestCase):
    """One wake -> follow-up conversation shares one session id."""

    def test_dispatch_to_chat_sends_session_id_when_supplied(self) -> None:
        runtime = pathlib.Path(
            REPO_ROOT / "scripts/wakeword/wake_conversation_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn('body_payload["session_id"] = session_id', runtime)
        self.assertIn("session_id=conversation_session_id", runtime)
        self.assertIn("conversation_session_id = new_session_id()", runtime)

    def test_new_session_is_minted_at_wake_not_per_turn(self) -> None:
        runtime = pathlib.Path(
            REPO_ROOT / "scripts/wakeword/wake_conversation_runtime.py"
        ).read_text(encoding="utf-8")
        # Exactly one mint site, and it sits in the wake-detected branch.
        self.assertEqual(runtime.count("conversation_session_id = new_session_id()"), 1)
        wake_block = runtime.split("controller.begin_listening())", 1)[1][:400]
        self.assertIn("conversation_session_id = new_session_id()", wake_block)


class DesktopSessionWiringTests(unittest.TestCase):
    def test_app_jsx_sends_a_stable_session_id(self) -> None:
        app_jsx = (REPO_ROOT / "apps/desktop/src/App.jsx").read_text(encoding="utf-8")
        self.assertIn("function createSessionId()", app_jsx)
        self.assertIn("const sessionIdRef = useRef(createSessionId());", app_jsx)
        self.assertIn("session_id: sessionIdRef.current", app_jsx)


if __name__ == "__main__":
    unittest.main()
