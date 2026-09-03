from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from services.api.app import (
    brain_agent,
    calendar_service,
    gmail_service,
    message_dispatch,
    tool_executor,
)
from services.api.app.tool_requests import (
    CalendarCreateRequest,
    CalendarReadRequest,
    CrossToolReadRequest,
    GmailComposeRequest,
    GmailReadRequest,
    GmailReplyRequest,
    ToolRequestValidationError,
    build_request,
)

# Part 10.2 Phase G regression suite.
#
# The audit proved that four of six Brain tool values collapsed into two
# function calls with identical arguments (tool_executor lines 81/92 and
# 106..112), and that every executor re-derived its own action class from the
# raw user text. These tests pin the invariant that closes that hole:
#
#   THE BRAIN DECIDES THE ACTION CLASS ONCE.
#
# A parser may fill missing fields inside an already-selected class. It may
# never turn a read into a write, or a write into a read.

FUTURE = datetime.now().astimezone() + timedelta(days=1)


def decision(tool: str, **arguments: object) -> brain_agent.BrainDecision:
    return brain_agent.BrainDecision(
        mode="tool",
        tool=tool,
        confidence=0.95,
        arguments=arguments,
        reply="",
        spoken_reply="",
        reason="test",
    )


class RequestTypeFixesTheActionClassTests(unittest.TestCase):
    """Structural half: a request type cannot express another class's action."""

    def test_calendar_read_request_has_no_create_mode(self) -> None:
        for mode in ("agenda", "free_busy", "open_slots"):
            self.assertIsInstance(
                CalendarReadRequest(raw_message="x", mode=mode), CalendarReadRequest
            )
        with self.assertRaises(Exception):
            CalendarReadRequest(raw_message="x", mode="create_event")

    def test_calendar_read_request_rejects_event_fields(self) -> None:
        # extra="forbid": a title cannot be smuggled into a read.
        with self.assertRaises(Exception):
            CalendarReadRequest(raw_message="x", title="Team sync")

    def test_calendar_create_request_requires_a_title(self) -> None:
        with self.assertRaises(Exception):
            CalendarCreateRequest(raw_message="x")

    def test_gmail_read_request_cannot_carry_a_recipient_or_body(self) -> None:
        with self.assertRaises(Exception):
            GmailReadRequest(raw_message="x", recipient="a@b.com")
        with self.assertRaises(Exception):
            GmailReadRequest(raw_message="x", body="hello")

    def test_gmail_compose_request_requires_a_recipient_hint(self) -> None:
        with self.assertRaises(Exception):
            GmailComposeRequest(raw_message="x")

    def test_cross_tool_read_needs_two_distinct_sources(self) -> None:
        with self.assertRaises(Exception):
            CrossToolReadRequest(raw_message="x", sources=("gmail",))
        with self.assertRaises(Exception):
            CrossToolReadRequest(raw_message="x", sources=("gmail", "gmail"))
        self.assertEqual(
            CrossToolReadRequest(raw_message="x", sources=("gmail", "calendar")).sources,
            ("gmail", "calendar"),
        )

    def test_build_request_drops_unknown_brain_arguments(self) -> None:
        request = build_request(
            "gmail_read",
            "check my unread mail",
            {"read_kind": "unread", "hallucinated_field": "boom"},
        )
        self.assertIsInstance(request, GmailReadRequest)
        self.assertTrue(request.unread_only)

    def test_invalid_optional_value_is_dropped_for_the_schema_default(self) -> None:
        # A live Gemini run supplied freshness="latest". Refusing the whole turn
        # over a defaulted, non-targeting field is a worse failure than ignoring
        # it, and the fallback (limit=10) is strictly safer than 9999.
        request = build_request("gmail_read", "check my email", {"limit": 9999})
        self.assertEqual(request.limit, 10)
        request = build_request("gmail_read", "check my email", {"freshness": "latest"})
        self.assertEqual(request.freshness, "fresh_required")

    def test_invalid_required_value_still_fails_closed(self) -> None:
        # For every write capability the required fields are exactly the
        # safety-critical ones, so these must never be silently dropped.
        for tool, arguments in (
            ("gmail_compose", {}),
            ("gmail_compose", {"recipient_hint": ""}),
            ("calendar_create", {}),
            ("calendar_create", {"title": ""}),
        ):
            with self.subTest(tool=tool, arguments=arguments):
                with self.assertRaises(ToolRequestValidationError):
                    build_request(tool, "do it", arguments)

    def test_an_optional_write_field_that_is_too_long_is_dropped_not_executed_blindly(self) -> None:
        request = build_request(
            "gmail_compose", "email rahul", {"recipient_hint": "rahul", "subject": "x" * 5000}
        )
        self.assertIsNone(request.subject)
        self.assertEqual(request.recipient_hint, "rahul")

    def test_every_request_type_reports_its_own_tool_name(self) -> None:
        cases = {
            "gmail_read": {},
            "gmail_compose": {"recipient_hint": "rahul"},
            "gmail_reply": {},
            "calendar_read": {},
            "calendar_create": {"title": "Sync"},
            "cross_tool_read": {},
            "general_answer": {},
        }
        for name, arguments in cases.items():
            self.assertEqual(build_request(name, "msg", arguments).tool_name, name)


class ExecutorHonoursTheBrainClassTests(unittest.TestCase):
    """Behavioural half: the executed action class matches the Brain's choice."""

    def _no_calendar_writes(self):
        """Any attempt to build a Calendar write proposal fails the test."""

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError(
                "a calendar_read decision must never build a create approval"
            )

        return patch.multiple(
            message_dispatch,
            calendar_event_proposal=forbidden,
            create_calendar_event_approval=forbidden,
        )

    def _no_gmail_sends(self):
        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("a read decision must never build a Gmail approval")

        return patch.multiple(
            message_dispatch,
            draft_new_email_from_request=forbidden,
            create_gmail_compose_approval=forbidden,
        )

    # ---------------- calendar_read must stay a read ---------------- #

    def test_calendar_read_with_create_vocabulary_never_mints_an_approval(self) -> None:
        # "book" is in calendar_service._is_create_request's verb set, so before
        # Phase G this availability question produced a Calendar write approval.
        messages = (
            "am I free to book the gym tomorrow at 5 pm?",
            "can I schedule anything tomorrow at 3 pm or am I busy?",
            "do I have time to book a call tomorrow at 4 pm?",
        )
        for message in messages:
            with self.subTest(message=message):
                with self._no_calendar_writes(), patch.object(
                    message_dispatch, "check_free_busy", return_value=[]
                ), patch.object(
                    message_dispatch, "list_events", return_value=[]
                ):
                    result = tool_executor.execute(decision("calendar_read"), message)
                self.assertIsNone(result.approval, message)
                self.assertNotEqual(result.action_type, "approval_required", message)

    def test_calendar_read_returns_a_read_action_type(self) -> None:
        with patch.object(message_dispatch, "list_events", return_value=[]):
            result = tool_executor.execute(
                decision("calendar_read"), "what's on my calendar tomorrow?"
            )
        self.assertIn(result.action_type, {"calendar_read", "error", "clarification_required"})
        self.assertIsNone(result.approval)

    # ---------------- calendar_create must stay a create ---------------- #

    def test_calendar_create_without_create_vocabulary_is_not_downgraded(self) -> None:
        # "put X on my calendar" contains none of the create verbs, so before
        # Phase G this create decision silently returned free/busy availability.
        captured: dict[str, object] = {}

        def fake_proposal(request: object) -> dict[str, object]:
            captured["action"] = getattr(request, "action", None)
            return {"title": "Dentist"}

        with patch.object(
            message_dispatch, "calendar_event_proposal", fake_proposal
        ), patch.object(
            message_dispatch,
            "create_calendar_event_approval",
            return_value=_FakeApproval(),
        ), patch.object(
            message_dispatch, "check_free_busy", return_value=[]
        ), patch.object(
            message_dispatch, "approval_public_dict", return_value={"id": 1}
        ):
            result = tool_executor.execute(
                decision("calendar_create", title="Dentist"),
                "put the dentist appointment on my calendar tomorrow at 4 pm",
            )

        self.assertIn(
            result.action_type,
            {"approval_required", "clarification_required", "error"},
            "a create decision must never answer with a free/busy read",
        )
        self.assertNotEqual(result.action_type, "calendar_read")

    def test_calendar_create_missing_details_clarifies_rather_than_reading(self) -> None:
        with self._no_calendar_writes():
            result = tool_executor.execute(
                decision("calendar_create", title="Sync"),
                "schedule a sync sometime next week",
            )
        self.assertNotEqual(result.action_type, "calendar_read")
        self.assertIsNone(result.approval)

    # ---------------- gmail_read must stay a read ---------------- #

    def test_gmail_read_with_reply_vocabulary_never_builds_an_approval(self) -> None:
        messages = (
            "did anyone reply to my email today?",
            "check whether Rahul replied to my message",
        )
        for message in messages:
            with self.subTest(message=message):
                with self._no_gmail_sends(), patch.object(
                    gmail_service, "get_recent_emails", return_value=[]
                ), patch.object(gmail_service, "get_unread_emails", return_value=[]):
                    result = tool_executor.execute(decision("gmail_read"), message)
                self.assertIsNone(result.approval, message)
                self.assertNotEqual(result.action_type, "approval_required", message)

    def test_gmail_read_unread_intent_survives_into_the_typed_request(self) -> None:
        with patch.object(
            gmail_service, "get_unread_emails", return_value=[]
        ) as unread, patch.object(gmail_service, "get_recent_emails") as recent:
            tool_executor.execute(
                decision("gmail_read", read_kind="unread"), "check my unread emails"
            )
        self.assertEqual(unread.call_count, 1)
        recent.assert_not_called()

    # ---------------- gmail_reply must stay a reply ---------------- #

    def test_gmail_reply_never_returns_an_inbox_summary(self) -> None:
        def forbidden(*_a: object, **_k: object) -> object:
            raise AssertionError("a gmail_reply decision must not read the inbox")

        with patch.multiple(
            gmail_service, get_recent_emails=forbidden, get_unread_emails=forbidden
        ), patch.object(
            gmail_service,
            "draft_reply_from_request",
            side_effect=gmail_service.GmailTargetResolutionError("need a clearer sender"),
        ):
            result = tool_executor.execute(
                decision("gmail_reply"), "reply to that email saying thanks"
            )
        self.assertNotEqual(result.action_type, "gmail_summary")

    # ---------------- cross_tool_read must stay read-only ---------------- #

    def test_cross_tool_read_never_produces_an_approval(self) -> None:
        with self._no_gmail_sends(), self._no_calendar_writes(), patch.object(
            tool_executor, "handle_cross_tool_fast_request"
        ) as handler:
            handler.side_effect = RuntimeError("forced failure")
            result = tool_executor.execute(
                decision("cross_tool_read"),
                "check my latest emails and what's on my calendar tomorrow",
            )
        self.assertIsNone(result.approval)


class _FakeApproval:
    id = 1
    task_type = "calendar_event"


class ReadRequestCannotReachAWriteCapabilityTests(unittest.TestCase):
    def test_registry_refuses_a_mismatched_request_type(self) -> None:
        """A read object routed at a write capability must be refused outright.

        This is the last line of defence: even if something managed to aim a
        read request at calendar_create, the registry's type guard stops it
        before any approval builder runs.
        """
        from services.api.app import capability_registry

        registry = capability_registry.registry()
        self.assertTrue(registry.has("calendar_create"))

        class SpoofedRequest(CalendarReadRequest):
            """A read request lying about which capability it belongs to."""

            @property
            def tool_name(self) -> str:
                return "calendar_create"

        spoofed = SpoofedRequest(raw_message="what is on my calendar")
        self.assertFalse(isinstance(spoofed, registry.get("calendar_create").request_model))

        with self.assertRaises(capability_registry.CapabilityRegistryError):
            registry.execute(spoofed)

    def test_each_capability_declares_the_matching_request_model(self) -> None:
        from services.api.app import capability_registry
        from services.api.app.tool_requests import request_model_for

        registry = capability_registry.registry()
        for name in registry.names():
            self.assertIs(
                registry.get(name).request_model,
                request_model_for(name),
                f"{name} must execute only its own request type",
            )

    def test_write_requests_are_classified_as_writes(self) -> None:
        from services.api.app.tool_requests import is_write_request

        self.assertTrue(
            is_write_request(GmailComposeRequest(raw_message="x", recipient_hint="r"))
        )
        self.assertTrue(is_write_request(GmailReplyRequest(raw_message="x")))
        self.assertTrue(
            is_write_request(CalendarCreateRequest(raw_message="x", title="t"))
        )
        self.assertFalse(is_write_request(GmailReadRequest(raw_message="x")))
        self.assertFalse(is_write_request(CalendarReadRequest(raw_message="x")))
        self.assertFalse(
            is_write_request(CrossToolReadRequest(raw_message="x"))
        )


if __name__ == "__main__":
    unittest.main()
