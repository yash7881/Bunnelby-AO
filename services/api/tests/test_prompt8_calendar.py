from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import (
    approval_service,
    brain_agent,
    calendar_service,
    message_dispatch,
)
from services.api.app.approval_service import (
    ApprovalConflictError,
    ApprovalPayloadError,
    approve_and_execute,
    create_calendar_event_approval,
    reject_approval,
)
from services.api.app.calendar_service import CalendarParseError
from services.api.app.database import Base
from services.api.app.models import Approval


ZONE = ZoneInfo("Asia/Kolkata")
PROPOSAL = {
    "title": "Project Review",
    "start": "2099-09-01T15:00:00+05:30",
    "end": "2099-09-01T16:00:00+05:30",
    "timezone": "Asia/Kolkata",
    "attendees": ["rahul@example.com"],
    "calendar_id": "primary",
    "duration_minutes": 60,
    "assumed_duration": False,
}


def _calendar_create_decision() -> brain_agent.BrainDecision:
    """Pin the Brain's decision for calendar-dispatch tests.

    These two tests assert what the CALENDAR path does with a create decision.
    Before Part 10.2 Phase G they reached brain_agent.decide() unmocked and so
    made live Gemini/Groq calls -- non-deterministic, rate-limit prone and a
    real quota cost on every suite run. Pinning the decision keeps each
    assertion exactly as written while removing the live dependency.
    """
    return brain_agent.BrainDecision(
        mode="tool",
        tool="calendar_create",
        confidence=0.95,
        arguments={"title": "Project Review"},
        reply="",
        spoken_reply="",
        reason="test fixture",
    )


class CalendarParsingTests(unittest.TestCase):
    def test_tomorrow_afternoon_is_deterministic(self) -> None:
        now = datetime(2026, 8, 30, 10, 0, tzinfo=ZONE)
        parsed = calendar_service.parse_calendar_request(
            "Am I free tomorrow afternoon?",
            now=now,
            timezone=ZONE,
        )
        self.assertEqual(parsed.action, "free_busy")
        self.assertEqual(parsed.start, datetime(2026, 8, 31, 12, 0, tzinfo=ZONE))
        self.assertEqual(parsed.end, datetime(2026, 8, 31, 17, 0, tzinfo=ZONE))

    def test_explicit_event_time_and_duration(self) -> None:
        now = datetime(2026, 8, 30, 10, 0, tzinfo=ZONE)
        parsed = calendar_service.parse_calendar_request(
            "Schedule Project Review tomorrow at 3 PM for 45 minutes with rahul@example.com",
            now=now,
            timezone=ZONE,
        )
        self.assertEqual(parsed.action, "create_event")
        self.assertEqual(parsed.title, "Project Review")
        self.assertEqual(parsed.duration_minutes, 45)
        self.assertEqual(parsed.start.hour, 15)
        self.assertEqual(parsed.end.hour, 15)
        self.assertEqual(parsed.end.minute, 45)
        self.assertEqual(parsed.attendees, ("rahul@example.com",))
        self.assertFalse(parsed.assumed_duration)

    def test_event_without_duration_uses_visible_one_hour_default(self) -> None:
        now = datetime(2026, 8, 30, 10, 0, tzinfo=ZONE)
        parsed = calendar_service.parse_calendar_request(
            "Schedule Project Review tomorrow at 3 PM",
            now=now,
            timezone=ZONE,
        )
        self.assertEqual(parsed.duration_minutes, 60)
        self.assertTrue(parsed.assumed_duration)
        self.assertEqual(parsed.end - parsed.start, timedelta(hours=1))

    def test_create_with_only_afternoon_fails_closed(self) -> None:
        now = datetime(2026, 8, 30, 10, 0, tzinfo=ZONE)
        with self.assertRaises(CalendarParseError):
            calendar_service.parse_calendar_request(
                "Schedule Project Review tomorrow afternoon",
                now=now,
                timezone=ZONE,
            )

    def test_missing_date_fails_closed(self) -> None:
        now = datetime(2026, 8, 30, 10, 0, tzinfo=ZONE)
        with self.assertRaises(CalendarParseError):
            calendar_service.parse_calendar_request(
                "Check my calendar",
                now=now,
                timezone=ZONE,
            )

    def test_calendar_oauth_requires_read_and_events_scopes(self) -> None:
        self.assertFalse(
            calendar_service._payload_has_required_scopes(
                {"scopes": [calendar_service.CALENDAR_READONLY_SCOPE]}
            )
        )
        self.assertTrue(
            calendar_service._payload_has_required_scopes(
                {
                    "scopes": [
                        calendar_service.CALENDAR_READONLY_SCOPE,
                        calendar_service.CALENDAR_EVENTS_SCOPE,
                    ]
                }
            )
        )

    def test_open_slot_calculation_subtracts_busy_periods(self) -> None:
        target = datetime(2099, 9, 1, 9, 0, tzinfo=ZONE).date()
        busy = [
            {
                "start": "2099-09-01T10:00:00+05:30",
                "end": "2099-09-01T11:00:00+05:30",
            }
        ]
        with (
            patch.object(calendar_service, "local_timezone", return_value=ZONE),
            patch.object(calendar_service, "check_free_busy", return_value=busy),
        ):
            slots = calendar_service.find_open_slots(target, 60, (9, 12))

        starts = [item["start"] for item in slots]
        self.assertIn("2099-09-01T09:00:00+05:30", starts)
        self.assertIn("2099-09-01T11:00:00+05:30", starts)
        self.assertNotIn("2099-09-01T10:00:00+05:30", starts)

    def test_free_busy_response_handles_fully_free_window(self) -> None:
        request = calendar_service.ParsedCalendarRequest(
            action="free_busy",
            start=datetime(2099, 9, 1, 12, 0, tzinfo=ZONE),
            end=datetime(2099, 9, 1, 17, 0, tzinfo=ZONE),
            duration_minutes=60,
            timezone="Asia/Kolkata",
            daypart="afternoon",
        )
        reply = calendar_service.format_free_busy_response(request, [])
        self.assertIn("You're free", reply)
        self.assertIn("12:00 PM", reply)


class CalendarApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "calendar-test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session_patch = patch.object(approval_service, "SessionLocal", self.Session)
        self.session_patch.start()

    def tearDown(self) -> None:
        self.session_patch.stop()
        self.engine.dispose()
        self.tempdir.cleanup()

    def _new_approval(self):
        return create_calendar_event_approval(PROPOSAL, spoken_language="en")

    def test_calendar_event_starts_pending_and_does_not_execute(self) -> None:
        approval = self._new_approval()
        self.assertEqual(approval.task_type, "calendar_event")
        self.assertEqual(approval.status, "pending")
        self.assertEqual(approval.execution_state, "not_started")

        with patch.object(approval_service, "create_event") as create:
            with self.assertRaises(ApprovalConflictError):
                approval_service.create_approved_calendar_event(approval.id)
        create.assert_not_called()

    def test_approve_creates_exactly_once(self) -> None:
        approval = self._new_approval()
        calls = []

        def fake_create(*args, **kwargs):
            calls.append((args, kwargs))
            return {"id": "calendar-event-1", "htmlLink": "https://calendar.example/event"}

        with patch.object(approval_service, "create_event", side_effect=fake_create):
            first = approve_and_execute(approval.id)
            second = approve_and_execute(approval.id)

        self.assertEqual(first.outcome, "created")
        self.assertEqual(second.outcome, "already_created")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "Project Review")
        self.assertEqual(calls[0][1]["calendar_id"], "primary")
        self.assertEqual(calls[0][1]["timezone_name"], "Asia/Kolkata")

    def test_rejection_permanently_blocks_calendar_creation(self) -> None:
        approval = self._new_approval()
        rejected = reject_approval(approval.id)
        self.assertEqual(rejected.outcome, "rejected")

        with patch.object(approval_service, "create_event") as create:
            with self.assertRaises(ApprovalConflictError):
                approve_and_execute(approval.id)
        create.assert_not_called()

    def test_calendar_preview_tampering_blocks_execution(self) -> None:
        approval = self._new_approval()
        with self.Session() as db:
            row = db.get(Approval, approval.id)
            row.status = "approved"
            row.preview_content = "Tampered calendar preview"
            db.commit()

        with patch.object(approval_service, "create_event") as create:
            with self.assertRaises(ApprovalPayloadError):
                approval_service.create_approved_calendar_event(approval.id)
        create.assert_not_called()

    def test_calendar_dispatch_creates_pending_approval_only(self) -> None:
        parsed = calendar_service.ParsedCalendarRequest(
            action="create_event",
            start=datetime(2099, 9, 1, 15, 0, tzinfo=ZONE),
            end=datetime(2099, 9, 1, 16, 0, tzinfo=ZONE),
            duration_minutes=60,
            title="Project Review",
            attendees=("rahul@example.com",),
            timezone="Asia/Kolkata",
        )
        with (
            patch.object(message_dispatch, "parse_calendar_request", return_value=parsed),
            patch.object(message_dispatch, "check_free_busy", return_value=[]),
            patch.object(message_dispatch, "create_calendar_event_approval", wraps=create_calendar_event_approval) as create_approval,
            patch.object(approval_service, "create_event") as external_create,
            patch.object(brain_agent, "decide", return_value=_calendar_create_decision()),
        ):
            result = message_dispatch.handle_message_result(
                "Schedule Project Review tomorrow at 3 PM with rahul@example.com"
            )

        self.assertEqual(create_approval.call_count, 1)
        external_create.assert_not_called()
        self.assertEqual(result.action_type, "approval_required")
        self.assertEqual(result.approval["task_type"], "calendar_event")
        self.assertEqual(result.approval["status"], "pending")

    def test_busy_create_request_never_creates_approval(self) -> None:
        parsed = calendar_service.ParsedCalendarRequest(
            action="create_event",
            start=datetime(2099, 9, 1, 15, 0, tzinfo=ZONE),
            end=datetime(2099, 9, 1, 16, 0, tzinfo=ZONE),
            duration_minutes=60,
            title="Project Review",
            timezone="Asia/Kolkata",
        )
        with (
            patch.object(message_dispatch, "parse_calendar_request", return_value=parsed),
            patch.object(message_dispatch, "check_free_busy", return_value=[{
                "start": parsed.start.isoformat(),
                "end": parsed.end.isoformat(),
            }]),
            patch.object(message_dispatch, "create_calendar_event_approval") as create_approval,
            patch.object(brain_agent, "decide", return_value=_calendar_create_decision()),
        ):
            result = message_dispatch.handle_message_result(
                "Schedule Project Review tomorrow at 3 PM"
            )

        create_approval.assert_not_called()
        self.assertIsNone(result.approval)
        self.assertIn("overlaps", result.reply)


if __name__ == "__main__":
    unittest.main()
