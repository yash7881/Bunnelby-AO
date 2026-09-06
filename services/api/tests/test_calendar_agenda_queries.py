from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from services.api.app import brain_agent, calendar_agenda_service, message_dispatch, tool_executor


ZONE = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 8, 30, 16, 30, tzinfo=ZONE)


class CalendarAgendaQueryTests(unittest.TestCase):
    def test_hinglish_aaj_schedule_routes_to_agenda(self) -> None:
        self.assertTrue(calendar_agenda_service.is_agenda_request("mera aaj ka schedule kya hai"))
        start, end, zone = calendar_agenda_service.agenda_range(
            "mera aaj ka schedule kya hai",
            now=NOW,
            timezone=ZONE,
        )
        self.assertEqual(start.date(), NOW.date())
        self.assertEqual(end.date().isoformat(), "2026-08-31")
        self.assertEqual(str(zone), "Asia/Kolkata")

    def test_english_today_timetable_routes_to_agenda(self) -> None:
        self.assertTrue(calendar_agenda_service.is_agenda_request("today my time table?"))
        self.assertTrue(message_dispatch._calendar_requested("today my time table?"))

    def test_agenda_response_uses_verified_calendar_event_titles_and_times(self) -> None:
        start = datetime(2026, 8, 30, 0, 0, tzinfo=ZONE)
        events = [
            {
                "id": "one",
                "title": "Project Review",
                "all_day": False,
                "start": "2026-08-30T15:00:00+05:30",
                "end": "2026-08-30T15:30:00+05:30",
                "location": "",
            },
            {
                "id": "two",
                "title": "Deep Work",
                "all_day": False,
                "start": "2026-08-30T18:00:00+05:30",
                "end": "2026-08-30T19:00:00+05:30",
                "location": "",
            },
        ]
        reply = calendar_agenda_service.format_agenda_response(start, events)
        self.assertIn("Project Review", reply)
        self.assertIn("3:00 PM–3:30 PM", reply)
        self.assertIn("Deep Work", reply)
        self.assertIn("6:00 PM–7:00 PM", reply)

    def test_empty_agenda_claim_is_calendar_specific(self) -> None:
        start = datetime(2026, 8, 30, 0, 0, tzinfo=ZONE)
        reply = calendar_agenda_service.format_agenda_response(start, [])
        self.assertIn("Google Calendar", reply)
        self.assertNotIn("email", reply.casefold())

    def test_dispatcher_does_not_delegate_today_timetable_to_general_chat(self) -> None:
        start = datetime(2026, 8, 30, 0, 0, tzinfo=ZONE)
        end = datetime(2026, 8, 31, 0, 0, tzinfo=ZONE)
        events = [
            {
                "id": "event-1",
                "title": "Calendar Source of Truth",
                "all_day": False,
                "start": "2026-08-30T17:00:00+05:30",
                "end": "2026-08-30T18:00:00+05:30",
                "location": "",
            }
        ]
        decision = brain_agent.BrainDecision(
            mode="tool",
            tool="calendar_read",
            confidence=0.95,
            reply="",
            spoken_reply="",
            reason="explicit agenda read",
        )
        with (
            patch.object(brain_agent, "decide", return_value=decision),
            patch.object(message_dispatch, "agenda_range", return_value=(start, end, ZONE)),
            patch.object(message_dispatch, "list_events", return_value=events),
        ):
            result = message_dispatch.handle_message_result("today my time table?")

        self.assertEqual(result.action_type, "calendar_read")
        self.assertIn("Calendar Source of Truth", result.reply)
        self.assertNotIn("email", result.reply.casefold())


if __name__ == "__main__":
    unittest.main()
