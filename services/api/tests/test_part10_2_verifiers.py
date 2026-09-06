from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import (
    audit_service,
    brain_agent,
    tool_execution,
    tool_executor,
    verification_service,
)
from services.api.app.database import Base
from services.api.app.models import VerificationEvidence
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.tool_requests import (
    CalendarReadRequest,
    CrossToolReadRequest,
    GmailReadRequest,
)

# NOTE: no test in this file performs a real Gmail send or Calendar create.
# Every external read-back is mocked; the verifiers are exercised against
# fixtures only.


def read_result(**kwargs) -> OrchestratorResult:
    base = {
        "reply": "Two emails.",
        "action_type": "gmail_summary",
        "memory_content": "Two emails.",
    }
    base.update(kwargs)
    return OrchestratorResult(**base)


class GmailReadVerifierTests(unittest.TestCase):
    def test_bounded_matching_read_is_verified(self) -> None:
        request = GmailReadRequest(raw_message="check my email", limit=10)
        result = read_result(spoken_metadata={"email_count": 2, "unread_only": False})
        verdict = verification_service.verify_gmail_read(request, result)
        self.assertEqual(verdict.verdict, "verified")
        self.assertTrue(verdict.verified)

    def test_more_messages_than_the_limit_fails(self) -> None:
        request = GmailReadRequest(raw_message="check my email", limit=5)
        result = read_result(spoken_metadata={"email_count": 50, "unread_only": False})
        self.assertEqual(
            verification_service.verify_gmail_read(request, result).verdict, "failed"
        )

    def test_read_kind_mismatch_fails(self) -> None:
        request = GmailReadRequest(raw_message="check unread", read_kind="unread")
        result = read_result(spoken_metadata={"email_count": 1, "unread_only": False})
        self.assertEqual(
            verification_service.verify_gmail_read(request, result).verdict, "failed"
        )

    def test_missing_count_is_uncertain_not_verified(self) -> None:
        request = GmailReadRequest(raw_message="check my email")
        result = read_result(spoken_metadata={})
        self.assertEqual(
            verification_service.verify_gmail_read(request, result).verdict, "uncertain"
        )

    def test_error_result_fails(self) -> None:
        request = GmailReadRequest(raw_message="check my email")
        result = read_result(action_type="error", spoken_metadata={"email_count": 0})
        self.assertEqual(
            verification_service.verify_gmail_read(request, result).verdict, "failed"
        )


class CalendarReadVerifierTests(unittest.TestCase):
    def test_read_class_result_is_verified(self) -> None:
        request = CalendarReadRequest(raw_message="what is on my calendar")
        result = OrchestratorResult(
            reply="Nothing scheduled.", action_type="calendar_read", memory_content="x"
        )
        self.assertEqual(
            verification_service.verify_calendar_read(request, result).verdict, "verified"
        )

    def test_a_read_that_produced_an_approval_fails(self) -> None:
        """The exact invariant that used to be violable before Phase G."""
        request = CalendarReadRequest(raw_message="am I free to book the gym at 5")
        result = OrchestratorResult(
            reply="Prepared event.",
            action_type="approval_required",
            memory_content="x",
            approval={"id": 1},
        )
        verdict = verification_service.verify_calendar_read(request, result)
        self.assertEqual(verdict.verdict, "failed")
        self.assertIn("approval", verdict.evidence_text)

    def test_clarification_is_uncertain(self) -> None:
        request = CalendarReadRequest(raw_message="am I free")
        result = OrchestratorResult(
            reply="Which day?", action_type="clarification_required", memory_content="x"
        )
        self.assertEqual(
            verification_service.verify_calendar_read(request, result).verdict, "uncertain"
        )


class CrossToolReadVerifierTests(unittest.TestCase):
    def test_all_sources_succeeding_is_verified(self) -> None:
        request = CrossToolReadRequest(raw_message="email and calendar")
        result = OrchestratorResult(
            reply="Combined.",
            action_type="task_complete",
            memory_content="x",
            spoken_metadata={"steps_total": 2, "steps_succeeded": 2, "steps_failed": 0},
        )
        self.assertEqual(
            verification_service.verify_cross_tool_read(request, result).verdict, "verified"
        )

    def test_partial_failure_is_uncertain(self) -> None:
        request = CrossToolReadRequest(raw_message="email and calendar")
        result = OrchestratorResult(
            reply="Partial.",
            action_type="task_complete",
            memory_content="x",
            spoken_metadata={"steps_total": 2, "steps_succeeded": 1, "steps_failed": 1},
        )
        verdict = verification_service.verify_cross_tool_read(request, result)
        self.assertEqual(verdict.verdict, "uncertain")
        self.assertIn("partial", verdict.evidence_text)

    def test_no_successful_step_is_uncertain(self) -> None:
        request = CrossToolReadRequest(raw_message="email and calendar")
        result = OrchestratorResult(
            reply="Nothing.", action_type="error", memory_content="x", spoken_metadata={}
        )
        self.assertEqual(
            verification_service.verify_cross_tool_read(request, result).verdict, "uncertain"
        )

    def test_an_approval_from_a_cross_tool_read_fails(self) -> None:
        request = CrossToolReadRequest(raw_message="email and calendar")
        result = OrchestratorResult(
            reply="x", action_type="approval_required", memory_content="x", approval={"id": 2}
        )
        self.assertEqual(
            verification_service.verify_cross_tool_read(request, result).verdict, "failed"
        )


class GmailSendVerifierTests(unittest.TestCase):
    """Read-back comparison. No message is ever actually sent here."""

    PAYLOAD = {
        "to": "Rahul <rahul@example.com>",
        "subject": "Invoice follow-up",
        "body": "Hi Rahul, checking on the invoice.",
    }
    COMPLETION = {"gmail_message_id": "msg-1", "gmail_thread_id": "thr-1"}

    def _headers(self, **overrides) -> dict[str, str]:
        base = {
            "to": "rahul@example.com",
            "subject": "Invoice follow-up",
            "message_id": "msg-1",
            "thread_id": "thr-1",
            "label_ids": "SENT",
            "snippet": "Hi Rahul",
        }
        base.update(overrides)
        return base

    def test_matching_read_back_is_verified(self) -> None:
        with patch.object(
            verification_service, "_gmail_sent_headers", return_value=self._headers()
        ):
            verdict = verification_service.verify_gmail_send(self.PAYLOAD, self.COMPLETION)
        self.assertEqual(verdict.verdict, "verified")
        self.assertEqual(verdict.expected["recipient"], "rahul@example.com")

    def test_recipient_mismatch_fails(self) -> None:
        with patch.object(
            verification_service,
            "_gmail_sent_headers",
            return_value=self._headers(to="attacker@evil.test"),
        ):
            verdict = verification_service.verify_gmail_send(self.PAYLOAD, self.COMPLETION)
        self.assertEqual(verdict.verdict, "failed")
        self.assertIn("recipient", verdict.evidence_text)

    def test_subject_mismatch_fails(self) -> None:
        with patch.object(
            verification_service,
            "_gmail_sent_headers",
            return_value=self._headers(subject="Something else entirely"),
        ):
            self.assertEqual(
                verification_service.verify_gmail_send(self.PAYLOAD, self.COMPLETION).verdict,
                "failed",
            )

    def test_missing_message_id_is_uncertain(self) -> None:
        verdict = verification_service.verify_gmail_send(self.PAYLOAD, {})
        self.assertEqual(verdict.verdict, "uncertain")

    def test_read_back_failure_is_uncertain_not_failed(self) -> None:
        """A provider read-back error must not be reported as a wrong send."""
        with patch.object(
            verification_service,
            "_gmail_sent_headers",
            side_effect=RuntimeError("network down"),
        ):
            verdict = verification_service.verify_gmail_send(self.PAYLOAD, self.COMPLETION)
        self.assertEqual(verdict.verdict, "uncertain")
        self.assertIn("unverified, not failed", verdict.evidence_text)

    def test_display_name_formatting_does_not_cause_a_false_mismatch(self) -> None:
        payload = dict(self.PAYLOAD, to="rahul@example.com")
        with patch.object(
            verification_service,
            "_gmail_sent_headers",
            return_value=self._headers(to='"Rahul K" <RAHUL@Example.com>'),
        ):
            self.assertEqual(
                verification_service.verify_gmail_send(payload, self.COMPLETION).verdict,
                "verified",
            )


class CalendarCreateVerifierTests(unittest.TestCase):
    """Read-back comparison. No event is ever actually created here."""

    PAYLOAD = {
        "title": "Project Review",
        "start": "2099-09-01T15:00:00+05:30",
        "end": "2099-09-01T16:00:00+05:30",
        "attendees": ["rahul@example.com"],
        "calendar_id": "primary",
    }
    COMPLETION = {"calendar_event_id": "evt-1"}

    def _event(self, **overrides) -> dict:
        base = {
            "id": "evt-1",
            "status": "confirmed",
            "summary": "Project Review",
            "start": {"dateTime": "2099-09-01T15:00:00+05:30"},
            "end": {"dateTime": "2099-09-01T16:00:00+05:30"},
            "attendees": [{"email": "rahul@example.com"}],
        }
        base.update(overrides)
        return base

    def test_matching_read_back_is_verified(self) -> None:
        with patch.object(verification_service, "_calendar_event", return_value=self._event()):
            verdict = verification_service.verify_calendar_create(self.PAYLOAD, self.COMPLETION)
        self.assertEqual(verdict.verdict, "verified")

    def test_title_mismatch_fails(self) -> None:
        with patch.object(
            verification_service, "_calendar_event", return_value=self._event(summary="Other")
        ):
            self.assertEqual(
                verification_service.verify_calendar_create(
                    self.PAYLOAD, self.COMPLETION
                ).verdict,
                "failed",
            )

    def test_time_mismatch_fails(self) -> None:
        with patch.object(
            verification_service,
            "_calendar_event",
            return_value=self._event(start={"dateTime": "2099-09-02T09:00:00+05:30"}),
        ):
            verdict = verification_service.verify_calendar_create(self.PAYLOAD, self.COMPLETION)
        self.assertEqual(verdict.verdict, "failed")
        self.assertIn("start", verdict.evidence_text)

    def test_attendee_mismatch_fails(self) -> None:
        with patch.object(
            verification_service,
            "_calendar_event",
            return_value=self._event(attendees=[{"email": "someone@else.test"}]),
        ):
            self.assertEqual(
                verification_service.verify_calendar_create(
                    self.PAYLOAD, self.COMPLETION
                ).verdict,
                "failed",
            )

    def test_a_cancelled_event_fails(self) -> None:
        with patch.object(
            verification_service, "_calendar_event", return_value=self._event(status="cancelled")
        ):
            self.assertEqual(
                verification_service.verify_calendar_create(
                    self.PAYLOAD, self.COMPLETION
                ).verdict,
                "failed",
            )

    def test_missing_event_id_is_uncertain(self) -> None:
        self.assertEqual(
            verification_service.verify_calendar_create(self.PAYLOAD, {}).verdict, "uncertain"
        )

    def test_read_back_failure_is_uncertain(self) -> None:
        with patch.object(
            verification_service, "_calendar_event", side_effect=RuntimeError("api down")
        ):
            self.assertEqual(
                verification_service.verify_calendar_create(
                    self.PAYLOAD, self.COMPLETION
                ).verdict,
                "uncertain",
            )


class ApprovedExecutionIntegrationTests(unittest.TestCase):
    """The post-approval hook, with the approval engine untouched."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/v.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.patch = patch.object(audit_service, "SessionLocal", self.Session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def _approval(self, task_type: str, payload: dict, completion: dict):
        return SimpleNamespace(
            id=11,
            task_type=task_type,
            payload_json=json.dumps(payload),
            execution_result=json.dumps(completion),
        )

    def test_a_sent_email_is_verified_and_evidence_is_recorded(self) -> None:
        approval = self._approval(
            "gmail_compose",
            {"to": "rahul@example.com", "subject": "Hi", "body": "b"},
            {"gmail_message_id": "msg-1"},
        )
        with patch.object(
            verification_service,
            "_gmail_sent_headers",
            return_value={"to": "rahul@example.com", "subject": "Hi", "message_id": "msg-1"},
        ):
            verdict = verification_service.verify_approved_execution(approval, "sent")
        self.assertEqual(verdict.verdict, "verified")
        with self.Session() as db:
            row = db.query(VerificationEvidence).one()
        self.assertEqual(row.verifier_name, "GmailSendVerifier")
        self.assertEqual(row.approval_id, 11)
        self.assertEqual(row.verdict, "verified")

    def test_a_created_event_is_verified(self) -> None:
        approval = self._approval(
            "calendar_event",
            {"title": "T", "start": "2099-01-01T10:00:00+00:00", "end": "2099-01-01T11:00:00+00:00"},
            {"calendar_event_id": "evt-9"},
        )
        with patch.object(
            verification_service,
            "_calendar_event",
            return_value={
                "id": "evt-9",
                "status": "confirmed",
                "summary": "T",
                "start": {"dateTime": "2099-01-01T10:00:00+00:00"},
                "end": {"dateTime": "2099-01-01T11:00:00+00:00"},
            },
        ):
            verdict = verification_service.verify_approved_execution(approval, "created")
        self.assertEqual(verdict.verdict, "verified")

    def test_outcomes_without_a_new_external_change_are_not_verified(self) -> None:
        approval = self._approval("gmail_compose", {}, {})
        for outcome in ("already_sent", "already_created", "failed", "unknown", "rejected"):
            self.assertIsNone(
                verification_service.verify_approved_execution(approval, outcome), outcome
            )

    def test_an_unknown_task_type_is_skipped_safely(self) -> None:
        approval = self._approval("some_future_type", {}, {})
        self.assertIsNone(verification_service.verify_approved_execution(approval, "sent"))

    def test_uncertainty_message_never_claims_success(self) -> None:
        self.assertIsNone(
            verification_service.uncertainty_message(
                verification_service.VerificationResult("V", "verified")
            )
        )
        uncertain = verification_service.uncertainty_message(
            verification_service.VerificationResult("V", "uncertain")
        )
        self.assertIn("could not verify", uncertain)
        failed = verification_service.uncertainty_message(
            verification_service.VerificationResult("V", "failed")
        )
        self.assertIn("did not match", failed)


class ReadVerifierWiringTests(unittest.TestCase):
    """Reads are verified inline and their evidence persisted."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{self.tmp.name}/v.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.patch = patch.object(audit_service, "SessionLocal", self.Session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def _decision(self, tool: str, **arguments):
        return brain_agent.BrainDecision(
            mode="tool", tool=tool, confidence=0.95, arguments=arguments, reason="t"
        )

    def test_a_gmail_read_records_verification_evidence(self) -> None:
        result = OrchestratorResult(
            reply="Two emails.",
            action_type="gmail_summary",
            memory_content="x",
            spoken_metadata={"email_count": 2, "unread_only": False},
        )
        with patch.object(tool_execution, "execute_gmail_read", return_value=result):
            tool_executor.execute(self._decision("gmail_read"), "check my email")
        with self.Session() as db:
            rows = db.query(VerificationEvidence).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].verifier_name, "GmailReadVerifier")
        self.assertEqual(rows[0].verdict, "verified")
        self.assertIsNotNone(rows[0].tool_run_id)

    def test_a_write_proposal_records_no_external_verification_yet(self) -> None:
        result = OrchestratorResult(
            reply="Drafted.",
            action_type="approval_required",
            memory_content="x",
            approval={"id": 3},
        )
        with patch.object(tool_execution, "execute_gmail_compose", return_value=result):
            tool_executor.execute(
                self._decision("gmail_compose", recipient_hint="rahul"), "email rahul"
            )
        with self.Session() as db:
            self.assertEqual(db.query(VerificationEvidence).count(), 0)

    def test_read_verifier_registry_covers_every_read_capability(self) -> None:
        from services.api.app.capability_registry import registry
        from services.api.app.risk_policy import RiskLevel

        reads = {
            name
            for name in registry().tool_names()
            if registry().get(name).risk_level is RiskLevel.L0_OBSERVE
        }
        self.assertEqual(reads, set(verification_service.READ_VERIFIERS))


if __name__ == "__main__":
    unittest.main()
