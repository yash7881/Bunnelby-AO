from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.app import approval_service, gmail_service, message_dispatch
from services.api.app.approval_service import (
    approve_and_execute,
    create_gmail_compose_approval,
    reject_approval,
)
from services.api.app.database import Base
from services.api.app.gmail_service import GmailDraftError


class GmailComposeApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "compose-test.db"
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

    def test_partial_email_clue_is_extracted(self) -> None:
        self.assertEqual(
            gmail_service._partial_email_clue(
                "Rahul ko email bhejo, uska email trj...@gmail.com type hai"
            ),
            "trj",
        )
        self.assertEqual(
            gmail_service._partial_email_clue(
                "Rahul ka email starts with trj"
            ),
            "trj",
        )

    def test_partial_email_match_has_priority_over_name_resolution(self) -> None:
        with patch.object(
            gmail_service,
            "_search_gmail_addresses_by_partial",
            return_value=["trj11114@gmail.com"],
        ):
            recipient = gmail_service.resolve_new_email_recipient(
                "Rahul ko email send karo. Uska email trj...@gmail.com type hai."
            )

        self.assertEqual(recipient, "trj11114@gmail.com")

    def test_named_hinglish_compose_is_detected_without_literal_address(self) -> None:
        command = "Rahul ko email send kro and likho ki me kal usse 9 PM milunga"
        self.assertTrue(message_dispatch._standalone_email_requested(command))

    def test_named_compose_uses_safe_recipient_resolver(self) -> None:
        fake = SimpleNamespace(
            text=json.dumps({
                "subject": "Meeting Tomorrow",
                "body": "Hi Rahul,\\n\\nI will meet you tomorrow at 9 PM.\\n\\nBest,\\nParth",
            }),
            provider="gemini",
            model="fixture",
        )

        with (
            patch.object(
                gmail_service,
                "resolve_new_email_recipient",
                return_value="rj-test@example.com",
            ) as resolver,
            patch.object(
                gmail_service,
                "generate_gemini_text",
                return_value=fake,
            ),
        ):
            draft = gmail_service.draft_new_email_from_request(
                "Rahul ko email send kro and likho ki me kal usse 9 PM milunga"
            )

        resolver.assert_called_once()
        self.assertEqual(draft["to"], "rj-test@example.com")
        self.assertEqual(draft["mode"], "compose")

    def test_hinglish_address_command_is_detected_as_new_email(self) -> None:
        command = (
            "trj11114@gmail.com es email pr mail kro ki mera automation project "
            "successful create ho gaya hai and apni taraf se long email create kro english me and send kro"
        )
        self.assertTrue(message_dispatch._standalone_email_requested(command))

    def test_compose_draft_keeps_explicit_recipient_and_uses_provider_output(self) -> None:
        fake = SimpleNamespace(
            text=json.dumps({
                "subject": "Automation Project Successfully Completed",
                "body": "Hello,\n\nI’m pleased to share that my automation project has been completed successfully.\n\nBest regards,\nParth",
            }),
            provider="groq",
            model="fixture",
        )
        with patch.object(gmail_service, "generate_gemini_text", return_value=fake):
            draft = gmail_service.draft_new_email_from_request(
                "Send a long professional email to trj11114@gmail.com saying my automation project was completed successfully."
            )

        self.assertEqual(draft["mode"], "compose")
        self.assertEqual(draft["to"], "trj11114@gmail.com")
        self.assertEqual(draft["provider"], "groq")
        self.assertIn("Automation Project", draft["subject"])
        self.assertIn("completed successfully", draft["body"])

    def test_compose_requires_exactly_one_explicit_recipient(self) -> None:
        with self.assertRaises(gmail_service.GmailDraftError):
            gmail_service.draft_new_email_from_request("Write and send a professional project update email.")

        with self.assertRaises(gmail_service.GmailDraftError):
            gmail_service.draft_new_email_from_request(
                "Send this to first@example.com and second@example.com."
            )

    def test_compose_approval_never_sends_before_approval_and_sends_once_after(self) -> None:
        draft = {
            "mode": "compose",
            "thread_id": "",
            "source_message_id": "",
            "source_rfc_message_id": "",
            "references": "",
            "to": "trj11114@gmail.com",
            "recipient_display": "trj11114@gmail.com",
            "subject": "Automation Project Successfully Completed",
            "body": "The automation project has been completed successfully.",
            "instruction": "Send an automation project completion update.",
            "provider": "groq",
            "status": "draft",
        }
        approval = create_gmail_compose_approval(draft, spoken_language="en")

        sent = []
        with patch.object(approval_service, "_send_reply_payload", side_effect=lambda payload: sent.append(dict(payload)) or {"id": "m1", "threadId": "t1"}):
            self.assertEqual(sent, [])
            first = approve_and_execute(approval.id)
            second = approve_and_execute(approval.id)

        self.assertEqual(first.outcome, "sent")
        self.assertEqual(second.outcome, "already_sent")
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["mode"], "compose")
        self.assertEqual(sent[0]["recipient"], "trj11114@gmail.com")

    def test_rejected_compose_can_never_send(self) -> None:
        draft = {
            "mode": "compose",
            "thread_id": "",
            "source_message_id": "",
            "source_rfc_message_id": "",
            "references": "",
            "to": "trj11114@gmail.com",
            "recipient_display": "trj11114@gmail.com",
            "subject": "Project update",
            "body": "A safe draft.",
            "instruction": "Draft a project update.",
            "provider": "groq",
            "status": "draft",
        }
        approval = create_gmail_compose_approval(draft, spoken_language="en")
        result = reject_approval(approval.id)
        self.assertEqual(result.outcome, "rejected")

        with patch.object(approval_service, "_send_reply_payload") as send:
            with self.assertRaises(approval_service.ApprovalConflictError):
                approve_and_execute(approval.id)
        send.assert_not_called()


class SpokenEmailNormalizationTests(unittest.TestCase):
    """Unit coverage for gmail_service.normalize_spoken_email(): the live production bug
    was that "trj11114 at gmail.com" (STT renders "at" as a word but "dot com" as a
    literal ".com") never normalized to an "@" address, so it fell through to Gmail
    correspondent/name lookup instead of being recognized as an explicit address.
    """

    def test_already_valid_address_passes_through_unchanged(self) -> None:
        self.assertEqual(
            gmail_service.normalize_spoken_email("Send an email to trj11114@gmail.com saying hello"),
            "Send an email to trj11114@gmail.com saying hello",
        )

    def test_spoken_at_with_literal_dot_domain_normalizes(self) -> None:
        self.assertIn(
            "trj11114@gmail.com",
            gmail_service.normalize_spoken_email("Send an email to trj11114 at gmail.com Say hello"),
        )

    def test_spoken_at_dot_com_normalizes(self) -> None:
        self.assertIn(
            "trj11114@gmail.com",
            gmail_service.normalize_spoken_email("Send an email to trj11114 at gmail dot com saying hello"),
        )

    def test_spoken_at_the_rate_dot_com_normalizes(self) -> None:
        self.assertIn(
            "trj11114@gmail.com",
            gmail_service.normalize_spoken_email(
                "Send an email to trj11114 at the rate gmail dot com saying hello"
            ),
        )

    def test_all_caps_spoken_form_normalizes_and_lowercases(self) -> None:
        normalized = gmail_service.normalize_spoken_email(
            "Send an email to TRJ11114 AT GMAIL DOT COM saying hello"
        )
        self.assertIn("trj11114@gmail.com", normalized)
        self.assertNotIn("TRJ11114@GMAIL.COM", normalized)

    def test_trailing_speech_punctuation_is_tolerated(self) -> None:
        self.assertIn(
            "trj11114@gmail.com",
            gmail_service.normalize_spoken_email(
                "Send an email to trj11114 at gmail dot com. Say hello"
            ),
        )

    def test_different_local_part_is_never_confused_with_another(self) -> None:
        normalized = gmail_service.normalize_spoken_email(
            "Send an email to PRJ11114 at gmail.com saying hello"
        )
        self.assertIn("prj11114@gmail.com", normalized)
        self.assertNotIn("trj11114@gmail.com", normalized)

    def test_malformed_glued_domain_is_not_silently_rewritten(self) -> None:
        # "theredgmail.com" / "therategmail.com" are not recognized common providers;
        # the normalizer must not guess these mean gmail.com, and must leave the text
        # unresolved rather than fabricate a domain.
        for message in (
            "Send an email to trj11114 at theredgmail.com saying hello",
            "Send an email to prj11114 at therategmail.com saying hello",
        ):
            normalized = gmail_service.normalize_spoken_email(message)
            self.assertNotIn("@gmail.com", normalized)
            self.assertEqual(normalized, message)


class GmailRecipientPrecedenceTests(unittest.TestCase):
    """Covers the deterministic recipient-resolution precedence: an explicit or
    normalizable email address must be used directly and must never reach Gmail
    correspondent/name lookup; only a genuine bare name/hint should reach it.
    """

    def _assert_never_calls_gmail(self, user_message: str) -> str:
        with patch.object(gmail_service, "_gmail_service") as gmail_client:
            recipient = gmail_service.resolve_new_email_recipient(user_message)
        gmail_client.assert_not_called()
        return recipient

    def test_spoken_at_literal_dot_domain_resolves_without_correspondent_lookup(self) -> None:
        recipient = self._assert_never_calls_gmail(
            "Send an email to trj11114 at gmail.com Say hello"
        )
        self.assertEqual(recipient, "trj11114@gmail.com")

    def test_spoken_at_dot_com_resolves_without_correspondent_lookup(self) -> None:
        recipient = self._assert_never_calls_gmail(
            "Send an email to trj11114 at gmail dot com saying hello"
        )
        self.assertEqual(recipient, "trj11114@gmail.com")

    def test_spoken_at_the_rate_dot_com_resolves_without_correspondent_lookup(self) -> None:
        recipient = self._assert_never_calls_gmail(
            "Send an email to trj11114 at the rate gmail dot com saying hello"
        )
        self.assertEqual(recipient, "trj11114@gmail.com")

    def test_already_explicit_address_resolves_without_correspondent_lookup(self) -> None:
        recipient = self._assert_never_calls_gmail(
            "Send an email to trj11114@gmail.com saying hello"
        )
        self.assertEqual(recipient, "trj11114@gmail.com")

    def test_different_local_part_never_becomes_another_address(self) -> None:
        recipient = self._assert_never_calls_gmail(
            "Send an email to PRJ11114 at gmail.com saying hello"
        )
        self.assertEqual(recipient, "prj11114@gmail.com")

    def test_malformed_domain_fails_closed_to_clarification_not_correspondent_lookup(self) -> None:
        with patch.object(gmail_service, "_gmail_service") as gmail_client:
            with self.assertRaises(GmailDraftError) as ctx:
                gmail_service.resolve_new_email_recipient(
                    "Send an email to trj11114 at theredgmail.com saying hello"
                )
        gmail_client.assert_not_called()
        self.assertIn("exact email address", str(ctx.exception))

    def test_bare_name_still_reaches_correspondent_lookup(self) -> None:
        # A genuine bare name/hint (no "@", no spoken " at <domain>" marker) must still
        # be resolved via the existing Gmail correspondent search, unchanged by the
        # explicit-address precedence fix.
        with patch.object(gmail_service, "_gmail_service") as gmail_client:
            with self.assertRaises(GmailDraftError):
                gmail_service.resolve_new_email_recipient("Send Rahul an email saying hello")
        gmail_client.assert_called()


class GmailComposePrecedenceIntegrationTests(unittest.TestCase):
    """End-to-end coverage through message_dispatch._gmail_compose_result(), proving the
    fixed recipient precedence reaches action_type=="approval_required" for an explicit
    or normalizable address, and that a real send is never reachable from this path.
    """

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "compose-precedence-test.db"
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

    def _fake_draft_llm(self) -> SimpleNamespace:
        return SimpleNamespace(
            text=json.dumps({"subject": "Hello", "body": "Hello there."}),
            provider="groq",
            model="fixture",
        )

    def test_spoken_email_address_reaches_approval_required_without_correspondent_lookup_or_send(
        self,
    ) -> None:
        with (
            patch.object(gmail_service, "generate_gemini_text", return_value=self._fake_draft_llm()),
            patch.object(gmail_service, "_gmail_service") as gmail_client,
            patch.object(approval_service, "_send_reply_payload") as send_payload,
        ):
            result = message_dispatch._gmail_compose_result(
                "Send an email to trj11114 at gmail.com Say hello"
            )

        gmail_client.assert_not_called()
        send_payload.assert_not_called()
        self.assertEqual(result.action_type, "approval_required")
        self.assertIsNotNone(result.approval)

    def test_malformed_spoken_address_is_clarification_not_approval_and_never_sends(self) -> None:
        with (
            patch.object(gmail_service, "generate_gemini_text", return_value=self._fake_draft_llm()),
            patch.object(gmail_service, "_gmail_service") as gmail_client,
            patch.object(approval_service, "_send_reply_payload") as send_payload,
        ):
            result = message_dispatch._gmail_compose_result(
                "Send an email to trj11114 at theredgmail.com saying hello"
            )

        gmail_client.assert_not_called()
        send_payload.assert_not_called()
        self.assertEqual(result.action_type, "clarification_required")
        self.assertIsNone(result.approval)


if __name__ == "__main__":
    unittest.main()
