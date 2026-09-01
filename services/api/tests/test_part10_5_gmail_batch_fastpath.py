from __future__ import annotations

import unittest
from unittest.mock import patch

from services.api.app import gmail_fast_read


class _FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


class _FakeBatch:
    def __init__(self):
        self.entries = []
        self.execute_calls = 0

    def add(self, request, *, callback, request_id):
        self.entries.append((request_id, request, callback))

    def execute(self):
        self.execute_calls += 1
        for request_id, request, callback in self.entries:
            try:
                response = request.execute()
                callback(request_id, response, None)
            except BaseException as exc:
                callback(request_id, None, exc)


class _FakeMessages:
    def __init__(self, list_payload, message_payloads):
        self.list_payload = list_payload
        self.message_payloads = message_payloads
        self.last_list_kwargs = None

    def list(self, **kwargs):
        self.last_list_kwargs = kwargs
        return _FakeRequest(self.list_payload)

    def get(self, **kwargs):
        return _FakeRequest(self.message_payloads[kwargs["id"]])


class _FakeUsers:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages


class _FakeService:
    def __init__(self, list_payload, message_payloads):
        self.messages_resource = _FakeMessages(list_payload, message_payloads)
        self.batch = _FakeBatch()

    def users(self):
        return _FakeUsers(self.messages_resource)

    def new_batch_http_request(self):
        return self.batch


def _payload(subject: str, internal_date: str = "1788292800000"):
    return {
        "threadId": "thread-1",
        "internalDate": internal_date,
        "snippet": f"Snippet for {subject}",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Tue, 1 Sep 2026 18:00:00 +0000"},
            ]
        },
    }


class GmailBatchFastPathTests(unittest.TestCase):
    def test_batch_metadata_preserves_message_order_and_fields(self) -> None:
        service = _FakeService(
            {"messages": [{"id": "m1"}, {"id": "m2"}]},
            {"m1": _payload("First"), "m2": _payload("Second")},
        )
        with patch.object(gmail_fast_read, "_gmail_service", return_value=service):
            emails = gmail_fast_read._fetch_fast_messages(max_results=10, unread_only=False)

        self.assertEqual([item["subject"] for item in emails], ["First", "Second"])
        self.assertEqual(emails[0]["sender"], "sender@example.com")
        self.assertEqual(service.batch.execute_calls, 1)
        self.assertEqual(service.messages_resource.last_list_kwargs["labelIds"], ["INBOX"])

    def test_failed_batch_subrequest_retries_through_proven_sequential_helper(self) -> None:
        service = _FakeService(
            {"messages": [{"id": "m1"}, {"id": "m2"}]},
            {"m1": _payload("First"), "m2": RuntimeError("batch-only failure")},
        )
        sequential = {
            "id": "m2",
            "thread_id": "thread-2",
            "sender": "fallback@example.com",
            "subject": "Recovered",
            "snippet": "Recovered snippet",
            "timestamp": "2026-09-01T18:00:00+00:00",
        }
        with (
            patch.object(gmail_fast_read, "_gmail_service", return_value=service),
            patch.object(gmail_fast_read, "_message_to_email", return_value=sequential) as fallback,
        ):
            emails = gmail_fast_read._fetch_fast_messages(max_results=10, unread_only=False)

        self.assertEqual([item["subject"] for item in emails], ["First", "Recovered"])
        fallback.assert_called_once_with(service, "m2")

    def test_unread_executor_keeps_existing_contract_and_caps_visible_items(self) -> None:
        refs = [{"id": f"m{i}"} for i in range(12)]
        payloads = {f"m{i}": _payload(f"Subject {i}") for i in range(12)}
        service = _FakeService({"messages": refs}, payloads)

        with patch.object(gmail_fast_read, "_gmail_service", return_value=service):
            result = gmail_fast_read.gmail_read_fast_executor(
                "read my latest unread emails",
                {},
            )

        self.assertTrue(result["unread_only"])
        self.assertFalse(result["today_only"])
        self.assertEqual(result["count"], 10)
        self.assertEqual(len(result["emails"]), 10)
        self.assertEqual(
            service.messages_resource.last_list_kwargs["labelIds"],
            ["INBOX", "UNREAD"],
        )


if __name__ == "__main__":
    unittest.main()
