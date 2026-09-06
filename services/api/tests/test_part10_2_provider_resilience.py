from __future__ import annotations

import contextlib
import os
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.api.app import main, model_gateway


FAKE_GEMINI_KEY = "AIzaFAKEKEYFORTESTSONLY0123456789"
FAKE_GROQ_KEY = "gsk_FAKEKEYFORTESTSONLY0123456789AB"


@contextlib.contextmanager
def resilience_env(**overrides: str):
    baseline = {
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "GEMINI_MODEL": "gemini-3.6-flash",
        "GEMINI_FAST_MODEL": "gemini-3.5-flash-lite",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "GEMINI_COOLDOWN_SECONDS": "900",
        "GROQ_COOLDOWN_SECONDS": "120",
        "LLM_MAX_TRANSIENT_RETRIES": "1",
        "LLM_TRANSIENT_RETRY_DELAY_MS": "1",
        "LLM_TRANSIENT_COOLDOWN_SECONDS": "30",
    }
    baseline.update(overrides)
    with patch.dict(os.environ, baseline, clear=False):
        yield


def provider_error(
    code: str,
    *,
    cooldown: int,
    recoverable: bool = True,
    message: str = "temporary provider failure",
):
    return model_gateway._ProviderCallError(
        error_code=code,
        message=message,
        recoverable=recoverable,
        cooldown_seconds=cooldown,
    )


class ProviderResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        model_gateway.REGISTRY.reset()
        self.addCleanup(model_gateway.REGISTRY.reset)
        model_gateway.register_local_provider(model_gateway.UnconfiguredLocalProvider())
        self.addCleanup(
            model_gateway.register_local_provider,
            model_gateway.UnconfiguredLocalProvider(),
        )

    def test_transport_failure_gets_one_bounded_retry(self) -> None:
        calls = 0

        def invoke():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise provider_error("transport", cooldown=30)
            return "ok"

        with resilience_env():
            with patch.object(model_gateway.time, "sleep") as sleep:
                result = model_gateway._call_with_transient_retry(
                    "gemini", "gemini-3.6-flash", invoke
                )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)
        sleep.assert_called_once()
        events = model_gateway.REGISTRY.recent_events()
        self.assertEqual(events[-1]["event"], "retry")
        self.assertEqual(events[-1]["error_code"], "transport")

    def test_429_never_retries_same_provider(self) -> None:
        calls = 0

        def invoke():
            nonlocal calls
            calls += 1
            raise provider_error("http_429", cooldown=120)

        with resilience_env():
            with patch.object(model_gateway.time, "sleep") as sleep:
                with self.assertRaises(model_gateway._ProviderCallError):
                    model_gateway._call_with_transient_retry(
                        "gemini", "gemini-3.6-flash", invoke
                    )

        self.assertEqual(calls, 1)
        sleep.assert_not_called()
        self.assertFalse(model_gateway.REGISTRY.recent_events())

    def test_transient_failure_uses_short_breaker(self) -> None:
        with resilience_env(LLM_TRANSIENT_COOLDOWN_SECONDS="23"):
            self.assertEqual(
                model_gateway._cooldown_for_failure(
                    "gemini", "transport", retry_after_seconds=None
                ),
                23,
            )
            self.assertEqual(
                model_gateway._cooldown_for_failure(
                    "groq", "http_503", retry_after_seconds=None
                ),
                23,
            )

    def test_429_without_retry_after_is_bounded_even_with_old_env(self) -> None:
        with resilience_env(GEMINI_COOLDOWN_SECONDS="900"):
            self.assertEqual(
                model_gateway._cooldown_for_failure(
                    "gemini", "http_429", retry_after_seconds=None
                ),
                model_gateway.DEFAULT_RATE_LIMIT_FALLBACK_SECONDS,
            )

    def test_retry_after_is_respected_and_capped(self) -> None:
        self.assertEqual(
            model_gateway._retry_after_seconds_from_headers({"Retry-After": "7"}),
            7,
        )
        self.assertEqual(
            model_gateway._retry_after_seconds_from_headers(
                {"Retry-After": "999999"}
            ),
            model_gateway.MAX_RETRY_AFTER_SECONDS,
        )

    def test_expired_breaker_reports_half_open_until_next_result(self) -> None:
        with resilience_env():
            model_gateway.REGISTRY.record_failure(
                "gemini",
                "gemini-3.6-flash",
                error_code="transport",
                detail="temporary",
                cooldown_seconds=30,
            )
            state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
            state.cooldown_until_monotonic = time.monotonic() - 1.0

        self.assertFalse(state.is_open())
        self.assertEqual(state.circuit_state(), "half_open")

    def test_provider_status_exposes_process_local_history(self) -> None:
        with resilience_env():
            model_gateway.REGISTRY.record_failure(
                "groq",
                "openai/gpt-oss-120b",
                error_code="transport",
                detail=f"Bearer {FAKE_GROQ_KEY}",
                cooldown_seconds=0,
            )
            status = model_gateway.provider_status()

        self.assertEqual(status["diagnostic_scope"], "current_backend_process")
        self.assertIn("process_started_at", status)
        self.assertIn("resilience_policy", status)
        self.assertTrue(status["recent_events"])
        self.assertNotIn(FAKE_GROQ_KEY, repr(status["recent_events"]))

    def test_health_endpoint_reads_the_running_process_registry(self) -> None:
        client = TestClient(main.app)
        with resilience_env():
            model_gateway.REGISTRY.record_failure(
                "groq",
                "openai/gpt-oss-120b",
                error_code="transport",
                detail="temporary",
                cooldown_seconds=0,
            )
            response = client.get("/health/providers")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["diagnostic_scope"], "current_backend_process")
        groq = next(
            item
            for item in payload["circuits"]
            if item["provider"] == "groq"
            and item["model"] == "openai/gpt-oss-120b"
        )
        self.assertEqual(groq["total_failures"], 1)

    def test_health_probe_reuses_read_only_preflight(self) -> None:
        client = TestClient(main.app)
        with resilience_env():
            with patch.object(model_gateway, "preflight", return_value=()) as preflight:
                response = client.get("/health/providers?probe=true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["preflight"], [])
        preflight.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
