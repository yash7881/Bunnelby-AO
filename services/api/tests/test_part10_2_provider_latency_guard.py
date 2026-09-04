from __future__ import annotations

import contextlib
import os
import unittest
import urllib.error
from unittest.mock import patch

from services.api.app import model_gateway
from services.api.app.provider_config import LLMResult


FAKE_GEMINI_KEY = "AIzaFAKEKEYFORTESTSONLY0123456789"
FAKE_GROQ_KEY = "gsk_FAKEKEYFORTESTSONLY0123456789AB"


@contextlib.contextmanager
def latency_env(**overrides: str):
    baseline = {
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "GEMINI_MODEL": "gemini-3.6-flash",
        "GEMINI_FAST_MODEL": "gemini-3.5-flash-lite",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "GEMINI_REQUEST_TIMEOUT_SECONDS": "20",
        "LLM_MAX_TRANSIENT_RETRIES": "1",
        "LLM_TRANSIENT_RETRY_DELAY_MS": "1",
        "LLM_TRANSIENT_COOLDOWN_SECONDS": "30",
    }
    baseline.update(overrides)
    with patch.dict(os.environ, baseline, clear=False):
        yield


def provider_error(code: str, cooldown: int = 30):
    return model_gateway._ProviderCallError(
        error_code=code,
        message="temporary provider failure",
        recoverable=True,
        cooldown_seconds=cooldown,
    )


class ProviderLatencyGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        model_gateway.REGISTRY.reset()
        model_gateway.reset_clients()
        self.addCleanup(model_gateway.REGISTRY.reset)
        self.addCleanup(model_gateway.reset_clients)
        model_gateway.register_local_provider(model_gateway.UnconfiguredLocalProvider())
        self.addCleanup(
            model_gateway.register_local_provider,
            model_gateway.UnconfiguredLocalProvider(),
        )

    def test_gemini_client_sets_hard_timeout_and_disables_sdk_retries(self) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def close(self) -> None:
                pass

        with latency_env(GEMINI_REQUEST_TIMEOUT_SECONDS="20"):
            with patch.object(model_gateway.genai, "Client", FakeClient):
                model_gateway.gemini_client()

        options = captured["http_options"]
        self.assertEqual(options.timeout, 20_000)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_timeout_change_builds_a_separate_cached_client(self) -> None:
        built: list[object] = []

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                built.append(kwargs["http_options"])

            def close(self) -> None:
                pass

        with patch.object(model_gateway.genai, "Client", FakeClient):
            with latency_env(GEMINI_REQUEST_TIMEOUT_SECONDS="20"):
                first = model_gateway.gemini_client()
                again = model_gateway.gemini_client()
            with latency_env(GEMINI_REQUEST_TIMEOUT_SECONDS="21"):
                second = model_gateway.gemini_client()

        self.assertIs(first, again)
        self.assertIsNot(first, second)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0].timeout, 20_000)
        self.assertEqual(built[1].timeout, 21_000)

    def test_gemini_timeout_is_capped_to_sixty_seconds(self) -> None:
        with latency_env(GEMINI_REQUEST_TIMEOUT_SECONDS="999"):
            self.assertEqual(model_gateway.gemini_request_timeout_seconds(), 60)

    def test_default_transient_retry_policy_is_fail_fast(self) -> None:
        with patch.dict(os.environ, {"LLM_MAX_TRANSIENT_RETRIES": ""}, clear=False):
            self.assertEqual(model_gateway.max_transient_retries(), 0)

    def test_timeout_never_retries_the_same_provider(self) -> None:
        calls = 0

        def invoke():
            nonlocal calls
            calls += 1
            raise provider_error("timeout")

        with latency_env():
            with patch.object(model_gateway.time, "sleep") as sleep:
                with self.assertRaises(model_gateway._ProviderCallError):
                    model_gateway._call_with_transient_retry(
                        "gemini", "gemini-3.5-flash-lite", invoke
                    )

        self.assertEqual(calls, 1)
        sleep.assert_not_called()
        self.assertFalse(model_gateway.REGISTRY.recent_events())

    def test_timeout_detection_handles_direct_and_wrapped_timeouts(self) -> None:
        self.assertTrue(model_gateway._is_timeout_exception(TimeoutError("slow")))
        self.assertTrue(
            model_gateway._is_timeout_exception(
                urllib.error.URLError(TimeoutError("slow"))
            )
        )
        self.assertFalse(model_gateway._is_timeout_exception(RuntimeError("boom")))

    def test_timeout_gets_short_breaker_not_rate_limit_cooldown(self) -> None:
        with latency_env(LLM_TRANSIENT_COOLDOWN_SECONDS="17"):
            self.assertEqual(
                model_gateway._cooldown_for_failure(
                    "gemini", "timeout", retry_after_seconds=None
                ),
                17,
            )

    def test_timeout_fails_over_to_groq_immediately(self) -> None:
        with latency_env():
            with patch.object(
                model_gateway,
                "_gemini_call",
                side_effect=provider_error("timeout"),
            ) as gemini, patch.object(
                model_gateway,
                "_groq_call",
                return_value=LLMResult(
                    text="fallback",
                    provider="groq",
                    model="openai/gpt-oss-120b",
                ),
            ) as groq:
                result = model_gateway.generate(
                    system_instruction="s",
                    user_content="u",
                    profile_name="fast",
                )

        self.assertEqual(result.provider, "groq")
        self.assertEqual(gemini.call_count, 1)
        self.assertEqual(groq.call_count, 1)
        state = model_gateway.REGISTRY.state("gemini", "gemini-3.5-flash-lite")
        self.assertEqual(state.last_error_code, "timeout")
        self.assertTrue(state.is_open())

    def test_provider_status_exposes_latency_guard_policy(self) -> None:
        with latency_env(GEMINI_REQUEST_TIMEOUT_SECONDS="19"):
            policy = model_gateway.provider_status()["resilience_policy"]

        self.assertEqual(policy["gemini_request_timeout_seconds"], 19)
        self.assertEqual(policy["gemini_sdk_retry_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
