from __future__ import annotations

import contextlib
import os
import unittest
from unittest.mock import patch

from services.api.app import cross_tool_fastpath, llm_service, model_gateway
from services.api.app.provider_config import (
    LLMConfigurationError,
    LLMResult,
    LLMUnavailableError,
)

FAKE_GEMINI_KEY = "AIzaFAKEKEYFORTESTSONLY0123456789"
FAKE_GROQ_KEY = "gsk_FAKEKEYFORTESTSONLY0123456789AB"


@contextlib.contextmanager
def gateway_env(**overrides: str):
    baseline = {
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "GEMINI_MODEL": "gemini-3.6-flash",
        "GEMINI_FAST_MODEL": "gemini-3.5-flash-lite",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "GEMINI_COOLDOWN_SECONDS": "900",
        "GROQ_COOLDOWN_SECONDS": "120",
    }
    baseline.update(overrides)
    with patch.dict(os.environ, baseline, clear=False):
        yield


def gemini_result(model: str = "gemini-3.6-flash") -> LLMResult:
    return LLMResult(text="gemini answer", provider="gemini", model=model)


def groq_result(model: str = "openai/gpt-oss-120b") -> LLMResult:
    return LLMResult(text="groq answer", provider="groq", model=model)


def provider_error(code: str, cooldown: int = 0, message: str = "boom"):
    return model_gateway._ProviderCallError(
        error_code=code, message=message, recoverable=cooldown > 0, cooldown_seconds=cooldown
    )


class GatewayTestCase(unittest.TestCase):
    """Every gateway test starts from clean circuit state."""

    def setUp(self) -> None:
        model_gateway.REGISTRY.reset()
        self.addCleanup(model_gateway.REGISTRY.reset)
        model_gateway.register_local_provider(model_gateway.UnconfiguredLocalProvider())
        self.addCleanup(
            model_gateway.register_local_provider, model_gateway.UnconfiguredLocalProvider()
        )


class ProviderRoutingTests(GatewayTestCase):
    def test_gemini_success_on_the_balanced_profile(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", return_value=gemini_result()
            ) as gemini, patch.object(model_gateway, "_groq_call") as groq:
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(gemini.call_count, 1)
        groq.assert_not_called()

    def test_fast_profile_uses_the_fast_gemini_model(self) -> None:
        seen: dict[str, object] = {}

        def capture(
            model, system_instruction, user_content, temperature, *, minimal_thinking, **_kw
        ):
            seen["model"] = model
            seen["minimal_thinking"] = minimal_thinking
            return gemini_result(model)

        with gateway_env():
            with patch.object(model_gateway, "_gemini_call", capture):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="fast"
                )
        self.assertEqual(seen["model"], "gemini-3.5-flash-lite")
        self.assertTrue(seen["minimal_thinking"])

    def test_one_successful_call_per_request(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", return_value=gemini_result()
            ) as gemini, patch.object(model_gateway, "_groq_call") as groq:
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(gemini.call_count + groq.call_count, 1)

    def test_gemini_429_opens_the_breaker_and_falls_back_to_groq(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_429", 900)
            ) as gemini, patch.object(
                model_gateway, "_groq_call", return_value=groq_result()
            ) as groq:
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
            self.assertEqual(result.provider, "groq")
            self.assertEqual(gemini.call_count, 1)
            self.assertEqual(groq.call_count, 1)

            state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
            self.assertTrue(state.is_open())
            self.assertEqual(state.last_error_code, "http_429")
            self.assertGreater(state.cooldown_remaining_seconds(), 0)

    def test_open_breaker_skips_gemini_entirely_on_the_next_turn(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_429", 900)
            ), patch.object(model_gateway, "_groq_call", return_value=groq_result()):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )

            with patch.object(model_gateway, "_gemini_call") as gemini, patch.object(
                model_gateway, "_groq_call", return_value=groq_result()
            ) as groq:
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        gemini.assert_not_called()
        self.assertEqual(groq.call_count, 1)
        self.assertEqual(result.provider, "groq")

    def test_gemini_transport_failure_falls_back(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("transport", 900)
            ), patch.object(model_gateway, "_groq_call", return_value=groq_result()):
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(result.provider, "groq")
        self.assertEqual(
            model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash").last_error_code,
            "transport",
        )

    def test_groq_unavailable_leaves_gemini_serving(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", return_value=gemini_result()
            ), patch.object(
                model_gateway, "_groq_call", side_effect=provider_error("transport", 120)
            ):
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(result.provider, "gemini")

    def test_groq_only_profile_failure_raises_unavailable(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_groq_call", side_effect=provider_error("http_503", 120)
            ):
                with self.assertRaises(LLMUnavailableError):
                    model_gateway.generate(
                        system_instruction="s", user_content="u", profile_name="groq_only"
                    )
        self.assertTrue(
            model_gateway.REGISTRY.state("groq", "openai/gpt-oss-120b").is_open(),
            "Groq now has its own breaker; before Phase E only Gemini did.",
        )

    def test_no_provider_configured_raises_configuration_error(self) -> None:
        with gateway_env(GEMINI_API_KEY="", GROQ_API_KEY=""):
            with self.assertRaises(LLMConfigurationError):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )

    def test_both_cloud_providers_failing_raises_unavailable(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_429", 900)
            ), patch.object(
                model_gateway, "_groq_call", side_effect=provider_error("http_503", 120)
            ):
                with self.assertRaises(LLMUnavailableError):
                    model_gateway.generate(
                        system_instruction="s", user_content="u", profile_name="balanced"
                    )

    def test_unknown_profile_is_a_configuration_error(self) -> None:
        with gateway_env():
            with self.assertRaises(LLMConfigurationError):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="does-not-exist"
                )


class BreakerRecoveryTests(GatewayTestCase):
    def test_cooldown_expiry_lets_the_primary_serve_again(self) -> None:
        with gateway_env(GEMINI_COOLDOWN_SECONDS="1"):
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_429", 1)
            ), patch.object(model_gateway, "_groq_call", return_value=groq_result()):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )

            state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
            self.assertTrue(state.is_open())

            # Expire the cooldown deterministically instead of sleeping.
            state.cooldown_until_monotonic = 0.0

            with patch.object(
                model_gateway, "_gemini_call", return_value=gemini_result()
            ) as gemini:
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(gemini.call_count, 1)

    def test_success_resets_failure_counters(self) -> None:
        with gateway_env(GEMINI_COOLDOWN_SECONDS="1"):
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_500", 1)
            ), patch.object(model_gateway, "_groq_call", return_value=groq_result()):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
            state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
            self.assertEqual(state.consecutive_failures, 1)
            state.cooldown_until_monotonic = 0.0

            with patch.object(model_gateway, "_gemini_call", return_value=gemini_result()):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(state.consecutive_failures, 0)
        self.assertIsNone(state.last_error_code)
        self.assertEqual(state.total_successes, 1)

    def test_latency_is_tracked_per_pair(self) -> None:
        with gateway_env():
            with patch.object(model_gateway, "_gemini_call", return_value=gemini_result()):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
        self.assertIsNotNone(state.average_latency_ms())

    def test_legacy_cooldown_surface_still_works(self) -> None:
        with gateway_env():
            self.assertFalse(llm_service.gemini_cooldown_active())
            llm_service.activate_gemini_cooldown("manual test")
            self.assertTrue(llm_service.gemini_cooldown_active())
            self.assertGreater(llm_service.gemini_cooldown_remaining_seconds(), 0)
            llm_service.clear_gemini_cooldown()
            self.assertFalse(llm_service.gemini_cooldown_active())


class DegradedAndLocalModeTests(GatewayTestCase):
    class _StubLocal:
        name = "stub-local"

        def available(self) -> bool:
            return True

        def generate(self, system_instruction, user_content, temperature):  # noqa: ANN001
            return LLMResult(text="local answer", provider="local", model="stub-local")

    def test_mode_is_cloud_when_a_cloud_pair_is_healthy(self) -> None:
        with gateway_env():
            self.assertEqual(model_gateway.gateway_mode(), "cloud")

    def test_mode_is_unavailable_with_no_cloud_and_no_local(self) -> None:
        with gateway_env(GEMINI_API_KEY="", GROQ_API_KEY=""):
            self.assertEqual(model_gateway.gateway_mode(), "unavailable")

    def test_mode_is_degraded_local_when_only_a_local_provider_exists(self) -> None:
        model_gateway.register_local_provider(self._StubLocal())
        with gateway_env(GEMINI_API_KEY="", GROQ_API_KEY=""):
            self.assertEqual(model_gateway.gateway_mode(), "degraded_local")

    def test_local_provider_serves_after_every_cloud_candidate_fails(self) -> None:
        model_gateway.register_local_provider(self._StubLocal())
        with gateway_env():
            with patch.object(
                model_gateway, "_gemini_call", side_effect=provider_error("http_429", 900)
            ), patch.object(
                model_gateway, "_groq_call", side_effect=provider_error("http_503", 120)
            ):
                result = model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
        self.assertEqual(result.provider, "local")

    def test_default_local_provider_is_unavailable_and_loads_no_model(self) -> None:
        provider = model_gateway.UnconfiguredLocalProvider()
        self.assertFalse(provider.available())
        with self.assertRaises(LLMConfigurationError):
            provider.generate("s", "u", 0.2)

    def test_breaker_open_on_all_pairs_reports_unavailable_mode(self) -> None:
        with gateway_env():
            model_gateway.REGISTRY.record_failure(
                "gemini", "gemini-3.6-flash", error_code="x", detail="d", cooldown_seconds=900
            )
            model_gateway.REGISTRY.record_failure(
                "gemini", "gemini-3.5-flash-lite", error_code="x", detail="d", cooldown_seconds=900
            )
            model_gateway.REGISTRY.record_failure(
                "groq", "openai/gpt-oss-120b", error_code="x", detail="d", cooldown_seconds=120
            )
            self.assertEqual(model_gateway.gateway_mode(), "unavailable")


class ClientReuseTests(GatewayTestCase):
    def test_gemini_client_is_cached_across_calls(self) -> None:
        built: list[str] = []

        class FakeClient:
            def __init__(self, **_: object) -> None:
                built.append("x")

            def close(self) -> None:
                pass

        model_gateway.reset_clients()
        self.addCleanup(model_gateway.reset_clients)
        with gateway_env():
            with patch.object(model_gateway.genai, "Client", FakeClient):
                first = model_gateway.gemini_client()
                second = model_gateway.gemini_client()
        self.assertIs(first, second)
        self.assertEqual(len(built), 1, "a new client per request was the Phase E defect")

    def test_missing_key_raises_before_building_a_client(self) -> None:
        model_gateway.reset_clients()
        self.addCleanup(model_gateway.reset_clients)

        def fail(**_: object) -> object:
            raise AssertionError("no client may be built without a key")

        with gateway_env(GEMINI_API_KEY=""):
            with patch.object(model_gateway.genai, "Client", fail):
                with self.assertRaises(LLMConfigurationError):
                    model_gateway.gemini_client()


class SecretRedactionTests(GatewayTestCase):
    def test_provider_failure_detail_never_carries_the_key(self) -> None:
        leak = provider_error("transport", 900, message=f"failed with Bearer {FAKE_GROQ_KEY}")
        with gateway_env():
            with patch.object(model_gateway, "_gemini_call", side_effect=leak), patch.object(
                model_gateway, "_groq_call", return_value=groq_result()
            ):
                model_gateway.generate(
                    system_instruction="s", user_content="u", profile_name="balanced"
                )
            state = model_gateway.REGISTRY.state("gemini", "gemini-3.6-flash")
        self.assertNotIn(FAKE_GROQ_KEY, state.last_error_detail)
        self.assertIn("[redacted]", state.last_error_detail)

    def test_aggregated_unavailable_message_is_redacted(self) -> None:
        with gateway_env():
            with patch.object(
                model_gateway,
                "_gemini_call",
                side_effect=provider_error("transport", 0, f"key {FAKE_GEMINI_KEY}"),
            ), patch.object(
                model_gateway,
                "_groq_call",
                side_effect=provider_error("transport", 0, f"key {FAKE_GROQ_KEY}"),
            ):
                with self.assertRaises(LLMUnavailableError) as ctx:
                    model_gateway.generate(
                        system_instruction="s", user_content="u", profile_name="balanced"
                    )
        message = str(ctx.exception)
        self.assertNotIn(FAKE_GEMINI_KEY, message)
        self.assertNotIn(FAKE_GROQ_KEY, message)

    def test_snapshots_are_redacted(self) -> None:
        with gateway_env():
            model_gateway.REGISTRY.record_failure(
                "groq",
                "openai/gpt-oss-120b",
                error_code="transport",
                detail=f"Authorization: Bearer {FAKE_GROQ_KEY}",
                cooldown_seconds=0,
            )
            snapshots = model_gateway.REGISTRY.snapshots()
        joined = repr(snapshots)
        self.assertNotIn(FAKE_GROQ_KEY, joined)


class ProfilePolicyTests(GatewayTestCase):
    def test_balanced_is_gemini_first(self) -> None:
        with gateway_env():
            self.assertEqual(
                model_gateway.candidate_models("balanced"),
                (("gemini", "gemini-3.6-flash"), ("groq", "openai/gpt-oss-120b")),
            )

    def test_low_latency_synthesis_is_groq_first(self) -> None:
        with gateway_env():
            self.assertEqual(
                model_gateway.candidate_models("low_latency_synthesis"),
                (("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.6-flash")),
            )

    def test_env_model_override_is_honored_per_call(self) -> None:
        with gateway_env(GROQ_MODEL="qwen/qwen3.8-27b"):
            self.assertEqual(
                model_gateway.candidate_models("balanced")[1], ("groq", "qwen/qwen3.8-27b")
            )

    def test_preview_models_are_flagged_as_such(self) -> None:
        with gateway_env(GROQ_MODEL="qwen/qwen3.8-27b"):
            state = model_gateway.REGISTRY.state("groq", "qwen/qwen3.8-27b")
        self.assertEqual(state.stability, "preview")

    def test_production_models_are_flagged_as_such(self) -> None:
        state = model_gateway.REGISTRY.state("groq", "openai/gpt-oss-120b")
        self.assertEqual(state.stability, "production")

    def test_provider_status_reports_every_profile(self) -> None:
        with gateway_env():
            status = model_gateway.provider_status()
        self.assertEqual(status["mode"], "cloud")
        self.assertEqual(
            set(status["profiles"]),  # type: ignore[arg-type]
            {"balanced", "fast", "low_latency_synthesis", "groq_only"},
        )

    def test_preflight_reuses_provider_health_without_duplicating_logic(self) -> None:
        with gateway_env():
            records = model_gateway.preflight(
                listers={
                    "gemini": lambda: ("gemini-3.6-flash", "gemini-3.5-flash-lite"),
                    "groq": lambda: ("openai/gpt-oss-120b",),
                }
            )
        by_model = {record.model: record.status for record in records}
        self.assertEqual(by_model["gemini-3.6-flash"], "available")
        self.assertEqual(by_model["gemini-3.5-flash-lite"], "available")
        self.assertEqual(by_model["openai/gpt-oss-120b"], "available")

    def test_preflight_reports_a_configured_but_missing_model(self) -> None:
        with gateway_env(GROQ_MODEL="llama-3.3-70b-versatile"):
            records = model_gateway.preflight(
                listers={
                    "gemini": lambda: ("gemini-3.6-flash", "gemini-3.5-flash-lite"),
                    "groq": lambda: ("openai/gpt-oss-120b",),
                }
            )
        missing = [r for r in records if r.model == "llama-3.3-70b-versatile"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].status, "missing")
        self.assertEqual(missing[0].error_code, "model_not_listed")


class PublicCompatibilityTests(GatewayTestCase):
    def test_generate_text_uses_the_balanced_profile(self) -> None:
        seen: dict[str, object] = {}

        def capture(*, system_instruction, user_content, profile_name, temperature=None, **_kw):
            seen["profile"] = profile_name
            return gemini_result()

        with gateway_env():
            with patch.object(model_gateway, "generate", capture):
                llm_service.generate_text(system_instruction="s", user_content="u")
        self.assertEqual(seen["profile"], "balanced")

    def test_generate_fast_text_uses_the_fast_profile(self) -> None:
        seen: dict[str, object] = {}

        def capture(*, system_instruction, user_content, profile_name, temperature=None, **_kw):
            seen["profile"] = profile_name
            return gemini_result()

        with gateway_env():
            with patch.object(model_gateway, "generate", capture):
                llm_service.generate_fast_text(system_instruction="s", user_content="u")
        self.assertEqual(seen["profile"], "fast")

    def test_generate_groq_text_uses_the_groq_only_profile(self) -> None:
        seen: dict[str, object] = {}

        def capture(*, system_instruction, user_content, profile_name, temperature=None, **_kw):
            seen["profile"] = profile_name
            return groq_result()

        with gateway_env():
            with patch.object(model_gateway, "generate", capture):
                llm_service.generate_groq_text(system_instruction="s", user_content="u")
        self.assertEqual(seen["profile"], "groq_only")

    def test_all_public_entry_points_remain_callable(self) -> None:
        for name in (
            "generate_text",
            "generate_fast_text",
            "generate_groq_text",
            "generate_gemini_text",
        ):
            self.assertTrue(callable(getattr(llm_service, name, None)), name)

    def test_compatibility_constants_are_still_exported(self) -> None:
        self.assertEqual(llm_service.DEFAULT_GROQ_MODEL, "openai/gpt-oss-120b")
        self.assertTrue(llm_service.GROQ_MODELS_URL.startswith("https://api.groq.com/"))
        self.assertIn(429, llm_service.RECOVERABLE_GEMINI_CODES)


class CrossToolGatewayPolicyTests(GatewayTestCase):
    def test_cross_tool_synthesis_asks_the_gateway_for_its_profile(self) -> None:
        seen: dict[str, object] = {}

        def capture(*, system_instruction, user_content, profile_name, temperature=None, **_kw):
            seen["profile"] = profile_name
            return LLMResult(
                text='{"reply":"r","spoken_reply":"r"}', provider="groq", model="m"
            )

        with gateway_env():
            with patch.object(model_gateway, "generate", capture), patch.object(
                cross_tool_fastpath, "_synthesis_envelope", return_value="envelope"
            ), patch.object(
                cross_tool_fastpath, "synthesize_results", return_value=("r", "r")
            ):
                cross_tool_fastpath._synthesize_low_latency("read gmail and calendar", None, ())

        self.assertEqual(
            seen["profile"],
            "low_latency_synthesis",
            "cross-tool must not keep a private provider-selection architecture.",
        )

    def test_cross_tool_no_longer_calls_a_provider_directly(self) -> None:
        source = model_gateway.__file__
        self.assertTrue(source.endswith("model_gateway.py"))
        import pathlib

        module = pathlib.Path(cross_tool_fastpath.__file__).read_text(encoding="utf-8")
        self.assertNotIn("generate_groq_text(", module)
        self.assertIn("profile_name=\"low_latency_synthesis\"", module)


if __name__ == "__main__":
    unittest.main()
