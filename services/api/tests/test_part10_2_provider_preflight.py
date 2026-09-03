from __future__ import annotations

import contextlib
import json
import os
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

from services.api.app import llm_service, provider_health
from services.api.app.provider_health import (
    ProviderListError,
    check_provider_model,
    configured_provider_models,
    get_provider_health,
    list_gemini_models,
    list_groq_models,
    redact_secrets,
)

FAKE_GEMINI_KEY = "AIzaFAKEKEYFORTESTSONLY0123456789"
FAKE_GROQ_KEY = "gsk_FAKEKEYFORTESTSONLY0123456789AB"

GROQ_LISTED = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
GEMINI_LISTED = ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash")


@contextlib.contextmanager
def provider_env(**overrides: str):
    """Set provider-related environment variables for one test.

    llm_service loads the real .env at import time, so every test states the
    values it depends on explicitly. An empty string means "not configured".
    """
    baseline = {
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
        "GROQ_API_KEY": FAKE_GROQ_KEY,
        "GEMINI_MODEL": "gemini-3.6-flash",
        "GEMINI_FAST_MODEL": "gemini-3.5-flash-lite",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "PROVIDER_PREFLIGHT_TIMEOUT_SECONDS": "3",
    }
    baseline.update(overrides)
    with patch.dict(os.environ, baseline, clear=False):
        yield


def groq_lister(names: tuple[str, ...] = GROQ_LISTED):
    return lambda: names


def gemini_lister(names: tuple[str, ...] = GEMINI_LISTED):
    return lambda: names


class _FakeHTTPResponse:
    """Minimal urlopen stand-in: the context-manager dunders must live on a type."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def exploding_lister(exc: Exception):
    def _raise() -> tuple[str, ...]:
        raise exc

    return _raise


class GroqDefaultRepairTests(unittest.TestCase):
    """Part 10.2: the dead Groq code default must be gone for good."""

    def test_code_default_is_no_longer_the_retired_llama_model(self) -> None:
        self.assertNotEqual(
            llm_service.DEFAULT_GROQ_MODEL,
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-versatile was retired by Groq and is absent from the live "
            "model list; it must never be the code default again.",
        )

    def test_code_default_is_the_production_tier_fallback(self) -> None:
        self.assertEqual(llm_service.DEFAULT_GROQ_MODEL, "openai/gpt-oss-120b")

    def test_env_still_overrides_the_code_default(self) -> None:
        with provider_env(GROQ_MODEL="qwen/qwen3.8-27b"):
            self.assertEqual(llm_service.groq_model_name(), "qwen/qwen3.8-27b")

    def test_blank_env_falls_back_to_the_code_default(self) -> None:
        with provider_env(GROQ_MODEL="   "):
            self.assertEqual(llm_service.groq_model_name(), "openai/gpt-oss-120b")

    def test_public_generation_functions_are_preserved(self) -> None:
        for name in ("generate_text", "generate_fast_text", "generate_groq_text"):
            self.assertTrue(
                callable(getattr(llm_service, name, None)),
                f"{name} must remain a public llm_service entry point.",
            )


class ProviderModelPreflightTests(unittest.TestCase):
    """The four verdicts: AVAILABLE / MISSING / DEGRADED / NOT_CONFIGURED."""

    def test_groq_configured_model_exists(self) -> None:
        with provider_env():
            health = check_provider_model(
                "groq", "openai/gpt-oss-120b", lister=groq_lister()
            )
        self.assertEqual(health.status, "available")
        self.assertTrue(health.available)
        self.assertTrue(health.configured)
        self.assertIsNone(health.error_code)
        self.assertEqual(health.provider, "groq")
        self.assertEqual(health.model, "openai/gpt-oss-120b")
        self.assertIsNotNone(health.checked_at.tzinfo)

    def test_groq_configured_model_missing(self) -> None:
        with provider_env():
            health = check_provider_model(
                "groq", "llama-3.3-70b-versatile", lister=groq_lister()
            )
        self.assertEqual(health.status, "missing")
        self.assertFalse(health.available)
        self.assertTrue(health.configured)
        self.assertEqual(health.error_code, "model_not_listed")
        self.assertIn("llama-3.3-70b-versatile", health.detail)

    def test_groq_key_missing(self) -> None:
        with provider_env(GROQ_API_KEY=""):
            health = check_provider_model(
                "groq", "openai/gpt-oss-120b", lister=groq_lister()
            )
        self.assertEqual(health.status, "not_configured")
        self.assertFalse(health.available)
        self.assertFalse(health.configured)
        self.assertEqual(health.error_code, "not_configured")
        self.assertIn("GROQ_API_KEY", health.detail)

    def test_gemini_configured_model_exists(self) -> None:
        with provider_env():
            health = check_provider_model(
                "gemini", "gemini-3.6-flash", lister=gemini_lister()
            )
        self.assertEqual(health.status, "available")
        self.assertTrue(health.available)
        self.assertIsNone(health.error_code)

    def test_gemini_fast_model_exists(self) -> None:
        with provider_env():
            health = check_provider_model(
                "gemini", "gemini-3.5-flash-lite", lister=gemini_lister()
            )
        self.assertEqual(health.status, "available")

    def test_gemini_configured_model_missing(self) -> None:
        with provider_env():
            health = check_provider_model(
                "gemini", "gemini-9.9-imaginary", lister=gemini_lister()
            )
        self.assertEqual(health.status, "missing")
        self.assertFalse(health.available)
        self.assertEqual(health.error_code, "model_not_listed")

    def test_gemini_key_missing(self) -> None:
        with provider_env(GEMINI_API_KEY=""):
            health = check_provider_model(
                "gemini", "gemini-3.6-flash", lister=gemini_lister()
            )
        self.assertEqual(health.status, "not_configured")
        self.assertFalse(health.configured)
        self.assertEqual(health.error_code, "not_configured")
        self.assertIn("GEMINI_API_KEY", health.detail)

    def test_provider_list_error_becomes_degraded(self) -> None:
        with provider_env():
            health = check_provider_model(
                "groq",
                "openai/gpt-oss-120b",
                lister=exploding_lister(ProviderListError("Groq is temporarily unreachable.")),
            )
        self.assertEqual(health.status, "degraded")
        self.assertFalse(health.available)
        self.assertTrue(health.configured)
        self.assertEqual(health.error_code, "list_failed")
        self.assertIn("unreachable", health.detail)

    def test_unexpected_exception_becomes_degraded_not_a_crash(self) -> None:
        with provider_env():
            health = check_provider_model(
                "gemini",
                "gemini-3.6-flash",
                lister=exploding_lister(RuntimeError("boom")),
            )
        self.assertEqual(health.status, "degraded")
        self.assertEqual(health.error_code, "list_failed")
        self.assertIn("RuntimeError", health.detail)

    def test_unknown_provider_is_not_configured(self) -> None:
        with provider_env():
            health = check_provider_model("openai", "gpt-4o", lister=groq_lister())
        self.assertEqual(health.status, "not_configured")
        self.assertEqual(health.error_code, "unknown_provider")

    def test_blank_model_name_is_degraded(self) -> None:
        with provider_env():
            health = check_provider_model("groq", "   ", lister=groq_lister())
        self.assertEqual(health.status, "degraded")
        self.assertEqual(health.error_code, "no_model_configured")


class SecretRedactionTests(unittest.TestCase):
    """No provider credential may reach a returned record or a log line."""

    def test_configured_key_never_appears_in_returned_detail(self) -> None:
        leaky = ProviderListError(
            f"GET https://api.groq.com/openai/v1/models failed; "
            f"Authorization: Bearer {FAKE_GROQ_KEY}"
        )
        with provider_env():
            health = check_provider_model(
                "groq", "openai/gpt-oss-120b", lister=exploding_lister(leaky)
            )
        self.assertEqual(health.status, "degraded")
        self.assertNotIn(FAKE_GROQ_KEY, health.detail)
        self.assertIn("[redacted]", health.detail)

    def test_configured_key_never_appears_in_log_output(self) -> None:
        leaky = ProviderListError(f"gemini list failed for key {FAKE_GEMINI_KEY}")
        with provider_env():
            with self.assertLogs(provider_health.logger, level="WARNING") as captured:
                check_provider_model(
                    "gemini", "gemini-3.6-flash", lister=exploding_lister(leaky)
                )
        joined = "\n".join(captured.output)
        self.assertNotIn(FAKE_GEMINI_KEY, joined)
        self.assertIn("[redacted]", joined)

    def test_redact_secrets_scrubs_generic_credential_shapes(self) -> None:
        with provider_env(GEMINI_API_KEY="", GROQ_API_KEY=""):
            scrubbed = redact_secrets(
                "AIzaSyUNRELATEDKEY0123456789 gsk_SOMEOTHERKEY0123456789 "
                "api_key=leakedvalue Bearer abcdefghijklmnop"
            )
        for fragment in (
            "AIzaSyUNRELATEDKEY0123456789",
            "gsk_SOMEOTHERKEY0123456789",
            "leakedvalue",
            "abcdefghijklmnop",
        ):
            self.assertNotIn(fragment, scrubbed)

    def test_detail_length_is_bounded(self) -> None:
        with provider_env():
            health = check_provider_model(
                "groq",
                "openai/gpt-oss-120b",
                lister=exploding_lister(ProviderListError("x" * 5000)),
            )
        self.assertLessEqual(len(health.detail), provider_health.MAX_DETAIL_CHARS)

    def test_short_unset_key_does_not_mangle_detail(self) -> None:
        with provider_env(GROQ_API_KEY="abc"):
            scrubbed = redact_secrets("a plain message about abc models")
        self.assertEqual(scrubbed, "a plain message about abc models")


class ProviderHealthAggregateTests(unittest.TestCase):
    def test_covers_every_configured_pair(self) -> None:
        with provider_env():
            self.assertEqual(
                configured_provider_models(),
                (
                    ("gemini", "gemini-3.6-flash"),
                    ("gemini", "gemini-3.5-flash-lite"),
                    ("groq", "openai/gpt-oss-120b"),
                ),
            )
            records = get_provider_health(
                listers={"gemini": gemini_lister(), "groq": groq_lister()}
            )
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record.status == "available" for record in records))

    def test_duplicate_primary_and_fast_model_is_reported_once(self) -> None:
        with provider_env(GEMINI_FAST_MODEL="gemini-3.6-flash"):
            records = get_provider_health(
                listers={"gemini": gemini_lister(), "groq": groq_lister()}
            )
        self.assertEqual(len(records), 2)

    def test_each_provider_list_is_fetched_at_most_once(self) -> None:
        calls: list[str] = []

        def counting_gemini() -> tuple[str, ...]:
            calls.append("gemini")
            return GEMINI_LISTED

        with provider_env():
            get_provider_health(
                listers={"gemini": counting_gemini, "groq": groq_lister()}
            )
        self.assertEqual(calls, ["gemini"], "two Gemini models must share one list call")

    def test_degraded_provider_does_not_raise_and_leaves_others_intact(self) -> None:
        with provider_env():
            records = get_provider_health(
                listers={
                    "gemini": exploding_lister(ProviderListError("gemini offline")),
                    "groq": groq_lister(),
                }
            )
        by_provider: dict[str, list[str]] = {}
        for record in records:
            by_provider.setdefault(record.provider, []).append(record.status)
        self.assertEqual(by_provider["gemini"], ["degraded", "degraded"])
        self.assertEqual(by_provider["groq"], ["available"])

    def test_no_provider_configured_still_returns_records(self) -> None:
        with provider_env(GEMINI_API_KEY="", GROQ_API_KEY=""):
            records = get_provider_health()
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record.status == "not_configured" for record in records))
        self.assertTrue(all(not record.available for record in records))


class GroqModelListerTests(unittest.TestCase):
    """The real Groq lister: read-only GET, no completion, no key leakage."""

    def _response(self, payload: dict) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    def test_parses_model_ids_from_a_successful_response(self) -> None:
        payload = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "qwen/qwen3.8-27b"}]}
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["method"] = request.method
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return self._response(payload)

        with provider_env():
            with patch.object(provider_health.urllib.request, "urlopen", fake_urlopen):
                names = list_groq_models()

        self.assertEqual(names, ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"))
        self.assertEqual(captured["method"], "GET", "preflight must never POST a completion")
        self.assertEqual(captured["url"], llm_service.GROQ_MODELS_URL)
        self.assertEqual(captured["timeout"], 3)

    def test_http_error_raises_provider_list_error_without_the_body(self) -> None:
        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            raise urllib.error.HTTPError(
                llm_service.GROQ_MODELS_URL,
                401,
                f"Unauthorized: Bearer {FAKE_GROQ_KEY}",
                {},  # type: ignore[arg-type]
                None,
            )

        with provider_env():
            with patch.object(provider_health.urllib.request, "urlopen", fake_urlopen):
                with self.assertRaises(ProviderListError) as ctx:
                    list_groq_models()
        message = str(ctx.exception)
        self.assertIn("401", message)
        self.assertNotIn(FAKE_GROQ_KEY, message)

    def test_network_failure_raises_provider_list_error(self) -> None:
        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            raise urllib.error.URLError("dns failure")

        with provider_env():
            with patch.object(provider_health.urllib.request, "urlopen", fake_urlopen):
                with self.assertRaises(ProviderListError):
                    list_groq_models()

    def test_malformed_payload_raises_provider_list_error(self) -> None:
        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            return self._response({"unexpected": True})

        with provider_env():
            with patch.object(provider_health.urllib.request, "urlopen", fake_urlopen):
                with self.assertRaises(ProviderListError):
                    list_groq_models()

    def test_missing_key_raises_before_any_network_call(self) -> None:
        def fail_urlopen(request, timeout=None):  # noqa: ANN001
            raise AssertionError("no network call may happen without a key")

        with provider_env(GROQ_API_KEY=""):
            with patch.object(provider_health.urllib.request, "urlopen", fail_urlopen):
                with self.assertRaises(ProviderListError):
                    list_groq_models()

    def test_models_endpoint_passes_the_fixed_https_allowlist(self) -> None:
        provider_health._validate_fixed_groq_models_endpoint()

    def test_tampered_models_endpoint_fails_closed(self) -> None:
        with patch.object(provider_health, "GROQ_MODELS_URL", "http://evil.test/openai/v1/models"):
            with self.assertRaises(ProviderListError):
                provider_health._validate_fixed_groq_models_endpoint()


class GeminiModelListerTests(unittest.TestCase):
    """The real Gemini lister: models.list() only, never generate_content."""

    def _client(self, names: tuple[str, ...], *, closed: list[bool]) -> SimpleNamespace:
        models = SimpleNamespace(
            list=lambda: [SimpleNamespace(name=f"models/{name}") for name in names],
            generate_content=lambda **_: (_ for _ in ()).throw(
                AssertionError("preflight must never generate content")
            ),
        )
        return SimpleNamespace(models=models, close=lambda: closed.append(True))

    def test_strips_the_models_prefix_and_closes_the_client(self) -> None:
        closed: list[bool] = []
        client = self._client(("gemini-3.6-flash", "gemini-3.5-flash-lite"), closed=closed)

        with provider_env():
            with patch.object(provider_health.genai, "Client", lambda **_: client):
                names = list_gemini_models()

        self.assertEqual(names, ("gemini-3.5-flash-lite", "gemini-3.6-flash"))
        self.assertEqual(closed, [True])

    def test_sdk_failure_raises_provider_list_error_named_by_type(self) -> None:
        def listing_failure() -> list[object]:
            raise ConnectionResetError("socket closed")

        client = SimpleNamespace(
            models=SimpleNamespace(list=listing_failure),
            close=lambda: None,
        )

        with provider_env():
            with patch.object(provider_health.genai, "Client", lambda **_: client):
                with self.assertRaises(ProviderListError) as ctx:
                    list_gemini_models()
        self.assertIn("ConnectionResetError", str(ctx.exception))

    def test_close_failure_does_not_break_a_successful_listing(self) -> None:
        def bad_close() -> None:
            raise RuntimeError("close failed")

        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: [SimpleNamespace(name="models/gemini-3.6-flash")]
            ),
            close=bad_close,
        )

        with provider_env():
            with patch.object(provider_health.genai, "Client", lambda **_: client):
                self.assertEqual(list_gemini_models(), ("gemini-3.6-flash",))

    def test_missing_key_raises_before_constructing_a_client(self) -> None:
        def fail_client(**_: object) -> object:
            raise AssertionError("no client may be built without a key")

        with provider_env(GEMINI_API_KEY=""):
            with patch.object(provider_health.genai, "Client", fail_client):
                with self.assertRaises(ProviderListError):
                    list_gemini_models()

    def test_empty_listing_raises_provider_list_error(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: []),
            close=lambda: None,
        )
        with provider_env():
            with patch.object(provider_health.genai, "Client", lambda **_: client):
                with self.assertRaises(ProviderListError):
                    list_gemini_models()


if __name__ == "__main__":
    unittest.main()
