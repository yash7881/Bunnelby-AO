from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Final, Literal, Mapping
from urllib.parse import urlsplit

from google import genai

from .provider_config import (
    GROQ_API_HOST,
    GROQ_MODELS_URL,
    gemini_fast_model_name,
    gemini_model_name,
    groq_model_name,
)

logger = logging.getLogger(__name__)

# Part 10.2 step 0: read-only provider/model availability preflight.
#
# This module answers exactly one question per configured (provider, model) pair:
# can Bunnelby prove -- without spending a completion -- that the model it is
# configured to call actually exists for this account?
#
# It never raises to its caller and never fails application startup. Every
# outcome is returned as a structured ProviderHealth record so the future Model
# Gateway and the Reality Layer can consume one shape.
#
# Deliberately NOT in scope here: request routing, client caching, circuit
# breaking, quota accounting. Those belong to the Model Gateway.

ProviderModelStatus = Literal["available", "missing", "degraded", "not_configured"]

ModelLister = Callable[[], "tuple[str, ...]"]

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS: Final[int] = 10
MAX_DETAIL_CHARS: Final[int] = 300
GEMINI_MODEL_NAME_PREFIX: Final[str] = "models/"

PROVIDER_KEY_ENV: Final[Mapping[str, str]] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Stable machine-readable codes for the future Reality Layer. Never reuse a code
# for a different meaning; add a new one instead.
ERROR_NOT_CONFIGURED: Final[str] = "not_configured"
ERROR_NO_MODEL_CONFIGURED: Final[str] = "no_model_configured"
ERROR_UNKNOWN_PROVIDER: Final[str] = "unknown_provider"
ERROR_MODEL_NOT_LISTED: Final[str] = "model_not_listed"
ERROR_LIST_FAILED: Final[str] = "list_failed"

_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"gsk_[0-9A-Za-z]{10,}"),
    re.compile(r"(?i)bearer\s+[0-9A-Za-z._\-]{10,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|authorization)\b\s*[=:]\s*\S+"),
)


class ProviderListError(RuntimeError):
    """Raised internally when a provider's model list cannot be retrieved."""


@dataclass(frozen=True)
class ProviderHealth:
    """One read-only availability verdict for a single (provider, model) pair."""

    provider: str
    configured: bool
    model: str
    available: bool
    status: ProviderModelStatus
    error_code: str | None
    detail: str
    checked_at: datetime


def preflight_timeout_seconds() -> int:
    raw = os.getenv("PROVIDER_PREFLIGHT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid PROVIDER_PREFLIGHT_TIMEOUT_SECONDS=%r; using %s",
            raw,
            DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        )
        return DEFAULT_PREFLIGHT_TIMEOUT_SECONDS


def redact_secrets(text: object) -> str:
    """Scrub credentials from any string that may reach a caller, log or ledger.

    Configured key values are removed by exact match first, then generic
    credential shapes. The result is whitespace-collapsed and length-bounded so a
    provider error can never smuggle a large or secret payload into evidence.
    """
    cleaned = str(text)
    for env_name in PROVIDER_KEY_ENV.values():
        secret = os.getenv(env_name, "").strip()
        # An 8-character floor avoids mangling the string when a key is unset or
        # is a placeholder short enough to collide with ordinary words.
        if len(secret) >= 8:
            cleaned = cleaned.replace(secret, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > MAX_DETAIL_CHARS:
        cleaned = cleaned[: MAX_DETAIL_CHARS - 1].rstrip() + "…"
    return cleaned


def provider_configured(provider: str) -> bool:
    env_name = PROVIDER_KEY_ENV.get(provider)
    if env_name is None:
        return False
    return bool(os.getenv(env_name, "").strip())


def _validate_fixed_groq_models_endpoint() -> None:
    """Fail closed if the compile-time Groq models endpoint is not exact HTTPS."""
    parsed = urlsplit(GROQ_MODELS_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != GROQ_API_HOST
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/openai/v1/models"
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderListError(
            "The configured Groq models endpoint failed the fixed HTTPS allowlist."
        )


def list_groq_models() -> tuple[str, ...]:
    """List Groq model ids. Read-only: no completion is ever requested."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ProviderListError("GROQ_API_KEY is not configured.")

    _validate_fixed_groq_models_endpoint()
    request = urllib.request.Request(
        GROQ_MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "AO-Desktop/1.0",
        },
        method="GET",
    )

    try:
        # B310 is suppressed only after the strict scheme/host/path validation
        # above; the URL is a compile-time constant and cannot be supplied or
        # redirected by user input.
        with urllib.request.urlopen(  # nosec B310
            request,
            timeout=preflight_timeout_seconds(),
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        # Only the status code is surfaced: an error body can echo the request,
        # which carries the Authorization header.
        raise ProviderListError(f"Groq model list failed (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderListError("Groq is temporarily unreachable.") from exc

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderListError("Groq returned an invalid model list payload.") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ProviderListError("Groq model list response contained no data array.")

    names = {
        str(item.get("id", "")).strip()
        for item in data
        if isinstance(item, Mapping) and str(item.get("id", "")).strip()
    }
    if not names:
        raise ProviderListError("Groq model list response contained no usable model ids.")
    return tuple(sorted(names))


def list_gemini_models() -> tuple[str, ...]:
    """List Gemini model names. Read-only: no content is ever generated."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ProviderListError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    names: set[str] = set()
    try:
        for model in client.models.list():
            raw = str(getattr(model, "name", "") or "").strip()
            if not raw:
                continue
            if raw.startswith(GEMINI_MODEL_NAME_PREFIX):
                raw = raw[len(GEMINI_MODEL_NAME_PREFIX) :]
            names.add(raw)
    except ProviderListError:
        raise
    except Exception as exc:
        raise ProviderListError(
            f"Gemini model list failed ({type(exc).__name__})."
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # A cleanup failure must never downgrade a successful listing.
                logger.debug("Gemini preflight client close failed", exc_info=True)

    if not names:
        raise ProviderListError("Gemini model list response contained no models.")
    return tuple(sorted(names))


MODEL_LISTERS: Final[Mapping[str, ModelLister]] = {
    "gemini": list_gemini_models,
    "groq": list_groq_models,
}


def _fetch_models(
    provider: str,
    lister: ModelLister | None = None,
) -> tuple[tuple[str, ...] | None, str | None, str | None]:
    """Resolve one provider's model list.

    Returns (names, error_code, detail); exactly one of names / error_code is set.
    """
    resolve = lister or MODEL_LISTERS.get(provider)
    if resolve is None:
        return (
            None,
            ERROR_UNKNOWN_PROVIDER,
            f"{provider!r} is not a known Bunnelby LLM provider.",
        )

    try:
        return tuple(resolve()), None, None
    except ProviderListError as exc:
        detail = redact_secrets(exc)
        logger.warning("Provider preflight could not list %s models: %s", provider, detail)
        return None, ERROR_LIST_FAILED, detail
    except Exception as exc:
        detail = redact_secrets(
            f"Unexpected {type(exc).__name__} while listing {provider} models."
        )
        logger.warning("Provider preflight failed for %s: %s", provider, detail)
        return None, ERROR_LIST_FAILED, detail


def _build(
    provider: str,
    model: str,
    *,
    configured: bool,
    status: ProviderModelStatus,
    error_code: str | None,
    detail: str,
    checked_at: datetime,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        configured=configured,
        model=model,
        available=status == "available",
        status=status,
        error_code=error_code,
        detail=redact_secrets(detail),
        checked_at=checked_at,
    )


def _evaluate(
    provider: str,
    model: str,
    fetched: tuple[tuple[str, ...] | None, str | None, str | None],
    *,
    checked_at: datetime,
) -> ProviderHealth:
    names, error_code, detail = fetched
    if names is None:
        status: ProviderModelStatus = (
            "not_configured" if error_code == ERROR_UNKNOWN_PROVIDER else "degraded"
        )
        return _build(
            provider,
            model,
            configured=provider_configured(provider),
            status=status,
            error_code=error_code,
            detail=detail or "The provider model list is unavailable.",
            checked_at=checked_at,
        )

    if model in names:
        return _build(
            provider,
            model,
            configured=True,
            status="available",
            error_code=None,
            detail=f"{model} is listed among {len(names)} available {provider} models.",
            checked_at=checked_at,
        )

    return _build(
        provider,
        model,
        configured=True,
        status="missing",
        error_code=ERROR_MODEL_NOT_LISTED,
        detail=(
            f"{model} is not listed among the {len(names)} models this "
            f"{provider} account can currently use."
        ),
        checked_at=checked_at,
    )


def check_provider_model(
    provider: str,
    model: str,
    *,
    lister: ModelLister | None = None,
) -> ProviderHealth:
    """Check one (provider, model) pair. Never raises; never generates content."""
    checked_at = datetime.now(timezone.utc)
    requested = (model or "").strip()

    if provider not in PROVIDER_KEY_ENV:
        return _build(
            provider,
            requested,
            configured=False,
            status="not_configured",
            error_code=ERROR_UNKNOWN_PROVIDER,
            detail=f"{provider!r} is not a known Bunnelby LLM provider.",
            checked_at=checked_at,
        )

    if not provider_configured(provider):
        env_name = PROVIDER_KEY_ENV[provider]
        return _build(
            provider,
            requested,
            configured=False,
            status="not_configured",
            error_code=ERROR_NOT_CONFIGURED,
            detail=f"{env_name} is not configured.",
            checked_at=checked_at,
        )

    if not requested:
        return _build(
            provider,
            requested,
            configured=True,
            status="degraded",
            error_code=ERROR_NO_MODEL_CONFIGURED,
            detail=f"No {provider} model name is configured.",
            checked_at=checked_at,
        )

    return _evaluate(
        provider,
        requested,
        _fetch_models(provider, lister),
        checked_at=checked_at,
    )


def configured_provider_models() -> tuple[tuple[str, str], ...]:
    """Every (provider, model) pair Bunnelby can actually call today."""
    return (
        ("gemini", gemini_model_name()),
        ("gemini", gemini_fast_model_name()),
        ("groq", groq_model_name()),
    )


def get_provider_health(
    *,
    listers: Mapping[str, ModelLister] | None = None,
) -> tuple[ProviderHealth, ...]:
    """Preflight every configured (provider, model) pair.

    Each provider's model list is fetched at most once per call, and duplicate
    (provider, model) pairs are reported once. Never raises, so a caller may run
    this at startup without risking application boot.
    """
    overrides = dict(listers or {})

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider, model in configured_provider_models():
        key = (provider, (model or "").strip())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    fetched: dict[str, tuple[tuple[str, ...] | None, str | None, str | None]] = {}
    records: list[ProviderHealth] = []

    for provider, model in pairs:
        checked_at = datetime.now(timezone.utc)

        if not provider_configured(provider):
            env_name = PROVIDER_KEY_ENV.get(provider, "the provider API key")
            records.append(
                _build(
                    provider,
                    model,
                    configured=False,
                    status="not_configured",
                    error_code=ERROR_NOT_CONFIGURED,
                    detail=f"{env_name} is not configured.",
                    checked_at=checked_at,
                )
            )
            continue

        if not model:
            records.append(
                _build(
                    provider,
                    model,
                    configured=True,
                    status="degraded",
                    error_code=ERROR_NO_MODEL_CONFIGURED,
                    detail=f"No {provider} model name is configured.",
                    checked_at=checked_at,
                )
            )
            continue

        if provider not in fetched:
            fetched[provider] = _fetch_models(provider, overrides.get(provider))
        records.append(_evaluate(provider, model, fetched[provider], checked_at=checked_at))

    return tuple(records)
