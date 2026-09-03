from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .database import PROJECT_ROOT

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")

# Part 10.2 Phase E: leaf configuration layer for cloud/local reasoning providers.
#
# This module deliberately imports nothing from llm_service, model_gateway or
# provider_health so the provider stack can be layered acyclically:
#
#   provider_config  (leaf: names, types, env policy)
#        ^      ^
#        |      +-- provider_health  (read-only availability preflight)
#        |               ^
#        +-- model_gateway  (client reuse, breaker, routing policy)
#                 ^
#                 +-- llm_service  (compatibility facade)

Provider = Literal["gemini", "groq", "local"]

DEFAULT_GEMINI_MODEL: Final[str] = "gemini-3.6-flash"
DEFAULT_GEMINI_FAST_MODEL: Final[str] = "gemini-3.5-flash-lite"
# openai/gpt-oss-120b is Groq production tier. The previous default
# (llama-3.3-70b-versatile) was retired and is absent from the live model list.
DEFAULT_GROQ_MODEL: Final[str] = "openai/gpt-oss-120b"

DEFAULT_GEMINI_COOLDOWN_SECONDS: Final[int] = 900
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[int] = 45

GROQ_CHAT_COMPLETIONS_URL: Final[str] = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL: Final[str] = "https://api.groq.com/openai/v1/models"
GROQ_API_HOST: Final[str] = "api.groq.com"

RECOVERABLE_GEMINI_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

PROVIDER_KEY_ENV: Final[dict[str, str]] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Groq classifies some models as Preview ("evaluation purposes only, may be
# discontinued without notice"). The gateway records this so a degraded-provider
# report can say *why* a fallback is fragile instead of only that it failed.
PREVIEW_MODEL_MARKERS: Final[tuple[str, ...]] = ("qwen/qwen3", "preview")


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str


class LLMServiceError(RuntimeError):
    """Base exception for Bunnelby language-model provider failures."""


class LLMConfigurationError(LLMServiceError):
    """Raised when no configured provider can serve a request."""


class LLMUnavailableError(LLMServiceError):
    """Raised when configured providers are temporarily unavailable."""


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def gemini_fast_model_name() -> str:
    return (
        os.getenv("GEMINI_FAST_MODEL", DEFAULT_GEMINI_FAST_MODEL).strip()
        or DEFAULT_GEMINI_FAST_MODEL
    )


def groq_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def gemini_cooldown_seconds() -> int:
    return env_int("GEMINI_COOLDOWN_SECONDS", DEFAULT_GEMINI_COOLDOWN_SECONDS)


def request_timeout_seconds() -> int:
    return env_int("LLM_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def provider_key(provider: str) -> str:
    env_name = PROVIDER_KEY_ENV.get(provider)
    if env_name is None:
        return ""
    return os.getenv(env_name, "").strip()


def provider_configured(provider: str) -> bool:
    return bool(provider_key(provider))


def model_stability(model: str) -> str:
    """Coarse stability class for a model id: 'production' or 'preview'."""
    lowered = model.strip().casefold()
    if any(marker in lowered for marker in PREVIEW_MODEL_MARKERS):
        return "preview"
    return "production"


def validate_fixed_groq_endpoint(url: str, expected_path: str) -> None:
    """Fail closed unless url is exactly the expected HTTPS Groq endpoint."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != GROQ_API_HOST
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise LLMConfigurationError(
            f"The configured Groq endpoint failed the fixed HTTPS allowlist: {expected_path}"
        )
