from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from .database import PROJECT_ROOT

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")

Provider = Literal["gemini", "groq"]

DEFAULT_GEMINI_MODEL: Final[str] = "gemini-3.6-flash"
DEFAULT_GEMINI_FAST_MODEL: Final[str] = "gemini-3.5-flash-lite"
DEFAULT_GROQ_MODEL: Final[str] = "openai/gpt-oss-120b"
DEFAULT_GEMINI_COOLDOWN_SECONDS: Final[int] = 900
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[int] = 45
GROQ_CHAT_COMPLETIONS_URL: Final[str] = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL: Final[str] = "https://api.groq.com/openai/v1/models"
GROQ_API_HOST: Final[str] = "api.groq.com"
RECOVERABLE_GEMINI_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: Provider
    model: str


class LLMServiceError(RuntimeError):
    """Base exception for AO language-model provider failures."""


class LLMConfigurationError(LLMServiceError):
    """Raised when no configured provider can serve a request."""


class LLMUnavailableError(LLMServiceError):
    """Raised when configured providers are temporarily unavailable."""


_gemini_lock = threading.Lock()
_gemini_unavailable_until: float = 0.0
_gemini_last_failure: str = ""


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _validate_fixed_groq_endpoint() -> None:
    """Fail closed if the compile-time Groq endpoint ever stops being exact HTTPS."""
    parsed = urlsplit(GROQ_CHAT_COMPLETIONS_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != GROQ_API_HOST
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/openai/v1/chat/completions"
        or parsed.query
        or parsed.fragment
    ):
        raise LLMConfigurationError("The configured Groq endpoint failed the fixed HTTPS allowlist.")


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
    return _env_int("GEMINI_COOLDOWN_SECONDS", DEFAULT_GEMINI_COOLDOWN_SECONDS)


def request_timeout_seconds() -> int:
    return _env_int("LLM_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def gemini_cooldown_active() -> bool:
    with _gemini_lock:
        return time.monotonic() < _gemini_unavailable_until


def gemini_cooldown_remaining_seconds() -> int:
    with _gemini_lock:
        remaining = _gemini_unavailable_until - time.monotonic()
    return max(0, int(remaining))


def activate_gemini_cooldown(reason: str) -> None:
    global _gemini_unavailable_until, _gemini_last_failure
    cooldown = gemini_cooldown_seconds()
    with _gemini_lock:
        _gemini_unavailable_until = max(
            _gemini_unavailable_until,
            time.monotonic() + cooldown,
        )
        _gemini_last_failure = reason[:300]
    logger.warning("Gemini temporarily bypassed for %ss: %s", cooldown, reason)


def clear_gemini_cooldown() -> None:
    global _gemini_unavailable_until, _gemini_last_failure
    with _gemini_lock:
        _gemini_unavailable_until = 0.0
        _gemini_last_failure = ""


def _gemini_generate(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError("GEMINI_API_KEY is not configured.")

    model = gemini_model_name()
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_content,
            config={"system_instruction": system_instruction, "temperature": temperature},
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LLMUnavailableError("Gemini returned an empty response.")
        return LLMResult(text=text, provider="gemini", model=model)
    except errors.APIError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code in RECOVERABLE_GEMINI_CODES:
            activate_gemini_cooldown(f"HTTP {code or 'unknown'}")
            raise LLMUnavailableError(
                f"Gemini is temporarily unavailable (HTTP {code or 'unknown'})."
            ) from exc
        raise LLMUnavailableError(f"Gemini request failed (HTTP {code or 'unknown'}).") from exc
    except LLMServiceError:
        raise
    except Exception as exc:
        # Network/transport failures are safe to fail over. Use a short provider cooldown
        # so repeated requests do not stall on the same failing primary provider.
        activate_gemini_cooldown(type(exc).__name__)
        raise LLMUnavailableError("Gemini request failed before a usable response arrived.") from exc
    finally:
        client.close()


def _gemini_generate_fast(
    system_instruction: str,
    user_content: str,
) -> LLMResult:
    """Latency-first Gemini path for ordinary conversational turns.

    Uses Flash-Lite with minimal reasoning. Complex reasoning continues through
    the established balanced Gemini path instead of sacrificing quality.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError("GEMINI_API_KEY is not configured.")

    model = gemini_fast_model_name()
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),
            ),
        )

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LLMUnavailableError("Fast Gemini returned an empty response.")

        return LLMResult(
            text=text,
            provider="gemini",
            model=model,
        )

    except errors.APIError as exc:
        code = int(getattr(exc, "code", 0) or 0)

        if code in RECOVERABLE_GEMINI_CODES:
            activate_gemini_cooldown(
                f"fast Gemini HTTP {code or 'unknown'}"
            )
            raise LLMUnavailableError(
                f"Fast Gemini is temporarily unavailable "
                f"(HTTP {code or 'unknown'})."
            ) from exc

        raise LLMUnavailableError(
            f"Fast Gemini request failed (HTTP {code or 'unknown'})."
        ) from exc

    except LLMServiceError:
        raise

    except Exception as exc:
        activate_gemini_cooldown(
            f"fast Gemini {type(exc).__name__}"
        )
        raise LLMUnavailableError(
            "Fast Gemini request failed before a usable response arrived."
        ) from exc

    finally:
        client.close()


def _groq_error_message(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
        message = payload.get("error", {}).get("message")
        if message:
            return str(message)[:400]
    except Exception:
        pass
    return ""


def generate_gemini_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    """Legacy Gemini-first entry point with Groq failover.

    Prompt 7 originally used this function as Gemini-only for Gmail drafting.
    Bunnelby's free-tier reliability policy now allows the same Gemini→Groq
    failover used by normal chat and Gmail summarization. Approval and send
    safety remain completely independent from model-provider selection.
    """
    return generate_text(
        system_instruction=system_instruction,
        user_content=user_content,
        temperature=temperature,
    )


def generate_groq_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError("GROQ_API_KEY is not configured.")

    _validate_fixed_groq_endpoint()
    model = groq_model_name()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AO-Desktop/1.0",
        },
        method="POST",
    )

    try:
        # B310 is suppressed only after strict scheme/host/path validation above; the URL is
        # a compile-time constant and cannot be supplied or redirected by user input here.
        with urllib.request.urlopen(  # nosec B310
            request,
            timeout=request_timeout_seconds(),
        ) as response:
            body = response.read()
        data = json.loads(body.decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise LLMUnavailableError("Groq returned no choices.")
        text = str(choices[0].get("message", {}).get("content", "")).strip()
        if not text:
            raise LLMUnavailableError("Groq returned an empty response.")
        return LLMResult(text=text, provider="groq", model=model)
    except urllib.error.HTTPError as exc:
        detail = _groq_error_message(exc.read())
        suffix = f": {detail}" if detail else ""
        raise LLMUnavailableError(f"Groq request failed (HTTP {exc.code}){suffix}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMUnavailableError("Groq is temporarily unreachable.") from exc
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError("Groq returned an invalid response payload.") from exc
    except LLMServiceError:
        raise
    except Exception as exc:
        raise LLMUnavailableError("Groq request failed before a usable response arrived.") from exc


def generate_fast_text(
    *,
    system_instruction: str,
    user_content: str,
) -> LLMResult:
    """Low-latency conversational generation with safe cloud failover."""
    failures: list[str] = []

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if gemini_key and not gemini_cooldown_active():
        try:
            result = _gemini_generate_fast(
                system_instruction,
                user_content,
            )
            clear_gemini_cooldown()
            logger.info(
                "AO LLM profile=fast provider=gemini model=%s",
                result.model,
            )
            return result
        except LLMServiceError as exc:
            failures.append(str(exc))
            logger.warning(
                "Fast Gemini generation failed; attempting Groq fallback: %s",
                exc,
            )

    elif gemini_key:
        failures.append(
            f"Gemini cooldown active "
            f"({gemini_cooldown_remaining_seconds()}s remaining)."
        )
    else:
        failures.append("Gemini is not configured.")

    if groq_key:
        try:
            result = generate_groq_text(
                system_instruction=system_instruction,
                user_content=user_content,
                temperature=0.2,
            )
            logger.info(
                "AO LLM profile=fast-fallback provider=groq model=%s",
                result.model,
            )
            return result
        except LLMServiceError as exc:
            failures.append(str(exc))

    if not gemini_key and not groq_key:
        raise LLMConfigurationError(
            "No cloud LLM provider is configured."
        )

    raise LLMUnavailableError("; ".join(failures))


def generate_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    """Gemini-first generation with automatic Groq failover.

    Gemini remains the primary provider. When Gemini is in cooldown after a quota,
    server, timeout, or transport failure, requests skip directly to Groq until the
    cooldown expires. Once it expires, AO probes Gemini again automatically.
    """

    failures: list[str] = []
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if gemini_key and not gemini_cooldown_active():
        try:
            result = _gemini_generate(
                system_instruction,
                user_content,
                temperature=temperature,
            )
            # Successful primary response means any previous cooldown can be cleared.
            clear_gemini_cooldown()
            logger.info("AO LLM provider=gemini model=%s", result.model)
            return result
        except LLMServiceError as exc:
            failures.append(str(exc))
            logger.warning("Gemini generation failed; attempting Groq fallback: %s", exc)
    elif gemini_key:
        failures.append(
            f"Gemini cooldown active ({gemini_cooldown_remaining_seconds()}s remaining)."
        )
    else:
        failures.append("Gemini is not configured.")

    if groq_key:
        try:
            result = generate_groq_text(
                system_instruction=system_instruction,
                user_content=user_content,
                temperature=temperature,
            )
            logger.info("AO LLM provider=groq model=%s", result.model)
            return result
        except LLMServiceError as exc:
            failures.append(str(exc))
            logger.warning("Groq fallback failed: %s", exc)
    else:
        failures.append("Groq is not configured.")

    if not gemini_key and not groq_key:
        raise LLMConfigurationError(
            "No cloud LLM provider is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY."
        )

    raise LLMUnavailableError("; ".join(failures))
