from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Final, Literal, Mapping, Protocol

from google import genai
from google.genai import errors, types

from .provider_config import (
    GROQ_CHAT_COMPLETIONS_URL,
    RECOVERABLE_GEMINI_CODES,
    LLMConfigurationError,
    LLMResult,
    LLMServiceError,
    LLMUnavailableError,
    env_int,
    gemini_cooldown_seconds,
    gemini_fast_model_name,
    gemini_model_name,
    groq_model_name,
    model_stability,
    provider_configured,
    provider_key,
    request_timeout_seconds,
    validate_fixed_groq_endpoint,
)
from .provider_health import ProviderHealth, check_provider_model, redact_secrets

logger = logging.getLogger(__name__)

# Part 10.2 Phase E: the Model Gateway.
#
# One place owns provider selection, client reuse, circuit state and degraded
# mode. Callers ask for an INFERENCE PROFILE ("fast conversational turn",
# "low-latency synthesis") and the gateway decides which provider/model serves
# it. Before this module, provider order was hard-coded in three different
# places, cross_tool_fastpath had its own Groq-first policy, only Gemini had a
# breaker, and a brand-new Gemini client was constructed per request.
#
# NOT in scope: loading a real local model. LocalReasoningProvider below is an
# interface plus state so Part 27 can attach Ollama without the Brain or the
# executor changing shape.

InferenceProfileName = Literal["balanced", "fast", "low_latency_synthesis", "groq_only"]
GatewayMode = Literal["cloud", "degraded_local", "unavailable"]

DEFAULT_BREAKER_THRESHOLD: Final[int] = 1
DEFAULT_GROQ_COOLDOWN_SECONDS: Final[int] = 120
MAX_TRACKED_LATENCIES: Final[int] = 20


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Circuit state
# --------------------------------------------------------------------------- #


@dataclass
class ProviderState:
    """Live circuit state for one (provider, model) pair."""

    provider: str
    model: str
    stability: str = "production"
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    cooldown_until_monotonic: float = 0.0
    last_error_code: str | None = None
    last_error_detail: str = ""
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    recent_latencies_ms: list[float] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def cooldown_remaining_seconds(self) -> int:
        return max(0, int(self.cooldown_until_monotonic - time.monotonic()))

    def is_open(self) -> bool:
        """True when the breaker is open, i.e. this pair must be skipped."""
        return time.monotonic() < self.cooldown_until_monotonic

    def average_latency_ms(self) -> float | None:
        if not self.recent_latencies_ms:
            return None
        return sum(self.recent_latencies_ms) / len(self.recent_latencies_ms)

    def snapshot(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "stability": self.stability,
            "healthy": not self.is_open(),
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds(),
            "last_error_code": self.last_error_code,
            "last_error_detail": self.last_error_detail,
            "average_latency_ms": self.average_latency_ms(),
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
        }


class ProviderRegistry:
    """Thread-safe circuit-breaker store keyed by (provider, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str], ProviderState] = {}

    def state(self, provider: str, model: str) -> ProviderState:
        key = (provider, model)
        with self._lock:
            existing = self._states.get(key)
            if existing is None:
                existing = ProviderState(
                    provider=provider, model=model, stability=model_stability(model)
                )
                self._states[key] = existing
            return existing

    def record_success(self, provider: str, model: str, latency_ms: float) -> None:
        state = self.state(provider, model)
        with self._lock:
            state.consecutive_failures = 0
            state.total_successes += 1
            state.cooldown_until_monotonic = 0.0
            state.last_error_code = None
            state.last_error_detail = ""
            state.last_success_at = _now()
            state.recent_latencies_ms.append(latency_ms)
            del state.recent_latencies_ms[:-MAX_TRACKED_LATENCIES]

    def record_failure(
        self,
        provider: str,
        model: str,
        *,
        error_code: str,
        detail: str,
        cooldown_seconds: int,
        threshold: int = DEFAULT_BREAKER_THRESHOLD,
    ) -> None:
        state = self.state(provider, model)
        safe_detail = redact_secrets(detail)
        with self._lock:
            state.consecutive_failures += 1
            state.total_failures += 1
            state.last_error_code = error_code
            state.last_error_detail = safe_detail
            state.last_failure_at = _now()
            if state.consecutive_failures >= threshold and cooldown_seconds > 0:
                state.cooldown_until_monotonic = max(
                    state.cooldown_until_monotonic, time.monotonic() + cooldown_seconds
                )
        logger.warning(
            "Model gateway breaker: %s/%s failure=%s cooldown=%ss detail=%s",
            provider,
            model,
            error_code,
            cooldown_seconds,
            safe_detail,
        )

    def clear(self, provider: str | None = None, model: str | None = None) -> None:
        with self._lock:
            for key, state in self._states.items():
                if provider is not None and key[0] != provider:
                    continue
                if model is not None and key[1] != model:
                    continue
                state.consecutive_failures = 0
                state.cooldown_until_monotonic = 0.0
                state.last_error_code = None
                state.last_error_detail = ""

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def snapshots(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            states = list(self._states.values())
        return tuple(state.snapshot() for state in states)


REGISTRY: Final[ProviderRegistry] = ProviderRegistry()


# --------------------------------------------------------------------------- #
# Local reasoning provider: interface + state only (Part 27 attaches a model)
# --------------------------------------------------------------------------- #


class LocalReasoningProvider(Protocol):
    """Contract a local reasoning backend must satisfy to join the gateway.

    Implementing this later (Ollama, Foundry Local, Windows ML) requires no
    change to the Brain, the executor or any caller: the gateway simply gains
    another candidate in its profile chains.
    """

    name: str

    def available(self) -> bool: ...

    def generate(
        self, system_instruction: str, user_content: str, temperature: float
    ) -> LLMResult: ...


@dataclass(frozen=True)
class UnconfiguredLocalProvider:
    """Placeholder local provider. Declares the seam without loading a model."""

    name: str = "local"

    def available(self) -> bool:
        return False

    def generate(
        self, system_instruction: str, user_content: str, temperature: float
    ) -> LLMResult:
        raise LLMConfigurationError(
            "No local reasoning provider is installed. Bunnelby runs cloud-first; "
            "a local fallback arrives with Part 27 Local AI Runtime."
        )


_local_provider: LocalReasoningProvider = UnconfiguredLocalProvider()


def register_local_provider(provider: LocalReasoningProvider) -> None:
    global _local_provider
    _local_provider = provider


def local_provider() -> LocalReasoningProvider:
    return _local_provider


# --------------------------------------------------------------------------- #
# Cached provider clients
# --------------------------------------------------------------------------- #

_client_lock = threading.Lock()
_gemini_clients: dict[str, genai.Client] = {}


def gemini_client() -> genai.Client:
    """Return a cached Gemini client for the configured key.

    Previously every request built and closed a fresh client, paying a new TLS
    handshake on each conversational turn. The client is keyed by API key so a
    rotated key produces a new client rather than reusing a stale one.
    """
    api_key = provider_key("gemini")
    if not api_key:
        raise LLMConfigurationError("GEMINI_API_KEY is not configured.")
    with _client_lock:
        client = _gemini_clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _gemini_clients[api_key] = client
        return client


def reset_clients() -> None:
    """Drop cached clients (used by tests and after a key rotation)."""
    with _client_lock:
        clients = list(_gemini_clients.values())
        _gemini_clients.clear()
    for client in clients:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Cached Gemini client close failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Provider call primitives
# --------------------------------------------------------------------------- #


def _gemini_call(
    model: str,
    system_instruction: str,
    user_content: str,
    temperature: float,
    *,
    minimal_thinking: bool,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    client = gemini_client()
    # Part 10.2 Phase H: when a caller supplies a schema, use provider-native
    # structured output instead of asking for JSON in prose. The model then
    # cannot emit a mode/tool value outside the enum.
    structured = (
        {"response_mime_type": "application/json", "response_schema": dict(response_schema)}
        if response_schema
        else {}
    )
    if minimal_thinking:
        config: object = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            **structured,
        )
    else:
        config = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            **structured,
        }

    try:
        response = client.models.generate_content(
            model=model, contents=user_content, config=config
        )
    except errors.APIError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        recoverable = code in RECOVERABLE_GEMINI_CODES
        raise _ProviderCallError(
            error_code=f"http_{code or 'unknown'}",
            message=f"Gemini request failed (HTTP {code or 'unknown'}).",
            recoverable=recoverable,
            cooldown_seconds=gemini_cooldown_seconds() if recoverable else 0,
        ) from exc
    except LLMServiceError:
        raise
    except Exception as exc:
        # Transport failures are safe to fail over and worth a cooldown so
        # repeated turns do not stall on the same failing primary.
        raise _ProviderCallError(
            error_code="transport",
            message="Gemini request failed before a usable response arrived.",
            recoverable=True,
            cooldown_seconds=gemini_cooldown_seconds(),
        ) from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise _ProviderCallError(
            error_code="empty_response",
            message="Gemini returned an empty response.",
            recoverable=True,
            cooldown_seconds=0,
        )
    return LLMResult(text=text, provider="gemini", model=model)


def _groq_error_message(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
        message = payload.get("error", {}).get("message")
        if message:
            return str(message)[:400]
    except Exception:
        pass
    return ""


def _groq_call(
    model: str,
    system_instruction: str,
    user_content: str,
    temperature: float,
    *,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    api_key = provider_key("groq")
    if not api_key:
        raise LLMConfigurationError("GROQ_API_KEY is not configured.")

    validate_fixed_groq_endpoint(
        GROQ_CHAT_COMPLETIONS_URL, "/openai/v1/chat/completions"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if response_schema:
        # Groq exposes JSON mode rather than a full schema; the envelope is
        # still validated by _parse_decision and tool_requests afterwards.
        payload["response_format"] = {"type": "json_object"}
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
        # B310 is suppressed only after the strict scheme/host/path validation
        # above; the URL is a compile-time constant.
        with urllib.request.urlopen(  # nosec B310
            request, timeout=request_timeout_seconds()
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = _groq_error_message(exc.read())
        suffix = f": {detail}" if detail else ""
        raise _ProviderCallError(
            error_code=f"http_{exc.code}",
            message=f"Groq request failed (HTTP {exc.code}){suffix}",
            recoverable=exc.code in RECOVERABLE_GEMINI_CODES,
            cooldown_seconds=(
                groq_cooldown_seconds() if exc.code in RECOVERABLE_GEMINI_CODES else 0
            ),
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise _ProviderCallError(
            error_code="transport",
            message="Groq is temporarily unreachable.",
            recoverable=True,
            cooldown_seconds=groq_cooldown_seconds(),
        ) from exc

    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _ProviderCallError(
            error_code="invalid_payload",
            message="Groq returned an invalid response payload.",
            recoverable=True,
            cooldown_seconds=0,
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise _ProviderCallError(
            error_code="empty_response",
            message="Groq returned no choices.",
            recoverable=True,
            cooldown_seconds=0,
        )
    text = str(choices[0].get("message", {}).get("content", "")).strip()
    if not text:
        raise _ProviderCallError(
            error_code="empty_response",
            message="Groq returned an empty response.",
            recoverable=True,
            cooldown_seconds=0,
        )
    return LLMResult(text=text, provider="groq", model=model)


def groq_cooldown_seconds() -> int:
    return env_int("GROQ_COOLDOWN_SECONDS", DEFAULT_GROQ_COOLDOWN_SECONDS)


class _ProviderCallError(LLMUnavailableError):
    """Internal: one provider attempt failed, carrying breaker metadata."""

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        recoverable: bool,
        cooldown_seconds: int,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable
        self.cooldown_seconds = cooldown_seconds


# --------------------------------------------------------------------------- #
# Inference profiles: the routing policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderCandidate:
    provider: str
    model_selector: Callable[[], str]
    temperature: float
    minimal_thinking: bool = False

    def model(self) -> str:
        return self.model_selector()


@dataclass(frozen=True)
class InferenceProfile:
    """An ordered provider policy for one class of request."""

    name: str
    description: str
    candidates: tuple[ProviderCandidate, ...]


def _profiles() -> Mapping[str, InferenceProfile]:
    """Build profiles fresh so env model overrides are always honored."""
    gemini_balanced = ProviderCandidate("gemini", gemini_model_name, 0.35)
    gemini_fast = ProviderCandidate(
        "gemini", gemini_fast_model_name, 0.2, minimal_thinking=True
    )
    groq_balanced = ProviderCandidate("groq", groq_model_name, 0.35)
    groq_fast = ProviderCandidate("groq", groq_model_name, 0.2)

    return {
        "balanced": InferenceProfile(
            name="balanced",
            description="Quality-first reasoning: Gemini primary, Groq fallback.",
            candidates=(gemini_balanced, groq_balanced),
        ),
        "fast": InferenceProfile(
            name="fast",
            description="Latency-first conversational turn: Gemini Flash-Lite, Groq fallback.",
            candidates=(gemini_fast, groq_fast),
        ),
        "low_latency_synthesis": InferenceProfile(
            name="low_latency_synthesis",
            description=(
                "Verbalize already-verified structured results. Reads, permissions and "
                "grounding are complete, so latency outranks depth: Groq first, Gemini "
                "as the quality-preserving fallback."
            ),
            candidates=(groq_fast, gemini_balanced),
        ),
        "groq_only": InferenceProfile(
            name="groq_only",
            description="Explicit Groq-only path retained for compatibility callers.",
            candidates=(groq_balanced,),
        ),
    }


def profile(name: str) -> InferenceProfile:
    profiles = _profiles()
    try:
        return profiles[name]
    except KeyError as exc:
        raise LLMConfigurationError(f"Unknown inference profile: {name}") from exc


def profile_names() -> tuple[str, ...]:
    return tuple(_profiles())


# --------------------------------------------------------------------------- #
# Gateway mode
# --------------------------------------------------------------------------- #


def gateway_mode() -> GatewayMode:
    """Report whether cloud reasoning is usable, or why it is not.

    'cloud'          at least one configured cloud pair has a closed breaker
    'degraded_local' no usable cloud pair, but a local provider is available
    'unavailable'    nothing can serve a reasoning request right now
    """
    for candidate in profile("balanced").candidates + profile("fast").candidates:
        if not provider_configured(candidate.provider):
            continue
        if not REGISTRY.state(candidate.provider, candidate.model()).is_open():
            return "cloud"
    if local_provider().available():
        return "degraded_local"
    return "unavailable"


def provider_status() -> dict[str, object]:
    """Structured gateway state for diagnostics and the future Reality Layer."""
    return {
        "mode": gateway_mode(),
        "checked_at": _now(),
        "local_provider": {
            "name": local_provider().name,
            "available": local_provider().available(),
        },
        "profiles": {
            name: [
                {
                    "provider": candidate.provider,
                    "model": candidate.model(),
                    "configured": provider_configured(candidate.provider),
                    "stability": model_stability(candidate.model()),
                    "breaker_open": REGISTRY.state(
                        candidate.provider, candidate.model()
                    ).is_open(),
                }
                for candidate in spec.candidates
            ]
            for name, spec in _profiles().items()
        },
        "circuits": REGISTRY.snapshots(),
    }


def preflight(*, listers: Mapping[str, object] | None = None) -> tuple[ProviderHealth, ...]:
    """Read-only availability preflight for every profile candidate.

    Reuses provider_health rather than duplicating model-list logic.
    """
    seen: set[tuple[str, str]] = set()
    records: list[ProviderHealth] = []
    for spec in _profiles().values():
        for candidate in spec.candidates:
            key = (candidate.provider, candidate.model())
            if key in seen:
                continue
            seen.add(key)
            lister = None
            if listers is not None:
                lister = listers.get(candidate.provider)  # type: ignore[assignment]
            records.append(
                check_provider_model(candidate.provider, candidate.model(), lister=lister)  # type: ignore[arg-type]
            )
    return tuple(records)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _attempt(
    candidate: ProviderCandidate,
    system_instruction: str,
    user_content: str,
    temperature: float | None,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    model = candidate.model()
    used_temperature = candidate.temperature if temperature is None else temperature
    started = time.perf_counter()

    if candidate.provider == "gemini":
        result = _gemini_call(
            model,
            system_instruction,
            user_content,
            used_temperature,
            minimal_thinking=candidate.minimal_thinking,
            response_schema=response_schema,
        )
    elif candidate.provider == "groq":
        result = _groq_call(
            model,
            system_instruction,
            user_content,
            used_temperature,
            response_schema=response_schema,
        )
    else:
        raise LLMConfigurationError(f"Unsupported gateway provider: {candidate.provider}")

    REGISTRY.record_success(
        candidate.provider, model, (time.perf_counter() - started) * 1000.0
    )
    return result


def generate(
    *,
    system_instruction: str,
    user_content: str,
    profile_name: str = "balanced",
    temperature: float | None = None,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    """Run one generation through the profile's provider chain.

    Exactly one successful provider call is made per invocation; the gateway
    never issues a second call for the same request beyond failing over.
    """
    spec = profile(profile_name)
    failures: list[str] = []
    configured_any = False

    for candidate in spec.candidates:
        model = candidate.model()
        if not provider_configured(candidate.provider):
            failures.append(f"{candidate.provider} is not configured.")
            continue
        configured_any = True

        state = REGISTRY.state(candidate.provider, model)
        if state.is_open():
            failures.append(
                f"{candidate.provider}/{model} breaker open "
                f"({state.cooldown_remaining_seconds()}s remaining)."
            )
            continue

        try:
            result = _attempt(
                candidate, system_instruction, user_content, temperature, response_schema
            )
            logger.info(
                "Model gateway profile=%s provider=%s model=%s",
                spec.name,
                result.provider,
                result.model,
            )
            return result
        except _ProviderCallError as exc:
            REGISTRY.record_failure(
                candidate.provider,
                model,
                error_code=exc.error_code,
                detail=str(exc),
                cooldown_seconds=exc.cooldown_seconds,
            )
            failures.append(redact_secrets(exc))
        except LLMConfigurationError as exc:
            failures.append(redact_secrets(exc))
        except LLMServiceError as exc:
            REGISTRY.record_failure(
                candidate.provider,
                model,
                error_code="service_error",
                detail=str(exc),
                cooldown_seconds=0,
            )
            failures.append(redact_secrets(exc))

    if not configured_any:
        raise LLMConfigurationError(
            "No cloud LLM provider is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY."
        )

    provider = local_provider()
    if provider.available():
        logger.warning(
            "Model gateway entering degraded local mode after cloud failures: %s",
            "; ".join(failures),
        )
        return provider.generate(
            system_instruction, user_content, 0.2 if temperature is None else temperature
        )

    raise LLMUnavailableError("; ".join(failures) or "No provider could serve the request.")


# --------------------------------------------------------------------------- #
# Legacy cooldown surface (llm_service re-exports these)
# --------------------------------------------------------------------------- #


def gemini_cooldown_active() -> bool:
    return REGISTRY.state("gemini", gemini_model_name()).is_open() or REGISTRY.state(
        "gemini", gemini_fast_model_name()
    ).is_open()


def gemini_cooldown_remaining_seconds() -> int:
    return max(
        REGISTRY.state("gemini", gemini_model_name()).cooldown_remaining_seconds(),
        REGISTRY.state("gemini", gemini_fast_model_name()).cooldown_remaining_seconds(),
    )


def activate_gemini_cooldown(reason: str) -> None:
    cooldown = gemini_cooldown_seconds()
    for model in (gemini_model_name(), gemini_fast_model_name()):
        REGISTRY.record_failure(
            "gemini",
            model,
            error_code="manual",
            detail=reason,
            cooldown_seconds=cooldown,
        )


def clear_gemini_cooldown() -> None:
    REGISTRY.clear(provider="gemini")


def candidate_models(profile_name: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (candidate.provider, candidate.model())
        for candidate in profile(profile_name).candidates
    )


__all__ = [
    "GatewayMode",
    "InferenceProfile",
    "LocalReasoningProvider",
    "ProviderCandidate",
    "ProviderRegistry",
    "ProviderState",
    "REGISTRY",
    "UnconfiguredLocalProvider",
    "activate_gemini_cooldown",
    "candidate_models",
    "clear_gemini_cooldown",
    "gateway_mode",
    "gemini_client",
    "gemini_cooldown_active",
    "gemini_cooldown_remaining_seconds",
    "generate",
    "groq_cooldown_seconds",
    "local_provider",
    "preflight",
    "profile",
    "profile_names",
    "provider_status",
    "register_local_provider",
    "reset_clients",
]
