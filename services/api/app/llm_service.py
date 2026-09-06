from __future__ import annotations

import logging
from typing import Mapping

from . import model_gateway
from .provider_config import (  # noqa: F401  (re-exported compatibility surface)
    DEFAULT_GEMINI_COOLDOWN_SECONDS,
    DEFAULT_GEMINI_FAST_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    GROQ_API_HOST,
    GROQ_CHAT_COMPLETIONS_URL,
    GROQ_MODELS_URL,
    RECOVERABLE_GEMINI_CODES,
    LLMConfigurationError,
    LLMResult,
    LLMServiceError,
    LLMUnavailableError,
    Provider,
    gemini_cooldown_seconds,
    gemini_fast_model_name,
    gemini_model_name,
    groq_model_name,
    request_timeout_seconds,
)

logger = logging.getLogger(__name__)

# Part 10.2 Phase E: llm_service is now a thin compatibility facade.
#
# Provider selection, client reuse, circuit state, degraded mode and preflight
# live in model_gateway; names, defaults and env policy live in provider_config.
# This module exists so every existing caller (brain_agent, gmail_service,
# cross_tool_*, orchestrator, tests) keeps working unchanged while there is
# exactly one provider-routing authority underneath.
#
# The three public generation entry points are preserved by contract:
#   generate_text       -> gateway "balanced"  (Gemini primary, Groq fallback)
#   generate_fast_text  -> gateway "fast"      (Flash-Lite, Groq fallback)
#   generate_groq_text  -> gateway "groq_only"
#
# Each call performs at most one successful provider request; the gateway issues
# another only when the previous candidate actually failed.

# Circuit-breaker surface, delegated so there is a single source of truth.
gemini_cooldown_active = model_gateway.gemini_cooldown_active
gemini_cooldown_remaining_seconds = model_gateway.gemini_cooldown_remaining_seconds
activate_gemini_cooldown = model_gateway.activate_gemini_cooldown
clear_gemini_cooldown = model_gateway.clear_gemini_cooldown


def generate_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    """Quality-first generation: Gemini primary with automatic Groq failover.

    `response_schema` (Part 10.2 Phase H) opts into provider-native structured
    output. It is optional so every existing caller is unaffected.
    """
    return model_gateway.generate(
        system_instruction=system_instruction,
        user_content=user_content,
        profile_name="balanced",
        temperature=temperature,
        response_schema=response_schema,
    )


def generate_fast_text(
    *,
    system_instruction: str,
    user_content: str,
    response_schema: Mapping[str, object] | None = None,
) -> LLMResult:
    """Low-latency conversational generation with safe cloud failover."""
    return model_gateway.generate(
        system_instruction=system_instruction,
        user_content=user_content,
        profile_name="fast",
        response_schema=response_schema,
    )


def generate_groq_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    """Groq-only generation.

    Retained because callers asked for Groq explicitly. New code should request a
    gateway profile instead, so provider order stays policy rather than a
    call-site decision.
    """
    return model_gateway.generate(
        system_instruction=system_instruction,
        user_content=user_content,
        profile_name="groq_only",
        temperature=temperature,
    )


def generate_gemini_text(
    *,
    system_instruction: str,
    user_content: str,
    temperature: float = 0.35,
) -> LLMResult:
    """Legacy Gemini-first entry point with Groq failover.

    Prompt 7 originally used this as Gemini-only for Gmail drafting. Bunnelby's
    free-tier reliability policy allows the same failover as normal chat; the
    approval and send safety paths are independent of provider selection.
    """
    return generate_text(
        system_instruction=system_instruction,
        user_content=user_content,
        temperature=temperature,
    )


__all__ = [
    "DEFAULT_GEMINI_COOLDOWN_SECONDS",
    "DEFAULT_GEMINI_FAST_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "GROQ_API_HOST",
    "GROQ_CHAT_COMPLETIONS_URL",
    "GROQ_MODELS_URL",
    "RECOVERABLE_GEMINI_CODES",
    "LLMConfigurationError",
    "LLMResult",
    "LLMServiceError",
    "LLMUnavailableError",
    "Provider",
    "activate_gemini_cooldown",
    "clear_gemini_cooldown",
    "gemini_cooldown_active",
    "gemini_cooldown_remaining_seconds",
    "gemini_cooldown_seconds",
    "gemini_fast_model_name",
    "gemini_model_name",
    "generate_fast_text",
    "generate_gemini_text",
    "generate_groq_text",
    "generate_text",
    "groq_model_name",
    "request_timeout_seconds",
]
