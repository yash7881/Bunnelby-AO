from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Final, Mapping

from . import audit_service, tool_execution, verification_service
from .brain_agent import BrainDecision
from .capability_registry import Capability, CapabilityRegistryError, registry
from .cross_tool_fastpath import handle_cross_tool_fast_request
from .cross_tool_reasoning import CrossToolWriteNotSupportedError
from .orchestrator import OrchestratorResult
from .risk_policy import (
    ApprovalPolicy,
    AuditPolicy,
    FreshnessPolicy,
    RiskDecision,
    RiskLevel,
)
from .tool_requests import (
    CalendarCreateRequest,
    CalendarReadRequest,
    CrossToolReadRequest,
    GeneralAnswerRequest,
    GmailComposeRequest,
    GmailReadRequest,
    GmailReplyRequest,
    ToolRequest,
    ToolRequestValidationError,
    build_request,
)

logger = logging.getLogger(__name__)

# Part 10.2 Phase G: the typed execution boundary.
#
# Before: execute() read only `decision.tool` and re-passed the raw
# user_message, so four of six Brain tool values collapsed into two identical
# calls and every handler re-derived its own action class from wording.
#
# Now: the Brain's decision is validated into a typed ToolRequest, the
# Capability Registry resolves the executor FROM THE REQUEST TYPE, and each
# executor serves exactly one action class. `decision.arguments` is real
# execution data -- it selects unread-vs-recent, the calendar read submode, the
# event title, the requested cross-tool sources.

# `handle_cross_tool_fast_request` and `CrossToolWriteNotSupportedError` stay
# bound here: tool_execution.execute_cross_tool_read resolves them through this
# module, which keeps the established test seam and one definition.
__all__ = [
    "CrossToolWriteNotSupportedError",
    "build_capabilities",
    "execute",
    "execute_answer",
    "execute_request",
    "handle_cross_tool_fast_request",
]


def _late(executor_name: str):
    """Bind a tool_execution executor by name, resolved at call time.

    Late binding keeps `tool_execution.<executor>` a real seam: patching the
    owning module affects dispatch, the same convention the rest of the codebase
    uses (patch.object(brain_agent, "decide"), patch.object(gmail_service, ...)).
    Capturing the function object at registration time would silently bypass it.
    """

    def _run(request: ToolRequest):
        return getattr(tool_execution, executor_name)(request)

    _run.__name__ = f"late_{executor_name}"
    _run.__qualname__ = _run.__name__
    return _run


def build_capabilities() -> tuple[Capability, ...]:
    """Declare every top-level capability the Brain may select.

    Risk and approval semantics are declared here once and validated at
    registration time by risk_policy.validate_declaration, so an unsafe
    declaration fails at import rather than when a user asks for a write.
    """
    return (
        Capability(
            name="general_answer",
            version="1.0",
            description="Answer conversationally or ask for clarification. Touches no external system.",
            request_model=GeneralAnswerRequest,
            risk_level=RiskLevel.L0_OBSERVE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=_late("execute_general_answer"),
            freshness_policy=FreshnessPolicy.CACHED_OK,
            audit_policy=AuditPolicy.METADATA_ONLY,
            selection_guidance=(
                "Default. Ordinary conversation, explanations, opinions, and any question "
                "ABOUT Gmail/Calendar as products rather than a request to use them."
            ),
            examples=(
                "What is a vector database?",
                "Explain the difference between Gmail and Google Calendar.",
                "my brother is too lazy",
            ),
        ),
        Capability(
            name="gmail_read",
            version="1.0",
            description="Read, list or summarize the user's real Gmail inbox. Read-only.",
            request_model=GmailReadRequest,
            risk_level=RiskLevel.L0_OBSERVE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=_late("execute_gmail_read"),
            selection_guidance=(
                "The user wants their actual inbox read. Set read_kind='unread' when they "
                "ask specifically about unread mail."
            ),
            examples=("Check my latest emails.", "Any unread mail?"),
        ),
        Capability(
            name="gmail_compose",
            version="1.0",
            description="Draft a brand-new email for explicit approval. Never sends.",
            request_model=GmailComposeRequest,
            risk_level=RiskLevel.L3_EXTERNAL_WRITE,
            approval_policy=ApprovalPolicy.ALWAYS,
            executor=_late("execute_gmail_compose"),
            selection_guidance=(
                "The user wants a NEW email sent to someone. Always supply recipient_hint; "
                "never invent an address."
            ),
            examples=("Send an email to Rahul about the invoice.",),
        ),
        Capability(
            name="gmail_reply",
            version="1.0",
            description="Draft a reply to an existing thread for explicit approval. Never sends.",
            request_model=GmailReplyRequest,
            risk_level=RiskLevel.L3_EXTERNAL_WRITE,
            approval_policy=ApprovalPolicy.ALWAYS,
            executor=_late("execute_gmail_reply"),
            selection_guidance="The user wants to reply to a message they already received.",
            examples=("Reply to Rahul saying I'll review it tonight.",),
        ),
        Capability(
            name="calendar_read",
            version="1.0",
            description=(
                "Read the user's real Google Calendar: agenda, free/busy, or open slots. "
                "Read-only; cannot create an event."
            ),
            request_model=CalendarReadRequest,
            risk_level=RiskLevel.L0_OBSERVE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=_late("execute_calendar_read"),
            selection_guidance=(
                "Any question about what is scheduled or whether the user is free. Use "
                "mode='free_busy' for availability questions even when they contain words "
                "like 'book' or 'schedule'; use mode='open_slots' to find a gap."
            ),
            examples=(
                "What's on my calendar tomorrow?",
                "Am I free to book the gym tomorrow at 5 pm?",
            ),
        ),
        Capability(
            name="calendar_create",
            version="1.0",
            description="Prepare a new calendar event for explicit approval. Never creates directly.",
            request_model=CalendarCreateRequest,
            risk_level=RiskLevel.L3_EXTERNAL_WRITE,
            approval_policy=ApprovalPolicy.ALWAYS,
            executor=_late("execute_calendar_create"),
            selection_guidance=(
                "The user wants an event actually added. Always supply title; supply start "
                "only when an exact clock time was stated."
            ),
            examples=("Schedule a project review tomorrow at 3 pm.",),
        ),
        Capability(
            name="cross_tool_read",
            version="1.0",
            description="Read Gmail AND Calendar together in one request. Read-only.",
            request_model=CrossToolReadRequest,
            risk_level=RiskLevel.L0_OBSERVE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=_late("execute_cross_tool_read"),
            selection_guidance=(
                "Only when the user unambiguously wants BOTH their real inbox and their "
                "real calendar checked in the same turn."
            ),
            examples=("Check my latest emails and what's on my calendar tomorrow.",),
        ),
    )


def _register_capabilities() -> None:
    active = registry()
    for capability in build_capabilities():
        if not active.has(capability.name):
            active.register(capability)


_register_capabilities()


_UNREAD_RE = re.compile(r"\bunread\b", re.IGNORECASE)


def _enrich_arguments(
    tool: str, user_message: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    """Fill class-internal fields the Brain omitted, from the wording.

    This is the only sanctioned use of raw text at the boundary, and it is
    strictly scoped: it may set FIELDS inside the capability the Brain already
    chose. It has no branch that can select a different capability.
    """
    enriched = dict(arguments or {})

    if tool == "gmail_read" and "read_kind" not in enriched:
        # "check my unread emails" -> read_kind='unread'. Still a gmail_read.
        if _UNREAD_RE.search(user_message):
            enriched["read_kind"] = "unread"

    return enriched


def _clarification(message: str, reason: str) -> OrchestratorResult:
    return OrchestratorResult(
        reply=message,
        action_type="clarification_required",
        memory_content=f"{message}\nRoute: brain\nWhy: {reason}",
        spoken_reply=message,
    )


def execute_request(request: ToolRequest) -> OrchestratorResult:
    """Run one already-validated request through its own capability.

    The capability is resolved from the request TYPE, so a request object can
    only ever reach the executor matching its class.
    """
    # The safety-relevant guard lives in CapabilityRegistry.execute: it refuses a
    # request whose type does not match the capability's declared request_model.
    # Return values are not type-checked here -- doing so bought nothing for the
    # invariant and made every executor unpatchable in tests.
    return registry().execute(request)


def execute_answer(
    request: ToolRequest,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> OrchestratorResult:
    """Run a conversational answer through the registry, and audit it.

    general_answer is a registered capability, so it earns a tool_runs row like
    any other. Without this, answer turns -- the majority of turns -- would have
    no durable execution record, which is the gap task_log used to leave.
    """
    started_at = datetime.now(timezone.utc)
    capability = registry().get(request.tool_name)
    result = execute_request(request)
    audit_service.record_tool_run(
        tool_name=request.tool_name,
        tool_version=capability.version,
        request_arguments=request.audit_arguments(),
        request_hash=request.request_hash(),
        risk_level=capability.risk_level.value,
        requires_approval=capability.requires_approval,
        status=audit_service.status_for_result(result.action_type),
        session_id=session_id,
        turn_id=turn_id,
        user_visible_summary=result.reply,
        started_at=started_at,
    )
    return result


def execute(
    decision: BrainDecision,
    user_message: str,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> OrchestratorResult:
    """Deterministic execution step, run only after the Brain chose mode=='tool'.

    Validates the Brain's arguments into the typed request for the selected
    capability, then dispatches on that request's type. A validation failure
    fails closed to clarification rather than guessing.
    """
    started_at = datetime.now(timezone.utc)
    tool = decision.tool
    if not tool or not registry().has(tool):
        logger.warning("tool_executor received unsupported tool=%r; clarifying", tool)
        audit_service.record_tool_run(
            tool_name=str(tool or "unknown"),
            tool_version=None,
            request_arguments={"rejected": "unsupported_tool"},
            request_hash="",
            risk_level="unknown",
            requires_approval=False,
            status="blocked",
            session_id=session_id,
            turn_id=turn_id,
            error_code="unsupported_tool",
            started_at=started_at,
        )
        return _clarification(
            "I need a bit more detail before I can do that.",
            f"unsupported tool {tool!r}",
        )

    capability = registry().get(tool)

    try:
        request = build_request(
            tool, user_message, _enrich_arguments(tool, user_message, decision.arguments)
        )
    except ToolRequestValidationError as exc:
        logger.warning("Rejected %s arguments from the brain: %s", tool, exc)
        audit_service.record_tool_run(
            tool_name=tool,
            tool_version=capability.version,
            request_arguments={"rejected": "invalid_arguments"},
            request_hash="",
            risk_level=capability.risk_level.value,
            requires_approval=capability.requires_approval,
            status="blocked",
            session_id=session_id,
            turn_id=turn_id,
            error_code="invalid_arguments",
            started_at=started_at,
        )
        return _clarification(
            "I need a bit more detail before I can do that safely.",
            f"typed request validation failed for {tool}",
        )

    risk = capability.risk_decision(
        model_requested_approval=bool(decision.arguments.get("requires_approval"))
        if isinstance(decision.arguments, dict)
        else None
    )
    logger.info(
        "tool_executor typed dispatch tool=%s risk=%s requires_approval=%s hash=%s",
        tool,
        risk.risk_level.value,
        risk.requires_approval,
        request.request_hash(),
    )

    def _audit(result: OrchestratorResult, error_code: str | None = None) -> None:
        approval = result.approval if isinstance(result.approval, Mapping) else None
        approval_id = approval.get("id") if approval else None
        run_id = audit_service.record_tool_run(
            tool_name=tool,
            tool_version=capability.version,
            # Sanitized: raw text and bodies never reach the audit row.
            request_arguments=request.audit_arguments(),
            request_hash=request.request_hash(),
            risk_level=risk.risk_level.value,
            requires_approval=risk.requires_approval,
            status=audit_service.status_for_result(result.action_type),
            session_id=session_id,
            turn_id=turn_id,
            approval_id=int(approval_id) if isinstance(approval_id, int) else None,
            error_code=error_code,
            user_visible_summary=result.reply,
            started_at=started_at,
        )

        # Read capabilities are verified inline: the observation is available
        # now. Write capabilities are only PROPOSED here, so their external
        # verifier runs after approval execution instead.
        if error_code is None and not risk.requires_approval:
            verdict = verification_service.verify_read(request, result)
            if verdict is not None:
                audit_service.record_verification(
                    verifier_name=verdict.verifier_name,
                    verdict=verdict.verdict,
                    tool_run_id=run_id,
                    expected=verdict.expected,
                    observed=verdict.observed,
                    evidence_text=verdict.evidence_text,
                )
                if verdict.verdict != "verified":
                    logger.warning(
                        "%s verdict=%s: %s",
                        verdict.verifier_name,
                        verdict.verdict,
                        verdict.evidence_text,
                    )

    if risk.blocked:
        blocked = _clarification(
            "That action is not permitted.", f"{tool} is forbidden by risk policy"
        )
        _audit(blocked, error_code="forbidden_by_policy")
        return blocked

    try:
        result = execute_request(request)
    except Exception as exc:
        logger.exception("Capability %s raised during execution: %s", tool, exc)
        failure = OrchestratorResult(
            reply="I couldn't complete that action. Nothing was changed.",
            action_type="error",
            memory_content=f"Route: {tool}\nWhy: executor raised {type(exc).__name__}",
            spoken_reply="I couldn't complete that action. Nothing was changed.",
        )
        _audit(failure, error_code=type(exc).__name__)
        return failure

    result = _enforce_approval_policy(tool, risk, result)
    _audit(result)
    return result


# An approval-gated capability may only ever report a proposal, a clarification
# or a failure. These are the action types that would mean the external write
# already happened.
_COMPLETED_ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"task_complete", "gmail_sent", "calendar_created"}
)


def _enforce_approval_policy(
    tool: str, risk: RiskDecision, result: OrchestratorResult
) -> OrchestratorResult:
    """Refuse to publish a completed external write that skipped approval.

    Policy is authoritative over both the model and the executor. The write
    executors structurally only build approval proposals, so this is a
    belt-and-braces check -- but it is the difference between "no code path does
    that today" and "no code path can do that".
    """
    if not risk.requires_approval:
        return result

    if result.action_type in _COMPLETED_ACTION_TYPES:
        logger.error(
            "POLICY VIOLATION: %s is %s and requires approval, but the executor "
            "reported %s. Refusing to publish the result.",
            tool,
            risk.risk_level.value,
            result.action_type,
        )
        message = (
            "That action requires your explicit approval, so I stopped before "
            "anything was changed."
        )
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=f"{message}\nRoute: policy\nWhy: approval bypass refused for {tool}",
            spoken_reply=message,
        )

    if result.action_type == "approval_required" and not result.approval:
        logger.error(
            "POLICY VIOLATION: %s reported approval_required without an approval "
            "payload; refusing to publish an unreviewable proposal.",
            tool,
        )
        message = "I couldn't prepare a reviewable preview for that, so nothing was changed."
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=f"{message}\nRoute: policy\nWhy: missing approval payload for {tool}",
            spoken_reply=message,
        )

    return result
