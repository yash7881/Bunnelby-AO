from __future__ import annotations

import logging

from . import message_dispatch
from .brain_agent import BrainDecision
from .calendar_agenda_service import is_agenda_request
from .cross_tool_fastpath import handle_cross_tool_fast_request
from .cross_tool_reasoning import CrossToolWriteNotSupportedError
from .gmail_service import normalize_spoken_email
from .orchestrator import OrchestratorResult, gmail_handler

logger = logging.getLogger(__name__)


def _execute_cross_tool_read(user_message: str) -> OrchestratorResult:
    """Explicit combined Gmail + Calendar read, run only after brain_agent selects it.

    Reuses the existing low-latency cross-tool fast-path implementation (parallel R1 reads
    plus quality-preserving synthesis); only the invocation point moved here, from behind
    intelligence_dispatch.py's old pre-brain keyword gate.
    """
    try:
        result = handle_cross_tool_fast_request(user_message)
    except CrossToolWriteNotSupportedError as exc:
        message = str(exc)
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=message,
            spoken_reply=(
                "Combined tool actions are read-only right now, sir. Request the write separately so I can preserve the approval check."
            ),
        )
    except Exception as exc:
        logger.exception("cross_tool_read execution failed: %s", exc)
        message = (
            "I couldn't complete the combined Gmail and Calendar request right now. "
            "Neither service was changed."
        )
        return OrchestratorResult(
            reply=message,
            action_type="error",
            memory_content=message,
            spoken_reply="I couldn't complete the combined check right now, sir. Nothing was changed.",
        )

    succeeded = sum(step.status == "success" for step in result.steps)
    failed = len(result.steps) - succeeded
    status_note = f"Cross-tool plan: {succeeded} succeeded, {failed} failed."
    spoken_metadata: dict[str, object] = {
        "cross_tool": True,
        "steps_total": len(result.steps),
        "steps_succeeded": succeeded,
        "steps_failed": failed,
        "plan_source": result.plan.source,
    }
    timings = getattr(result, "timings_ms", None)
    if timings:
        spoken_metadata["latency_ms"] = dict(timings)

    return OrchestratorResult(
        reply=result.reply,
        action_type="task_complete" if succeeded else "error",
        memory_content=f"{result.reply}\n{status_note}",
        spoken_reply=result.spoken_reply,
        spoken_metadata=spoken_metadata,
    )


def execute(decision: BrainDecision, user_message: str) -> OrchestratorResult:
    """Deterministic execution step, run only after brain_agent has decided mode=='tool'.

    Reuses the existing proven Gmail/Calendar builder functions in message_dispatch.py and
    orchestrator.py; never calls a send/create function directly. Every write still goes
    through approval_service via those existing builders.
    """
    tool = decision.tool

    if tool == "gmail_read":
        handler_result = gmail_handler(user_message)
        return OrchestratorResult(
            reply=handler_result.reply,
            action_type=handler_result.action_type_override or "gmail_summary",
            memory_content=f"{handler_result.reply}\nRoute: gmail\nWhy: brain_agent selected gmail_read",
            spoken_reply=handler_result.spoken_reply,
            spoken_metadata=handler_result.spoken_metadata,
            approval=handler_result.approval,
        )

    if tool == "gmail_reply":
        handler_result = gmail_handler(user_message)
        return OrchestratorResult(
            reply=handler_result.reply,
            action_type=handler_result.action_type_override or "gmail_summary",
            memory_content=f"{handler_result.reply}\nRoute: gmail\nWhy: brain_agent selected gmail_reply",
            spoken_reply=handler_result.spoken_reply,
            spoken_metadata=handler_result.spoken_metadata,
            approval=handler_result.approval,
        )

    if tool == "gmail_compose":
        normalized_message = normalize_spoken_email(user_message)
        return message_dispatch._gmail_compose_result(normalized_message)

    if tool == "calendar_read":
        if is_agenda_request(user_message):
            return message_dispatch._calendar_agenda_result(user_message)
        return message_dispatch._calendar_result(user_message)

    if tool == "calendar_create":
        return message_dispatch._calendar_result(user_message)

    if tool == "cross_tool_read":
        return _execute_cross_tool_read(user_message)

    logger.warning("tool_executor received unsupported tool=%r; falling back to clarify", tool)
    message = "I need a bit more detail before I can do that."
    return OrchestratorResult(
        reply=message,
        action_type="clarification_required",
        memory_content=message,
        spoken_reply=message,
    )
