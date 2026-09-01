from __future__ import annotations

import logging

from .cross_tool_fastpath import handle_cross_tool_fast_request
from .cross_tool_reasoning import (
    CrossToolWriteNotSupportedError,
    is_cross_tool_request,
)
from .message_dispatch import handle_message_result as _legacy_dispatch
from .orchestrator import OrchestratorResult

logger = logging.getLogger(__name__)

# Keep the established facade symbol so existing tests/extensions can patch the combined
# handler without depending on which internal implementation serves it.
handle_cross_tool_request = handle_cross_tool_fast_request


def handle_message_result(user_message: str) -> OrchestratorResult:
    """Top-level intelligence facade introduced in Part 9.

    Explicit Gmail+Calendar reads use the typed local fast path: deterministic routing,
    independent R1 reads in parallel, then one quality-preserving synthesis stage. Existing
    single-tool Gmail/Calendar behavior remains delegated to the proven dispatcher.
    """
    if not is_cross_tool_request(user_message):
        return _legacy_dispatch(user_message)

    try:
        result = handle_cross_tool_request(user_message)
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
        logger.exception("Part 9 cross-tool request failed: %s", exc)
        return OrchestratorResult(
            reply=(
                "I couldn't complete the combined Gmail and Calendar request right now. "
                "Neither service was changed."
            ),
            action_type="error",
            memory_content=(
                "I couldn't complete the combined Gmail and Calendar request right now. Neither service was changed."
            ),
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
