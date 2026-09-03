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

# Kept for backward compatibility: some callers/tests still reference these symbols on
# this module. They are no longer consulted by handle_message_result below -- the single
# semantic authority for every turn, including explicit combined Gmail+Calendar reads, is
# brain_agent.decide() (invoked from message_dispatch.handle_message_result), which routes
# to tool_executor.execute(). tool_executor's cross_tool_read branch is what now calls
# handle_cross_tool_fast_request.
handle_cross_tool_request = handle_cross_tool_fast_request

__all__ = [
    "handle_message_result",
    "handle_cross_tool_request",
    "is_cross_tool_request",
    "CrossToolWriteNotSupportedError",
]


def handle_message_result(user_message: str) -> OrchestratorResult:
    """Top-level intelligence facade.

    This used to run its own pre-brain keyword gate here (`is_cross_tool_request(...)`)
    and, on a match, execute a combined Gmail+Calendar read *before* the semantic brain
    ever saw the message -- which caused real Gmail/Calendar API calls on purely
    conceptual questions like "Explain the difference between Gmail and Google Calendar"
    whenever both sets of keywords appeared together.

    There is now exactly one semantic authority: brain_agent.decide(), reached via
    message_dispatch.handle_message_result. This facade always delegates to it. Explicit
    combined-read requests are still supported, but only after the brain has classified
    the turn as mode="tool", tool="cross_tool_read" -- see tool_executor.execute().
    """
    return _legacy_dispatch(user_message)
