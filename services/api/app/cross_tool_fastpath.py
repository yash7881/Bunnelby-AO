from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

from .cross_tool_reasoning import (
    CrossToolPlan,
    CrossToolWriteNotSupportedError,
    StepResult,
    _deterministic_plan,
    build_cross_tool_registry,
    contains_cross_tool_write,
    execute_plan,
    synthesize_results,
)
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

_PARALLEL_READ_TOOLS = frozenset({"gmail.read", "calendar.read"})


@dataclass(frozen=True)
class FastCrossToolResult:
    reply: str
    spoken_reply: str
    plan: CrossToolPlan
    steps: tuple[StepResult, ...]
    timings_ms: Mapping[str, float]


def _build_fast_plan(user_message: str) -> CrossToolPlan:
    """Build the obvious Gmail+Calendar read plan locally with zero LLM latency.

    `intelligence_dispatch` only enters this path after both tool families were detected.
    The existing deterministic planner already preserves the user's full wording for both
    tools and preserves cue order, so an extra cloud planner call cannot add permissions or
    facts and is unnecessary on this narrow, typed two-tool path.
    """
    base = _deterministic_plan(user_message)
    return CrossToolPlan(
        steps=base.steps,
        reason="Typed local fast path for explicit Gmail + Calendar read request.",
        source="local_fastpath",
    )


def _validate_parallel_read_plan(plan: CrossToolPlan, registry: ToolRegistry) -> None:
    names = [step.tool for step in plan.steps]
    if not names or len(names) != len(set(names)):
        raise CrossToolWriteNotSupportedError(
            "Fast parallel execution requires unique independent read tools."
        )
    if not set(names).issubset(_PARALLEL_READ_TOOLS):
        raise CrossToolWriteNotSupportedError(
            "Only Gmail and Calendar read tools are allowed in the cross-tool fast path."
        )

    for name in names:
        spec = registry.get(name)
        if spec.requires_approval or spec.risk_level != "R1":
            raise CrossToolWriteNotSupportedError(
                f"Tool {name} is not eligible for approval-free parallel read execution."
            )


def _execute_one_step(
    step_index: int,
    plan: CrossToolPlan,
    registry: ToolRegistry,
) -> tuple[StepResult, float]:
    step = plan.steps[step_index]
    single_step_plan = CrossToolPlan(
        steps=(step,),
        reason=plan.reason,
        source=plan.source,
    )
    started = perf_counter()
    result = execute_plan(single_step_plan, registry)[0]
    elapsed_ms = (perf_counter() - started) * 1000.0

    # execute_plan numbers a single-step plan from one. Restore the original order so all
    # downstream synthesis/audit behavior remains deterministic regardless of completion order.
    result = StepResult(
        index=step_index + 1,
        tool=result.tool,
        instruction=result.instruction,
        status=result.status,
        data=result.data,
        error=result.error,
    )
    return result, elapsed_ms


def execute_read_plan_parallel(
    plan: CrossToolPlan,
    registry: ToolRegistry | None = None,
) -> tuple[tuple[StepResult, ...], dict[str, float]]:
    """Execute independent R1 Gmail/Calendar reads concurrently and fail closed otherwise."""
    active_registry = registry or build_cross_tool_registry()
    _validate_parallel_read_plan(plan, active_registry)

    results: list[StepResult | None] = [None] * len(plan.steps)
    tool_timings: dict[str, float] = {}

    with ThreadPoolExecutor(
        max_workers=len(plan.steps),
        thread_name_prefix="bunnelby-read",
    ) as executor:
        futures = {
            executor.submit(_execute_one_step, index, plan, active_registry): index
            for index in range(len(plan.steps))
        }
        for future, index in [(future, index) for future, index in futures.items()]:
            result, elapsed_ms = future.result()
            results[index] = result
            tool_timings[f"tool_{result.tool}_ms"] = round(elapsed_ms, 2)

    completed = tuple(result for result in results if result is not None)
    if len(completed) != len(plan.steps):
        raise RuntimeError("Cross-tool fast path lost a tool result.")
    return completed, tool_timings


def handle_cross_tool_fast_request(user_message: str) -> FastCrossToolResult:
    """Low-latency, quality-preserving path for explicit Gmail + Calendar read requests."""
    if contains_cross_tool_write(user_message):
        raise CrossToolWriteNotSupportedError(
            "Combined Gmail + Calendar mode is read-only in this phase. Request external writes separately so Bunnelby can preserve the existing approval gate."
        )

    total_started = perf_counter()

    plan_started = perf_counter()
    plan = _build_fast_plan(user_message)
    planner_ms = (perf_counter() - plan_started) * 1000.0

    tools_started = perf_counter()
    steps, tool_timings = execute_read_plan_parallel(plan)
    tools_wall_ms = (perf_counter() - tools_started) * 1000.0

    synthesis_started = perf_counter()
    reply, spoken_reply = synthesize_results(user_message, plan, steps)
    synthesis_ms = (perf_counter() - synthesis_started) * 1000.0

    total_ms = (perf_counter() - total_started) * 1000.0
    timings: dict[str, float] = {
        "planner_ms": round(planner_ms, 2),
        "tools_wall_ms": round(tools_wall_ms, 2),
        **tool_timings,
        "synthesis_ms": round(synthesis_ms, 2),
        "cross_tool_total_ms": round(total_ms, 2),
    }

    logger.info(
        "Bunnelby cross-tool fast-path latency_ms=%s",
        timings,
    )
    return FastCrossToolResult(
        reply=reply,
        spoken_reply=spoken_reply,
        plan=plan,
        steps=steps,
        timings_ms=timings,
    )
