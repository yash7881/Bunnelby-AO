from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from services.api.app.cross_tool_fastpath import (
    _build_fast_plan,
    execute_read_plan_parallel,
    handle_cross_tool_fast_request,
)
from services.api.app.cross_tool_reasoning import (
    CrossToolPlan,
    CrossToolWriteNotSupportedError,
    PlannedStep,
    StepResult,
)
from services.api.app.tool_registry import ToolRegistry, ToolSpec


class CrossToolFastPathTests(unittest.TestCase):
    def test_fast_plan_is_local_and_preserves_explicit_tool_order(self) -> None:
        plan = _build_fast_plan("check tomorrow's calendar and then unread Gmail")
        self.assertEqual(plan.source, "local_fastpath")
        self.assertEqual([step.tool for step in plan.steps], ["calendar.read", "gmail.read"])
        self.assertTrue(all("tomorrow" in step.instruction for step in plan.steps))

    def test_parallel_executor_really_overlaps_independent_reads(self) -> None:
        barrier = threading.Barrier(2)

        def gmail_executor(_instruction, _context):
            barrier.wait(timeout=1.0)
            return {"count": 1, "emails": []}

        def calendar_executor(_instruction, _context):
            barrier.wait(timeout=1.0)
            return {"formatted": "No events scheduled."}

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="gmail.read",
                description="test",
                risk_level="R1",
                requires_approval=False,
                executor=gmail_executor,
            )
        )
        registry.register(
            ToolSpec(
                name="calendar.read",
                description="test",
                risk_level="R1",
                requires_approval=False,
                executor=calendar_executor,
            )
        )
        plan = CrossToolPlan(
            steps=(
                PlannedStep(tool="calendar.read", instruction="calendar"),
                PlannedStep(tool="gmail.read", instruction="gmail"),
            ),
            reason="test",
            source="local_fastpath",
        )

        results, timings = execute_read_plan_parallel(plan, registry)

        self.assertEqual([result.tool for result in results], ["calendar.read", "gmail.read"])
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertIn("tool_calendar.read_ms", timings)
        self.assertIn("tool_gmail.read_ms", timings)

    def test_parallel_executor_fails_closed_for_approval_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="gmail.read",
                description="unsafe test",
                risk_level="R3",
                requires_approval=True,
                executor=lambda _instruction, _context: {},
            )
        )
        registry.register(
            ToolSpec(
                name="calendar.read",
                description="test",
                risk_level="R1",
                requires_approval=False,
                executor=lambda _instruction, _context: {},
            )
        )
        plan = CrossToolPlan(
            steps=(
                PlannedStep(tool="gmail.read", instruction="gmail"),
                PlannedStep(tool="calendar.read", instruction="calendar"),
            ),
            reason="test",
            source="local_fastpath",
        )

        with self.assertRaises(CrossToolWriteNotSupportedError):
            execute_read_plan_parallel(plan, registry)

    def test_handle_fast_request_keeps_synthesis_quality_stage(self) -> None:
        fake_steps = (
            StepResult(
                index=1,
                tool="gmail.read",
                instruction="gmail",
                status="success",
                data={"count": 1, "emails": []},
            ),
            StepResult(
                index=2,
                tool="calendar.read",
                instruction="calendar",
                status="success",
                data={"formatted": "No events."},
            ),
        )
        with (
            patch(
                "services.api.app.cross_tool_fastpath.execute_read_plan_parallel",
                return_value=(fake_steps, {"tool_gmail.read_ms": 10.0, "tool_calendar.read_ms": 12.0}),
            ),
            patch(
                "services.api.app.cross_tool_fastpath.synthesize_results",
                return_value=("combined answer", "combined spoken answer"),
            ) as synthesis,
        ):
            result = handle_cross_tool_fast_request("read Gmail and check calendar")

        self.assertEqual(result.reply, "combined answer")
        self.assertEqual(result.spoken_reply, "combined spoken answer")
        self.assertEqual(result.plan.source, "local_fastpath")
        self.assertIn("cross_tool_total_ms", result.timings_ms)
        synthesis.assert_called_once()


if __name__ == "__main__":
    unittest.main()
