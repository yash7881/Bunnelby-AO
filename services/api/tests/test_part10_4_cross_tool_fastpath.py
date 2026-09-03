from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from services.api.app.cross_tool_fastpath import (
    _build_fast_plan,
    _synthesize_low_latency,
    execute_read_plan_parallel,
    handle_cross_tool_fast_request,
)
from services.api.app.cross_tool_reasoning import (
    CrossToolPlan,
    CrossToolWriteNotSupportedError,
    PlannedStep,
    StepResult,
)
from services.api.app.llm_service import LLMResult, LLMUnavailableError
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

    def _fake_plan_and_steps(self):
        plan = CrossToolPlan(
            steps=(
                PlannedStep(tool="gmail.read", instruction="gmail"),
                PlannedStep(tool="calendar.read", instruction="calendar"),
            ),
            reason="test",
            source="local_fastpath",
        )
        steps = (
            StepResult(
                index=1,
                tool="gmail.read",
                instruction="gmail",
                status="success",
                data={
                    "count": 1,
                    "emails": [
                        {
                            "sender": "GitHub <noreply@github.com>",
                            "subject": "Security Gates workflow failed",
                            "timestamp": "2026-09-01T18:00:00+00:00",
                            "snippet": "Backend tests and security audit failed.",
                        }
                    ],
                },
            ),
            StepResult(
                index=2,
                tool="calendar.read",
                instruction="calendar",
                status="success",
                data={"formatted": "Tomorrow is free from 9:00 AM to 6:00 PM."},
            ),
        )
        return plan, steps

    def test_low_latency_synthesis_prefers_configured_groq(self) -> None:
        plan, steps = self._fake_plan_and_steps()
        model_result = LLMResult(
            text=(
                '{"reply":"Tomorrow is clear. Focus on the failed Security Gates workflow first.",'
                '"spoken_reply":"Tomorrow is clear. Focus on the failed Security Gates workflow first."}'
            ),
            provider="groq",
            model="llama-3.3-70b-versatile",
        )
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False),
            # Part 10.2 Phase E: provider order is now gateway policy, not a
            # call-site decision, so assert the profile the module asks for.
            patch(
                "services.api.app.model_gateway.generate",
                return_value=model_result,
            ) as groq,
            patch("services.api.app.cross_tool_fastpath.synthesize_results") as standard,
        ):
            reply, spoken, provider = _synthesize_low_latency(
                "check my calendar and email, then tell me what to focus on first",
                plan,
                steps,
            )

        self.assertEqual(provider, "groq")
        self.assertIn("Security Gates", reply)
        self.assertIn("Security Gates", spoken)
        groq.assert_called_once()
        standard.assert_not_called()

    def test_low_latency_synthesis_falls_back_to_standard_on_groq_failure(self) -> None:
        plan, steps = self._fake_plan_and_steps()
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False),
            patch(
                "services.api.app.model_gateway.generate",
                side_effect=LLMUnavailableError("forced failure"),
            ),
            patch(
                "services.api.app.cross_tool_fastpath.synthesize_results",
                return_value=("quality fallback", "quality fallback spoken"),
            ) as standard,
        ):
            reply, spoken, provider = _synthesize_low_latency(
                "read Gmail and check calendar",
                plan,
                steps,
            )

        self.assertEqual((reply, spoken), ("quality fallback", "quality fallback spoken"))
        self.assertEqual(provider, "standard_fallback")
        standard.assert_called_once_with("read Gmail and check calendar", plan, steps)

    def test_low_latency_synthesis_uses_standard_when_groq_is_not_configured(self) -> None:
        plan, steps = self._fake_plan_and_steps()
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False),
            patch(
                "services.api.app.cross_tool_fastpath.synthesize_results",
                return_value=("standard", "standard spoken"),
            ) as standard,
            patch("services.api.app.model_gateway.generate") as groq,
        ):
            reply, spoken, provider = _synthesize_low_latency(
                "read Gmail and check calendar",
                plan,
                steps,
            )

        self.assertEqual((reply, spoken, provider), ("standard", "standard spoken", "standard"))
        standard.assert_called_once()
        groq.assert_not_called()

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
                "services.api.app.cross_tool_fastpath._synthesize_low_latency",
                return_value=("combined answer", "combined spoken answer", "groq"),
            ) as synthesis,
        ):
            result = handle_cross_tool_fast_request("read Gmail and check calendar")

        self.assertEqual(result.reply, "combined answer")
        self.assertEqual(result.spoken_reply, "combined spoken answer")
        self.assertEqual(result.plan.source, "local_fastpath")
        self.assertIn("cross_tool_total_ms", result.timings_ms)
        self.assertEqual(result.timings_ms["synthesis_groq_fastpath"], 1.0)
        synthesis.assert_called_once()


if __name__ == "__main__":
    unittest.main()
