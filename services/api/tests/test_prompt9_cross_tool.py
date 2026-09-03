from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from services.api.app import cross_tool_reasoning, intelligence_dispatch
from services.api.app.cross_tool_reasoning import (
    CrossToolPlan,
    CrossToolResult,
    PlannedStep,
    StepResult,
)
from services.api.app.llm_service import LLMResult


EXAMPLE = (
    "Check today's emails, tell me if anything's urgent, "
    "and see if I'm free tomorrow afternoon."
)


class Prompt9CrossToolTests(unittest.TestCase):
    def test_input_builds_ordered_gmail_calendar_task_list_without_cloud(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            plan = cross_tool_reasoning.plan_cross_tool_request(EXAMPLE)

        self.assertEqual(plan.source, "local_fallback")
        self.assertEqual(
            [step.tool for step in plan.steps],
            ["gmail.read", "calendar.read"],
        )
        self.assertTrue(all(step.instruction == EXAMPLE for step in plan.steps))

    def test_combined_output_is_one_grounded_response_and_one_spoken_answer(self) -> None:
        plan = CrossToolPlan(
            steps=(
                PlannedStep("gmail.read", "Check today's emails and identify anything needing attention."),
                PlannedStep("calendar.read", "See if I'm free tomorrow afternoon."),
            ),
            reason="Need Gmail and Calendar reads.",
            source="test",
        )
        steps = (
            StepResult(
                index=1,
                tool="gmail.read",
                instruction=plan.steps[0].instruction,
                status="success",
                data={
                    "count": 1,
                    "emails": [
                        {
                            "sender": "Rahul <rahul@example.com>",
                            "subject": "Need review by noon",
                            "snippet": "Please review this before 12 PM today.",
                            "timestamp": "2026-08-30T08:00:00+05:30",
                        }
                    ],
                },
            ),
            StepResult(
                index=2,
                tool="calendar.read",
                instruction=plan.steps[1].instruction,
                status="success",
                data={
                    "formatted": "You're free on Monday, August 31 from 12:00 PM to 5:00 PM.",
                    "busy": [],
                },
            ),
        )
        model_payload = (
            '{"reply":"Rahul needs your review by noon. You are free tomorrow afternoon '
            'from 12:00 PM to 5:00 PM.",'
            '"spoken_reply":"Rahul needs your review by noon, sir. You are free tomorrow '
            'afternoon from noon to five."}'
        )

        with patch.object(
            cross_tool_reasoning,
            "generate_text",
            return_value=LLMResult(text=model_payload, provider="groq", model="test"),
        ) as generate:
            reply, spoken = cross_tool_reasoning.synthesize_results(EXAMPLE, plan, steps)

        self.assertEqual(generate.call_count, 1)
        self.assertIn("Rahul", reply)
        self.assertIn("12:00 PM", reply)
        self.assertIn("Rahul", spoken)
        self.assertIn("noon to five", spoken)
        self.assertNotIn("details are on screen", spoken.lower())

    def test_step_failure_does_not_block_next_tool(self) -> None:
        plan = CrossToolPlan(
            steps=(
                PlannedStep("gmail.read", "Check my emails."),
                PlannedStep("calendar.read", "Check tomorrow afternoon availability."),
            ),
            reason="Need both reads.",
            source="test",
        )
        calls: list[str] = []

        def gmail_executor(_instruction, _context):
            calls.append("gmail")
            raise cross_tool_reasoning.GmailServiceError("forced Gmail failure")

        def calendar_executor(_instruction, _context):
            calls.append("calendar")
            return {
                "formatted": "You're free tomorrow afternoon.",
                "busy": [],
            }

        registry = cross_tool_reasoning.ToolRegistry()
        registry.register(
            cross_tool_reasoning.ToolSpec(
                name="gmail.read",
                description="test",
                risk_level="R1",
                requires_approval=False,
                executor=gmail_executor,
            )
        )
        registry.register(
            cross_tool_reasoning.ToolSpec(
                name="calendar.read",
                description="test",
                risk_level="R1",
                requires_approval=False,
                executor=calendar_executor,
            )
        )

        results = cross_tool_reasoning.execute_plan(plan, registry)

        self.assertEqual(calls, ["gmail", "calendar"])
        self.assertEqual(results[0].status, "failed")
        self.assertEqual(results[1].status, "success")
        self.assertIn("forced Gmail failure", results[0].error or "")

    def test_combined_write_request_is_refused_before_any_tool_runs(self) -> None:
        request = "Reply to my latest email and schedule a meeting tomorrow at 3 PM."
        self.assertTrue(cross_tool_reasoning.is_cross_tool_request(request))
        with self.assertRaises(cross_tool_reasoning.CrossToolWriteNotSupportedError):
            cross_tool_reasoning.plan_cross_tool_request(request)

    def test_intelligence_facade_always_delegates_to_brain_first_dispatch(self) -> None:
        """intelligence_dispatch is a thin facade: no pre-brain keyword gate runs here.

        Combined Gmail+Calendar requests are routed by brain_agent.decide() choosing
        tool="cross_tool_read", not by a regex check in this module (see
        test_brain_v2_policy.py for that routing behavior end to end).
        """
        sentinel = object()
        with (
            patch.object(intelligence_dispatch, "handle_cross_tool_request") as cross_tool,
            patch.object(intelligence_dispatch, "_legacy_dispatch", return_value=sentinel) as legacy,
        ):
            result = intelligence_dispatch.handle_message_result(EXAMPLE)

        # Part 10.2 Phase D: the facade now forwards the active session id too.
        legacy.assert_called_once_with(EXAMPLE, session_id=None, turn_id=None)
        cross_tool.assert_not_called()
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
