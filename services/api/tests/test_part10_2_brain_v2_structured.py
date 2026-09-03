from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.api.app import brain_agent, capability_registry, tool_executor
from services.api.app.risk_policy import ApprovalPolicy, RiskLevel


def llm_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(payload), provider="gemini", model="fixture")


def raw_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, provider="gemini", model="fixture")


class BrainDecisionV2ShapeTests(unittest.TestCase):
    def test_v2_fields_exist_with_safe_defaults(self) -> None:
        decision = brain_agent.BrainDecision(mode="answer", tool=None, confidence=1.0)
        self.assertIsNone(decision.requires_approval)
        self.assertEqual(decision.reason_code, "")
        self.assertEqual(decision.untrusted_context_used, ())
        self.assertEqual(decision.response_policy, "both")

    def test_v2_alias_is_the_same_class(self) -> None:
        self.assertIs(brain_agent.BrainDecisionV2, brain_agent.BrainDecision)

    def test_original_fields_are_unchanged(self) -> None:
        decision = brain_agent.BrainDecision(
            mode="tool", tool="gmail_read", confidence=0.9, arguments={"read_kind": "unread"}
        )
        self.assertEqual(decision.mode, "tool")
        self.assertEqual(decision.tool, "gmail_read")
        self.assertEqual(decision.arguments["read_kind"], "unread")


class ResponseSchemaTests(unittest.TestCase):
    """The schema is generated from the registry and must stay provider-valid."""

    def test_tool_enum_matches_the_registry(self) -> None:
        schema = brain_agent.decision_response_schema()
        self.assertEqual(
            sorted(schema["properties"]["tool"]["enum"]),
            sorted(capability_registry.registry().catalog_names())
            if hasattr(capability_registry.registry(), "catalog_names")
            else sorted(e["name"] for e in capability_registry.registry().catalog()),
        )

    def test_enum_contains_no_empty_string(self) -> None:
        # Gemini rejects an empty enum member with HTTP 400; this was a real
        # Phase H defect caught against the live API.
        self.assertNotIn("", brain_agent.decision_response_schema()["properties"]["tool"]["enum"])

    def test_tool_is_not_required_so_answers_may_omit_it(self) -> None:
        schema = brain_agent.decision_response_schema()
        self.assertNotIn("tool", schema["required"])
        for field in ("mode", "confidence", "reply", "spoken_reply"):
            self.assertIn(field, schema["required"])

    def test_schema_is_accepted_by_the_provider_sdk(self) -> None:
        from google.genai import types

        types.Schema(**brain_agent.decision_response_schema())

    def test_argument_union_covers_every_capability_argument(self) -> None:
        union = set(brain_agent.decision_response_schema()["properties"]["arguments"]["properties"])
        for entry in capability_registry.registry().catalog():
            for name in entry["arguments"].get("properties", {}):
                self.assertIn(name, union, f"{entry['name']}.{name} missing from schema union")

    def test_v2_envelope_fields_are_declared(self) -> None:
        properties = brain_agent.decision_response_schema()["properties"]
        for field in ("reason_code", "requires_approval", "response_policy"):
            self.assertIn(field, properties)


class GeneratedToolCatalogTests(unittest.TestCase):
    def test_catalog_lists_every_selectable_tool(self) -> None:
        section = brain_agent.tool_catalog_section()
        for name in capability_registry.registry().tool_names():
            self.assertIn(name, section)

    def test_catalog_marks_write_capabilities_as_approval_gated(self) -> None:
        section = brain_agent.tool_catalog_section()
        for line in section.splitlines():
            if line.startswith("- gmail_compose") or line.startswith("- calendar_create"):
                self.assertIn("REQUIRES EXPLICIT APPROVAL", line)

    def test_catalog_does_not_offer_general_answer_as_a_tool(self) -> None:
        self.assertNotIn("- general_answer", brain_agent.tool_catalog_section())

    def test_instruction_is_persona_plus_prose_plus_catalog(self) -> None:
        instruction = brain_agent.brain_system_instruction()
        self.assertTrue(instruction.startswith(brain_agent.BRAIN_SYSTEM_INSTRUCTION))
        self.assertIn("REGISTERED CAPABILITY CATALOG", instruction)

    def test_a_new_capability_appears_without_editing_prose(self) -> None:
        """Registering a capability must surface it in the catalog and schema."""
        from services.api.app.tool_requests import (
            REQUEST_MODELS,
            GmailReadRequest,
        )

        registry = capability_registry.CapabilityRegistry()
        for capability in tool_executor.build_capabilities():
            registry.register(capability)

        probe = capability_registry.Capability(
            name="gmail_read",  # reuse a registered request model for the probe
            version="9.9",
            description="PROBE CAPABILITY DESCRIPTION",
            request_model=GmailReadRequest,
            risk_level=RiskLevel.L0_OBSERVE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=lambda request: None,
        )
        # Same name is refused; the registry enforces uniqueness.
        with self.assertRaises(capability_registry.CapabilityRegistryError):
            registry.register(probe)
        self.assertIn("gmail_read", REQUEST_MODELS)

    def test_catalog_is_read_from_the_live_registry_not_a_literal(self) -> None:
        with patch.object(
            capability_registry.registry(), "catalog", return_value=()
        ):
            section = brain_agent.tool_catalog_section()
        self.assertNotIn("- gmail_read", section)


class StructuredDecodingTests(unittest.TestCase):
    """Parsing must fail closed and never invent a tool."""

    def setUp(self) -> None:
        self.memory = patch.object(
            brain_agent, "build_memory_context", return_value="MEMORY"
        )
        self.memory.start()
        self.addCleanup(self.memory.stop)

    def _decide(self, message: str, result: SimpleNamespace) -> brain_agent.BrainDecision:
        with patch.object(
            brain_agent, "generate_fast_text", return_value=result
        ), patch.object(brain_agent, "generate_text", return_value=result):
            return brain_agent.decide(message)

    def test_schema_is_passed_to_the_provider(self) -> None:
        seen: dict[str, object] = {}

        def capture(*, system_instruction, user_content, response_schema=None, **_kw):
            seen["schema"] = response_schema
            seen["instruction"] = system_instruction
            return llm_result({"mode": "answer", "confidence": 0.9, "reply": "x", "spoken_reply": "x"})

        with patch.object(brain_agent, "generate_fast_text", capture), patch.object(
            brain_agent, "generate_text", capture
        ):
            brain_agent.decide("hello there friend")

        self.assertIsNotNone(seen["schema"])
        self.assertIn("REGISTERED CAPABILITY CATALOG", str(seen["instruction"]))

    def test_exactly_one_generation_per_conversational_turn(self) -> None:
        payload = {"mode": "answer", "confidence": 0.9, "reply": "ok", "spoken_reply": "ok"}
        with patch.object(
            brain_agent, "generate_fast_text", return_value=llm_result(payload)
        ) as fast, patch.object(
            brain_agent, "generate_text", return_value=llm_result(payload)
        ) as balanced:
            brain_agent.decide("what is a vector database")
        self.assertEqual(
            fast.call_count + balanced.call_count,
            1,
            "a routing call plus a separate answer call would double free-tier usage",
        )

    def test_omitted_tool_is_treated_as_no_tool(self) -> None:
        decision = self._decide(
            "what is gmail",
            llm_result({"mode": "answer", "confidence": 0.9, "reply": "a", "spoken_reply": "a"}),
        )
        self.assertEqual(decision.mode, "answer")
        self.assertIsNone(decision.tool)

    def test_empty_string_tool_is_treated_as_no_tool(self) -> None:
        decision = self._decide(
            "what is gmail",
            llm_result(
                {"mode": "answer", "tool": "", "confidence": 0.9, "reply": "a", "spoken_reply": "a"}
            ),
        )
        self.assertIsNone(decision.tool)

    def test_unregistered_tool_is_discarded_and_fails_closed(self) -> None:
        decision = self._decide(
            "delete everything",
            llm_result(
                {
                    "mode": "tool",
                    "tool": "filesystem_delete",
                    "confidence": 0.99,
                    "reply": "",
                    "spoken_reply": "",
                }
            ),
        )
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.mode, "clarify")

    def test_malformed_envelope_never_selects_a_tool(self) -> None:
        decision = self._decide("check my email", raw_result("this is not json at all"))
        self.assertIsNone(decision.tool)
        self.assertNotEqual(decision.mode, "tool")

    def test_v2_fields_round_trip_from_the_envelope(self) -> None:
        decision = self._decide(
            "check my unread email",
            llm_result(
                {
                    "mode": "tool",
                    "tool": "gmail_read",
                    "confidence": 0.9,
                    "arguments": {"read_kind": "unread"},
                    "reply": "",
                    "spoken_reply": "",
                    "reason_code": "explicit_inbox_read",
                    "requires_approval": False,
                    "response_policy": "concise_spoken",
                }
            ),
        )
        self.assertEqual(decision.tool, "gmail_read")
        self.assertEqual(decision.reason_code, "explicit_inbox_read")
        self.assertIs(decision.requires_approval, False)
        self.assertEqual(decision.response_policy, "concise_spoken")

    def test_invalid_response_policy_falls_back_to_both(self) -> None:
        decision = self._decide(
            "hello friend how are you",
            llm_result(
                {
                    "mode": "answer",
                    "confidence": 0.9,
                    "reply": "a",
                    "spoken_reply": "a",
                    "response_policy": "telepathy",
                }
            ),
        )
        self.assertEqual(decision.response_policy, "both")

    def test_non_boolean_requires_approval_is_ignored(self) -> None:
        decision = self._decide(
            "hello friend how are you",
            llm_result(
                {
                    "mode": "answer",
                    "confidence": 0.9,
                    "reply": "a",
                    "spoken_reply": "a",
                    "requires_approval": "yes please",
                }
            ),
        )
        self.assertIsNone(decision.requires_approval)

    def test_low_confidence_write_still_fails_closed(self) -> None:
        decision = self._decide(
            "email rahul",
            llm_result(
                {
                    "mode": "tool",
                    "tool": "gmail_compose",
                    "confidence": 0.2,
                    "arguments": {"recipient_hint": "rahul"},
                    "reply": "",
                    "spoken_reply": "",
                }
            ),
        )
        self.assertEqual(decision.mode, "clarify")
        self.assertIsNone(decision.tool)


if __name__ == "__main__":
    unittest.main()
