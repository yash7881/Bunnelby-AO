from __future__ import annotations

import unittest
from unittest.mock import patch

from services.api.app import brain_agent, capability_registry, risk_policy, tool_executor
from services.api.app.orchestrator import OrchestratorResult
from services.api.app.risk_policy import (
    APPROVAL_REQUIRED_AT_OR_ABOVE,
    ApprovalPolicy,
    RiskLevel,
    RiskPolicyError,
    evaluate,
    requires_approval_for,
    validate_declaration,
)
from services.api.app.tool_requests import GmailReadRequest


class RiskLevelTests(unittest.TestCase):
    def test_canonical_tiers_exist_in_order(self) -> None:
        ordered = [
            RiskLevel.L0_OBSERVE,
            RiskLevel.L1_SAFE_CONTROL,
            RiskLevel.L2_MODIFY_LOCAL,
            RiskLevel.L3_EXTERNAL_WRITE,
            RiskLevel.L4_DESTRUCTIVE_SYSTEM,
            RiskLevel.FORBIDDEN,
        ]
        self.assertEqual([level.rank for level in ordered], sorted(l.rank for l in ordered))

    def test_at_least_compares_by_rank(self) -> None:
        self.assertTrue(RiskLevel.L3_EXTERNAL_WRITE.at_least(RiskLevel.L0_OBSERVE))
        self.assertFalse(RiskLevel.L0_OBSERVE.at_least(RiskLevel.L3_EXTERNAL_WRITE))

    def test_approval_threshold_is_external_write(self) -> None:
        self.assertIs(APPROVAL_REQUIRED_AT_OR_ABOVE, RiskLevel.L3_EXTERNAL_WRITE)


class ApprovalDerivationTests(unittest.TestCase):
    def test_reads_do_not_require_approval(self) -> None:
        self.assertFalse(requires_approval_for(RiskLevel.L0_OBSERVE, ApprovalPolicy.NEVER))

    def test_external_writes_always_require_approval(self) -> None:
        for policy in (ApprovalPolicy.NEVER, ApprovalPolicy.ALWAYS):
            self.assertTrue(
                requires_approval_for(RiskLevel.L3_EXTERNAL_WRITE, policy),
                f"L3 must require approval even when declared {policy}",
            )

    def test_destructive_system_requires_approval(self) -> None:
        self.assertTrue(
            requires_approval_for(RiskLevel.L4_DESTRUCTIVE_SYSTEM, ApprovalPolicy.NEVER)
        )

    def test_a_capability_may_tighten_below_the_threshold(self) -> None:
        self.assertTrue(requires_approval_for(RiskLevel.L2_MODIFY_LOCAL, ApprovalPolicy.ALWAYS))

    def test_blocked_policy_is_never_executed(self) -> None:
        decision = evaluate("bad", RiskLevel.FORBIDDEN, ApprovalPolicy.BLOCKED)
        self.assertTrue(decision.blocked)
        self.assertFalse(decision.executable)


class PolicyOverridesTheModelTests(unittest.TestCase):
    """The load-bearing rule: policy is authoritative over model preference."""

    def test_model_claiming_no_approval_cannot_unlock_a_write(self) -> None:
        decision = evaluate(
            "gmail_compose",
            RiskLevel.L3_EXTERNAL_WRITE,
            ApprovalPolicy.ALWAYS,
            model_requested_approval=False,
        )
        self.assertTrue(decision.requires_approval)

    def test_model_asking_for_approval_can_tighten_a_read(self) -> None:
        decision = evaluate(
            "gmail_read",
            RiskLevel.L0_OBSERVE,
            ApprovalPolicy.NEVER,
            model_requested_approval=True,
        )
        self.assertTrue(decision.requires_approval)

    def test_brain_requires_approval_field_is_advisory_only(self) -> None:
        capability = capability_registry.registry().get("gmail_compose")
        self.assertTrue(
            capability.risk_decision(model_requested_approval=False).requires_approval
        )


class DeclarationValidationTests(unittest.TestCase):
    def test_an_external_write_declared_without_always_is_rejected(self) -> None:
        with self.assertRaises(RiskPolicyError):
            validate_declaration(
                "sneaky_send", RiskLevel.L3_EXTERNAL_WRITE, ApprovalPolicy.NEVER
            )

    def test_forbidden_must_declare_blocked(self) -> None:
        with self.assertRaises(RiskPolicyError):
            validate_declaration("nope", RiskLevel.FORBIDDEN, ApprovalPolicy.ALWAYS)

    def test_valid_declarations_pass(self) -> None:
        validate_declaration("gmail_read", RiskLevel.L0_OBSERVE, ApprovalPolicy.NEVER)
        validate_declaration(
            "gmail_compose", RiskLevel.L3_EXTERNAL_WRITE, ApprovalPolicy.ALWAYS
        )

    def test_registry_refuses_an_unsafe_capability_at_registration(self) -> None:
        from services.api.app.tool_requests import GmailComposeRequest

        registry = capability_registry.CapabilityRegistry()
        unsafe = capability_registry.Capability(
            name="gmail_compose",
            version="1.0",
            description="unsafe declaration",
            request_model=GmailComposeRequest,
            risk_level=RiskLevel.L3_EXTERNAL_WRITE,
            approval_policy=ApprovalPolicy.NEVER,
            executor=lambda request: None,
        )
        with self.assertRaises(RiskPolicyError):
            registry.register(unsafe)


class RegisteredCapabilityPolicyTests(unittest.TestCase):
    """Current user-visible behaviour must be preserved exactly."""

    EXPECTED = {
        "general_answer": (RiskLevel.L0_OBSERVE, False),
        "gmail_read": (RiskLevel.L0_OBSERVE, False),
        "calendar_read": (RiskLevel.L0_OBSERVE, False),
        "cross_tool_read": (RiskLevel.L0_OBSERVE, False),
        "file_search": (RiskLevel.L0_OBSERVE, False),
        "gmail_compose": (RiskLevel.L3_EXTERNAL_WRITE, True),
        "gmail_reply": (RiskLevel.L3_EXTERNAL_WRITE, True),
        "calendar_create": (RiskLevel.L3_EXTERNAL_WRITE, True),
    }

    def test_every_capability_matches_the_expected_policy(self) -> None:
        registry = capability_registry.registry()
        self.assertEqual(set(registry.names()), set(self.EXPECTED))
        for name, (level, approval) in self.EXPECTED.items():
            capability = registry.get(name)
            self.assertIs(capability.risk_level, level, name)
            self.assertEqual(capability.requires_approval, approval, name)

    def test_no_capability_is_registered_as_forbidden_or_destructive(self) -> None:
        for name in capability_registry.registry().names():
            level = capability_registry.registry().get(name).risk_level
            self.assertNotIn(level, {RiskLevel.FORBIDDEN, RiskLevel.L4_DESTRUCTIVE_SYSTEM}, name)

    def test_risk_metadata_is_exposed_in_the_catalog(self) -> None:
        for entry in capability_registry.registry().catalog():
            self.assertIn("risk_level", entry)
            self.assertIn("requires_approval", entry)


class ApprovalBypassEnforcementTests(unittest.TestCase):
    """A write must never publish a completed action without approval."""

    def _decision(self, tool: str) -> brain_agent.BrainDecision:
        arguments = {"gmail_compose": {"recipient_hint": "rahul"}, "calendar_create": {"title": "T"}}
        return brain_agent.BrainDecision(
            mode="tool",
            tool=tool,
            confidence=0.95,
            arguments=arguments.get(tool, {}),
            reason="test",
        )

    def test_completed_write_without_approval_is_refused(self) -> None:
        rogue = OrchestratorResult(
            reply="Email sent!",
            action_type="task_complete",
            memory_content="Email sent!",
        )
        with patch.object(tool_executor, "execute_request", return_value=rogue):
            result = tool_executor.execute(self._decision("gmail_compose"), "email rahul")
        self.assertEqual(result.action_type, "error")
        self.assertIn("approval", result.reply.lower())

    def test_approval_required_without_a_payload_is_refused(self) -> None:
        rogue = OrchestratorResult(
            reply="Prepared.",
            action_type="approval_required",
            memory_content="Prepared.",
            approval=None,
        )
        with patch.object(tool_executor, "execute_request", return_value=rogue):
            result = tool_executor.execute(
                self._decision("calendar_create"), "schedule a review tomorrow at 3 pm"
            )
        self.assertEqual(result.action_type, "error")

    def test_a_proper_proposal_passes_through_untouched(self) -> None:
        good = OrchestratorResult(
            reply="Prepared.",
            action_type="approval_required",
            memory_content="Prepared.",
            approval={"id": 7, "task_type": "gmail_compose"},
        )
        with patch.object(tool_executor, "execute_request", return_value=good):
            result = tool_executor.execute(self._decision("gmail_compose"), "email rahul")
        self.assertEqual(result.action_type, "approval_required")
        self.assertEqual(result.approval, {"id": 7, "task_type": "gmail_compose"})

    def test_reads_are_not_subject_to_the_approval_check(self) -> None:
        read_result = OrchestratorResult(
            reply="Two emails.", action_type="task_complete", memory_content="Two emails."
        )
        with patch.object(tool_executor, "execute_request", return_value=read_result):
            result = tool_executor.execute(self._decision("gmail_read"), "check my email")
        self.assertEqual(result.action_type, "task_complete")


class LegacyRiskStringCompatibilityTests(unittest.TestCase):
    def test_cross_tool_registry_legacy_strings_map_to_canonical_tiers(self) -> None:
        from services.api.app.cross_tool_reasoning import build_cross_tool_registry

        registry = build_cross_tool_registry()
        for name in registry.names():
            spec = registry.get(name)
            self.assertIs(spec.canonical_risk_level, RiskLevel.L0_OBSERVE, name)
            self.assertIs(spec.effective_approval_policy, ApprovalPolicy.NEVER, name)

    def test_unknown_risk_string_fails_loudly(self) -> None:
        from services.api.app.tool_registry import ToolRegistryError, coerce_risk_level

        with self.assertRaises(ToolRegistryError):
            coerce_risk_level("totally-made-up")


if __name__ == "__main__":
    unittest.main()
