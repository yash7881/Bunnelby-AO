from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping

# Part 10.2 Phase I: one authoritative risk and approval policy.
#
# Before this module, "does this need approval?" was answered in three places
# (orchestrator.gmail_handler, message_dispatch._calendar_result,
# message_dispatch._gmail_compose_result) and `risk_level` on ToolSpec was a
# free-form string that nothing enforced.
#
# The invariant: POLICY IS AUTHORITATIVE OVER MODEL PREFERENCE. A Brain that
# returns requires_approval=false for an external write is overridden here.


class RiskLevel(Enum):
    """Canonical Bunnelby action-impact tiers (PRD 27.1)."""

    L0_OBSERVE = "L0_OBSERVE"
    L1_SAFE_CONTROL = "L1_SAFE_CONTROL"
    L2_MODIFY_LOCAL = "L2_MODIFY_LOCAL"
    L3_EXTERNAL_WRITE = "L3_EXTERNAL_WRITE"
    L4_DESTRUCTIVE_SYSTEM = "L4_DESTRUCTIVE_SYSTEM"
    FORBIDDEN = "FORBIDDEN"

    @property
    def rank(self) -> int:
        return _RANKS[self]

    def at_least(self, other: "RiskLevel") -> bool:
        return self.rank >= other.rank


_RANKS: Final[Mapping[RiskLevel, int]] = {
    RiskLevel.L0_OBSERVE: 0,
    RiskLevel.L1_SAFE_CONTROL: 1,
    RiskLevel.L2_MODIFY_LOCAL: 2,
    RiskLevel.L3_EXTERNAL_WRITE: 3,
    RiskLevel.L4_DESTRUCTIVE_SYSTEM: 4,
    RiskLevel.FORBIDDEN: 5,
}


class ApprovalPolicy(Enum):
    """How a capability's approval requirement is decided."""

    NEVER = "never"
    """Read/observe: no approval, but still audited."""

    ALWAYS = "always"
    """Every invocation produces a preview the user must explicitly approve."""

    BLOCKED = "blocked"
    """Never executable, regardless of approval."""


class FreshnessPolicy(Enum):
    CACHED_OK = "cached_ok"
    FRESH_REQUIRED = "fresh_required"


class AuditPolicy(Enum):
    """How much of a request may be persisted to tool_runs."""

    SANITIZED_ARGUMENTS = "sanitized_arguments"
    """Typed fields only; bodies reduced to length + fingerprint."""

    METADATA_ONLY = "metadata_only"
    """Tool name, status and timing only."""


# Approval is required for anything at or above this tier.
APPROVAL_REQUIRED_AT_OR_ABOVE: Final[RiskLevel] = RiskLevel.L3_EXTERNAL_WRITE


@dataclass(frozen=True)
class RiskDecision:
    """The authoritative verdict for one capability invocation."""

    tool_name: str
    risk_level: RiskLevel
    approval_policy: ApprovalPolicy
    requires_approval: bool
    executable: bool
    reason: str

    @property
    def blocked(self) -> bool:
        return not self.executable


class RiskPolicyError(RuntimeError):
    """Raised when a capability declares an incoherent risk/approval policy."""


def requires_approval_for(
    risk_level: RiskLevel, approval_policy: ApprovalPolicy
) -> bool:
    """Approval requirement derived from the declared tier, not from the model.

    A capability may tighten its policy (declare ALWAYS below L3) but it can
    never loosen it: at or above L3_EXTERNAL_WRITE approval is mandatory.
    """
    if approval_policy is ApprovalPolicy.BLOCKED:
        return False
    if risk_level.at_least(APPROVAL_REQUIRED_AT_OR_ABOVE):
        return True
    return approval_policy is ApprovalPolicy.ALWAYS


def evaluate(
    tool_name: str,
    risk_level: RiskLevel,
    approval_policy: ApprovalPolicy,
    *,
    model_requested_approval: bool | None = None,
) -> RiskDecision:
    """Decide the risk posture for one invocation.

    `model_requested_approval` is accepted only so the policy can *tighten* on a
    cautious model. A model claim of False for an L3+ write is discarded.
    """
    if risk_level is RiskLevel.FORBIDDEN or approval_policy is ApprovalPolicy.BLOCKED:
        return RiskDecision(
            tool_name=tool_name,
            risk_level=RiskLevel.FORBIDDEN,
            approval_policy=ApprovalPolicy.BLOCKED,
            requires_approval=False,
            executable=False,
            reason="Capability is forbidden by policy and is never executed.",
        )

    mandated = requires_approval_for(risk_level, approval_policy)
    requires = mandated or bool(model_requested_approval)

    if mandated:
        reason = (
            f"{risk_level.value} requires explicit approval with an exact preview."
        )
    elif requires:
        reason = "Policy does not require approval, but the request asked for it."
    else:
        reason = f"{risk_level.value} is automatic within approved scope, and audited."

    return RiskDecision(
        tool_name=tool_name,
        risk_level=risk_level,
        approval_policy=approval_policy,
        requires_approval=requires,
        executable=True,
        reason=reason,
    )


def validate_declaration(
    tool_name: str, risk_level: RiskLevel, approval_policy: ApprovalPolicy
) -> None:
    """Reject a capability whose declaration would weaken the approval gate.

    Called at registration time so an unsafe declaration fails at import rather
    than at the moment a user asks for a write.
    """
    if (
        risk_level.at_least(APPROVAL_REQUIRED_AT_OR_ABOVE)
        and risk_level is not RiskLevel.FORBIDDEN
        and approval_policy is not ApprovalPolicy.ALWAYS
    ):
        raise RiskPolicyError(
            f"{tool_name} declares {risk_level.value} with approval_policy="
            f"{approval_policy.value}; anything at or above "
            f"{APPROVAL_REQUIRED_AT_OR_ABOVE.value} must declare ALWAYS."
        )
    if risk_level is RiskLevel.FORBIDDEN and approval_policy is not ApprovalPolicy.BLOCKED:
        raise RiskPolicyError(
            f"{tool_name} declares FORBIDDEN but not approval_policy=BLOCKED."
        )


__all__ = [
    "APPROVAL_REQUIRED_AT_OR_ABOVE",
    "ApprovalPolicy",
    "AuditPolicy",
    "FreshnessPolicy",
    "RiskDecision",
    "RiskLevel",
    "RiskPolicyError",
    "evaluate",
    "requires_approval_for",
    "validate_declaration",
]
