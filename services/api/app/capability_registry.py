from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

from .risk_policy import (
    ApprovalPolicy,
    AuditPolicy,
    FreshnessPolicy,
    RiskDecision,
    RiskLevel,
    evaluate as evaluate_risk,
    validate_declaration,
)
from .tool_requests import ToolRequest, json_schema_for, request_model_for

logger = logging.getLogger(__name__)

# Part 10.2 Phase F: the Capability Registry.
#
# tool_registry.ToolSpec is a sound container but describes a *step inside a
# cross-tool plan*, and its executor signature takes a free-text instruction
# first -- which is precisely the untyped convention Part 10.2 removes. That
# module is extended (typed RiskLevel + metadata) and keeps serving the
# cross-tool planner; this module is the registry of TOP-LEVEL capabilities the
# Brain can select, where the executor accepts a validated ToolRequest and
# nothing else.


class CapabilityExecutor(Protocol):
    """Executes one validated request. Never receives raw user text alone."""

    def __call__(self, request: ToolRequest) -> Any: ...


class CapabilityVerifier(Protocol):
    """Observes post-action state and returns a verdict for the evidence ledger."""

    def __call__(self, request: ToolRequest, result: Any) -> Any: ...


@dataclass(frozen=True)
class Capability:
    """Everything Bunnelby must know to select, gate, run and verify one tool."""

    name: str
    version: str
    description: str
    request_model: type[ToolRequest]
    risk_level: RiskLevel
    approval_policy: ApprovalPolicy
    executor: CapabilityExecutor
    verifier: CapabilityVerifier | None = None
    freshness_policy: FreshnessPolicy = FreshnessPolicy.FRESH_REQUIRED
    audit_policy: AuditPolicy = AuditPolicy.SANITIZED_ARGUMENTS
    availability: Callable[[], bool] | None = None
    examples: tuple[str, ...] = ()
    selection_guidance: str = ""
    output_schema: Mapping[str, Any] | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        return json_schema_for(self.name)

    @property
    def requires_approval(self) -> bool:
        return self.risk_decision().requires_approval

    def risk_decision(self, *, model_requested_approval: bool | None = None) -> RiskDecision:
        return evaluate_risk(
            self.name,
            self.risk_level,
            self.approval_policy,
            model_requested_approval=model_requested_approval,
        )

    def is_available(self) -> bool:
        if self.availability is None:
            return True
        try:
            return bool(self.availability())
        except Exception:
            logger.warning("Capability %s availability check failed", self.name, exc_info=True)
            return False

    def catalog_entry(self) -> dict[str, Any]:
        """Machine-readable description used to build the Brain's tool catalog."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "selection_guidance": self.selection_guidance,
            "risk_level": self.risk_level.value,
            "requires_approval": self.requires_approval,
            "arguments": self.input_schema,
            "examples": list(self.examples),
        }


class CapabilityRegistryError(RuntimeError):
    """Raised when a capability cannot be registered or resolved safely."""


class CapabilityRegistry:
    """Ordered registry of top-level capabilities the Brain may select."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        name = capability.name.strip()
        if not name:
            raise CapabilityRegistryError("Capability name cannot be empty.")
        if name in self._capabilities:
            raise CapabilityRegistryError(f"Capability is already registered: {name}")
        if capability.request_model is not request_model_for(name):
            raise CapabilityRegistryError(
                f"{name} must use the request model declared in tool_requests."
            )
        # An unsafe risk/approval declaration fails at import, not at use.
        validate_declaration(name, capability.risk_level, capability.approval_policy)
        self._capabilities[name] = capability

    def get(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise CapabilityRegistryError(f"Unknown Bunnelby capability: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._capabilities

    def names(self) -> tuple[str, ...]:
        return tuple(self._capabilities)

    def selectable_names(self) -> tuple[str, ...]:
        """Names the Brain may choose: registered, available, not forbidden."""
        return tuple(
            name
            for name, capability in self._capabilities.items()
            if capability.risk_level is not RiskLevel.FORBIDDEN and capability.is_available()
        )

    def tool_names(self) -> tuple[str, ...]:
        """Selectable names excluding the conversational pseudo-capability."""
        return tuple(name for name in self.selectable_names() if name != "general_answer")

    def metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            capability.catalog_entry() for capability in self._capabilities.values()
        )

    def catalog(self, *, include_general_answer: bool = False) -> tuple[dict[str, Any], ...]:
        """Tool catalog for the Brain prompt, generated from the registry."""
        return tuple(
            self.get(name).catalog_entry()
            for name in self.selectable_names()
            if include_general_answer or name != "general_answer"
        )

    def execute(self, request: ToolRequest) -> Any:
        """Run a validated request through its own capability's executor.

        The capability is resolved from the REQUEST TYPE, so a request object can
        only ever reach the executor that matches its class. This is the
        structural half of the split-brain fix.
        """
        capability = self.get(request.tool_name)
        if not isinstance(request, capability.request_model):
            raise CapabilityRegistryError(
                f"{capability.name} executor requires {capability.request_model.__name__}, "
                f"received {type(request).__name__}."
            )
        decision = capability.risk_decision()
        if decision.blocked:
            raise CapabilityRegistryError(
                f"{capability.name} is forbidden by risk policy and was not executed."
            )
        return capability.executor(request)


REGISTRY: Final[CapabilityRegistry] = CapabilityRegistry()


def registry() -> CapabilityRegistry:
    return REGISTRY


__all__ = [
    "Capability",
    "CapabilityExecutor",
    "CapabilityRegistry",
    "CapabilityRegistryError",
    "CapabilityVerifier",
    "REGISTRY",
    "registry",
]
