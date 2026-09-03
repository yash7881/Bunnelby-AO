from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping

from .risk_policy import ApprovalPolicy, AuditPolicy, FreshnessPolicy, RiskLevel

ToolExecutor = Callable[[str, Mapping[str, object]], object]

# Part 10.2 Phase F: ToolSpec describes one STEP inside a cross-tool plan. It is
# extended here rather than replaced -- the container, the duplicate-name guard
# and metadata() are sound. Top-level capabilities the Brain selects live in
# capability_registry.Capability, whose executor takes a validated ToolRequest
# instead of a free-text instruction.
#
# risk_level accepts the canonical RiskLevel enum; the historical free-form
# strings ("R1") still work and are mapped, so existing registrations and tests
# keep passing while new code is typed.

_LEGACY_RISK_ALIASES: Final[Mapping[str, RiskLevel]] = {
    "r0": RiskLevel.L0_OBSERVE,
    "r1": RiskLevel.L0_OBSERVE,
    "read": RiskLevel.L0_OBSERVE,
    "observe": RiskLevel.L0_OBSERVE,
    "r2": RiskLevel.L2_MODIFY_LOCAL,
    "modify": RiskLevel.L2_MODIFY_LOCAL,
    "r3": RiskLevel.L3_EXTERNAL_WRITE,
    "write": RiskLevel.L3_EXTERNAL_WRITE,
    "external_write": RiskLevel.L3_EXTERNAL_WRITE,
    "forbidden": RiskLevel.FORBIDDEN,
}


def coerce_risk_level(value: object) -> RiskLevel:
    """Map a declared risk value onto the canonical enum, or fail loudly."""
    if isinstance(value, RiskLevel):
        return value
    text = str(value or "").strip()
    try:
        return RiskLevel(text)
    except ValueError:
        pass
    alias = _LEGACY_RISK_ALIASES.get(text.casefold())
    if alias is not None:
        return alias
    raise ToolRegistryError(f"Unknown tool risk level: {value!r}")


@dataclass(frozen=True)
class ToolSpec:
    """Typed metadata for one Bunnelby tool exposed to the planner/executor."""

    name: str
    description: str
    risk_level: str
    requires_approval: bool
    executor: ToolExecutor
    version: str = "1.0"
    input_schema: Mapping[str, Any] | None = None
    output_schema: Mapping[str, Any] | None = None
    approval_policy: ApprovalPolicy | None = None
    freshness_policy: FreshnessPolicy = FreshnessPolicy.FRESH_REQUIRED
    audit_policy: AuditPolicy = AuditPolicy.SANITIZED_ARGUMENTS
    verifier: Callable[..., object] | None = None
    examples: tuple[str, ...] = ()

    @property
    def canonical_risk_level(self) -> RiskLevel:
        return coerce_risk_level(self.risk_level)

    @property
    def effective_approval_policy(self) -> ApprovalPolicy:
        if self.approval_policy is not None:
            return self.approval_policy
        return ApprovalPolicy.ALWAYS if self.requires_approval else ApprovalPolicy.NEVER


class ToolRegistryError(RuntimeError):
    """Raised when a tool cannot be registered or resolved safely."""


class ToolRegistry:
    """Small in-process registry that keeps orchestration independent from tool modules."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ToolRegistryError("Tool name cannot be empty.")
        if name in self._tools:
            raise ToolRegistryError(f"Tool is already registered: {name}")
        self._tools[name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"Unknown Bunnelby tool: {name}") from exc

    def execute(
        self,
        name: str,
        instruction: str,
        context: Mapping[str, object] | None = None,
    ) -> object:
        spec = self.get(name)
        return spec.executor(instruction, context or {})

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def metadata(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "name": spec.name,
                "description": spec.description,
                "risk_level": spec.risk_level,
                "canonical_risk_level": spec.canonical_risk_level.value,
                "requires_approval": spec.requires_approval,
                "version": spec.version,
            }
            for spec in self._tools.values()
        )


READ_TOOL_NAMES: Final[tuple[str, ...]] = ("gmail.read", "calendar.read")
