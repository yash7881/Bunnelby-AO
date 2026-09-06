from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Final, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Part 10.2 Phase F: typed tool requests.
#
# Before this module the Brain produced a free-form `arguments` mapping that no
# executor ever read (`grep -rn "\.arguments"` returned zero hits), and every
# handler re-derived its own action class from the raw user text. Two classifiers
# disagreeing is what made `calendar_read` able to mint a Calendar write.
#
# The rule these types encode:
#
#   The REQUEST TYPE fixes the action class. A parser may fill missing FIELDS
#   inside an already-chosen request type; it may never choose a different type.
#
# Every request therefore carries `raw_message`. That is deliberate and explicit
# rather than incidental: deterministic parsers still need the original wording
# to recover a date, a title or a recipient, but because they run *inside* a
# concrete request class they can only refine it, never re-route it.

MAX_RAW_MESSAGE_CHARS: Final[int] = 8000
MAX_BODY_CHARS: Final[int] = 8000
MAX_SUBJECT_CHARS: Final[int] = 400
MAX_GMAIL_READ_LIMIT: Final[int] = 25
DEFAULT_GMAIL_READ_LIMIT: Final[int] = 10

logger = logging.getLogger(__name__)

GmailReadKind = Literal["recent", "unread"]
CalendarReadMode = Literal["agenda", "free_busy", "open_slots"]
FreshnessPolicy = Literal["cached_ok", "fresh_required"]
CrossToolSource = Literal["gmail", "calendar"]
FileSearchMode = Literal["filename", "content", "hybrid"]
DesktopActionName = Literal[
    "list_windows",
    "inspect_window",
    "find_control",
    "open_app",
    "focus_app",
    "close_app",
    "safe_shortcut",
]


class ToolRequest(BaseModel):
    """Base for every validated tool request.

    `extra="forbid"` is the point of this class: a hallucinated argument name
    from the model is a validation error, not a silently ignored field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    raw_message: str = Field(min_length=1, max_length=MAX_RAW_MESSAGE_CHARS)

    @field_validator("raw_message")
    @classmethod
    def _reject_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("raw_message contains an invalid NUL character")
        return value

    @property
    def tool_name(self) -> str:
        raise NotImplementedError

    def audit_arguments(self) -> dict[str, Any]:
        """Sanitized arguments safe to persist in tool_runs.

        Bodies and raw user text are replaced by lengths and fingerprints:
        an audit row must never become a copy of an email or a secret.
        """
        payload = self.model_dump(mode="json")
        payload.pop("raw_message", None)
        payload["raw_message_chars"] = len(self.raw_message)
        for sensitive in ("body", "body_instruction"):
            value = payload.get(sensitive)
            if isinstance(value, str) and value:
                payload[sensitive] = f"<{len(value)} chars, sha256:{_fingerprint(value)}>"
        return payload

    def request_hash(self) -> str:
        return _fingerprint(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        )


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #


class GeneralAnswerRequest(ToolRequest):
    """A conversational answer or clarification. No external system is touched."""

    reply: str = Field(default="", max_length=MAX_BODY_CHARS)
    spoken_reply: str = Field(default="", max_length=MAX_BODY_CHARS)
    is_clarification: bool = False

    @property
    def tool_name(self) -> str:
        return "general_answer"


# --------------------------------------------------------------------------- #
# Gmail
# --------------------------------------------------------------------------- #


class GmailReadRequest(ToolRequest):
    """Read the user's inbox. Cannot express a write of any kind."""

    read_kind: GmailReadKind = "recent"
    limit: int = Field(default=DEFAULT_GMAIL_READ_LIMIT, ge=1, le=MAX_GMAIL_READ_LIMIT)
    freshness: FreshnessPolicy = "fresh_required"

    @property
    def unread_only(self) -> bool:
        return self.read_kind == "unread"

    @property
    def tool_name(self) -> str:
        return "gmail_read"


class GmailComposeRequest(ToolRequest):
    """Propose a brand-new email. Always an approval proposal, never a send."""

    recipient_hint: str = Field(min_length=1, max_length=MAX_SUBJECT_CHARS)
    # Filled by deterministic recipient resolution during execution. Once set it
    # is authoritative: the approval preview and the send verifier both use it.
    recipient: str | None = Field(default=None, max_length=MAX_SUBJECT_CHARS)
    subject: str | None = Field(default=None, max_length=MAX_SUBJECT_CHARS)
    body: str | None = Field(default=None, max_length=MAX_BODY_CHARS)

    @property
    def tool_name(self) -> str:
        return "gmail_compose"

    def expected_recipient(self) -> str | None:
        return self.recipient

    def body_fingerprint(self) -> str | None:
        return _fingerprint(self.body) if self.body else None


class GmailReplyRequest(ToolRequest):
    """Propose a reply to an existing thread. Always an approval proposal."""

    thread_id: str | None = Field(default=None, max_length=256)
    sender_hint: str | None = Field(default=None, max_length=MAX_SUBJECT_CHARS)
    subject_hint: str | None = Field(default=None, max_length=MAX_SUBJECT_CHARS)
    recipient: str | None = Field(default=None, max_length=MAX_SUBJECT_CHARS)
    body_instruction: str | None = Field(default=None, max_length=MAX_BODY_CHARS)

    @property
    def tool_name(self) -> str:
        return "gmail_reply"


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


class CalendarReadRequest(ToolRequest):
    """Read the calendar.

    `mode` selects a READ submode only. There is no representable value of this
    type that creates an event, which is what closes the read->write hole.
    """

    mode: CalendarReadMode = "agenda"
    date_hint: str | None = Field(default=None, max_length=200)
    timezone_hint: str | None = Field(default=None, max_length=100)
    duration_minutes: int | None = Field(default=None, ge=5, le=24 * 60)

    @property
    def tool_name(self) -> str:
        return "calendar_read"


class CalendarCreateRequest(ToolRequest):
    """Propose a calendar event. Always an approval proposal, never a create."""

    title: str = Field(min_length=1, max_length=MAX_SUBJECT_CHARS)
    start: datetime | None = None
    end: datetime | None = None
    timezone_name: str | None = Field(default=None, max_length=100)
    attendees: tuple[str, ...] = ()
    duration_minutes: int | None = Field(default=None, ge=5, le=24 * 60)

    @property
    def tool_name(self) -> str:
        return "calendar_create"

    def event_fingerprint(self) -> str:
        """Stable identity for verification: title + exact window + attendees."""
        return _fingerprint(
            json.dumps(
                {
                    "title": self.title,
                    "start": self.start.isoformat() if self.start else None,
                    "end": self.end.isoformat() if self.end else None,
                    "timezone": self.timezone_name,
                    "attendees": sorted(self.attendees),
                },
                sort_keys=True,
            )
        )


# --------------------------------------------------------------------------- #
# Cross-tool
# --------------------------------------------------------------------------- #


class CrossToolReadRequest(ToolRequest):
    """Read Gmail AND Calendar together. Read-only by construction."""

    sources: tuple[CrossToolSource, ...] = ("gmail", "calendar")
    time_scope: str | None = Field(default=None, max_length=200)

    @field_validator("sources")
    @classmethod
    def _require_two_sources(
        cls, value: tuple[CrossToolSource, ...]
    ) -> tuple[CrossToolSource, ...]:
        unique = tuple(dict.fromkeys(value))
        if len(unique) < 2:
            raise ValueError(
                "cross_tool_read requires at least two distinct sources; a single-source "
                "request must use gmail_read or calendar_read."
            )
        return unique

    @property
    def tool_name(self) -> str:
        return "cross_tool_read"


class FileSearchRequest(ToolRequest):
    """Search only the deterministic allowlisted local index; paths are never accepted."""

    query: str = Field(min_length=1, max_length=500)
    search_mode: FileSearchMode = "hybrid"
    limit: int = Field(default=8, ge=1, le=20)
    root_scope: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    within_result_set_id: str | None = Field(default=None, max_length=128)
    freshness: FreshnessPolicy = "cached_ok"

    @field_validator("root_scope")
    @classmethod
    def _root_aliases_only(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        aliases = tuple(dict.fromkeys(item.strip().casefold() for item in value))
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item) for item in aliases):
            raise ValueError("root_scope accepts aliases only, never filesystem paths")
        return aliases

    @field_validator("extensions")
    @classmethod
    def _safe_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys((item if item.startswith(".") else "." + item).casefold() for item in value))
        if any(not re.fullmatch(r"\.[a-z0-9]{1,12}", item) for item in normalized):
            raise ValueError("invalid extension filter")
        return normalized

    @property
    def tool_name(self) -> str:
        return "file_search"

    def audit_arguments(self) -> dict[str, Any]:
        payload = super().audit_arguments()
        payload["query_chars"] = len(self.query)
        payload["query_sha256"] = _fingerprint(self.query)
        payload.pop("query", None)
        return payload


_DESKTOP_ACTIONS_NEEDING_TARGET: Final[frozenset[str]] = frozenset(
    {"inspect_window", "find_control", "open_app", "focus_app", "close_app"}
)

# Reviewed, exact synonyms for the canonical desktop actions.
#
# WHY THIS EXISTS. The provider response schema now publishes the action enum,
# so the model should emit `open_app` directly -- that is the primary contract.
# This map is resilience for provider variation, not permission: live hardware
# showed Gemini answering "Open Notepad" with action="open" while the schema
# still flattened the Literal to a bare STRING, and the turn failed closed with
# "I need a bit more detail before I can do that safely."
#
# WHAT IT IS NOT. It is a closed dictionary lookup with no default, applied only
# once `desktop_control` has ALREADY been selected. It cannot select the
# capability, cannot reach `target`, and cannot invent an action: an unmapped
# token passes through untouched and is rejected by the Literal below, failing
# the turn exactly as before. Every entry is an exact synonym of an action that
# is already policy-gated downstream -- close_app still refuses File Explorer,
# safe_shortcut still refuses Ctrl+Alt+Del -- so nothing here widens authority.
#
# "run" and "execute" are deliberately ABSENT. "run powershell -Command ..."
# must not acquire a foothold, however harmless the mapping would look.
_DESKTOP_ACTION_ALIASES: Final[Mapping[str, str]] = {
    # open_app: start a registered application
    "open": "open_app",
    "launch": "open_app",
    "start": "open_app",
    # focus_app: bring an already-running application to the front
    "focus": "focus_app",
    "switch": "focus_app",
    "switch_to": "focus_app",
    "activate": "focus_app",
    "bring_to_front": "focus_app",
    # close_app: graceful WM_CLOSE, still policy-gated per application
    "close": "close_app",
    "quit": "close_app",
    # read-only actions: no state change is possible through any of these
    "list": "list_windows",
    "list_apps": "list_windows",
    "list_applications": "list_windows",
    "windows": "list_windows",
    "inspect": "inspect_window",
    "find": "find_control",
    "find_controls": "find_control",
    # safe_shortcut: the shortcut allowlist remains the gate
    "shortcut": "safe_shortcut",
    "send_shortcut": "safe_shortcut",
}


def canonical_desktop_action(value: Any) -> str:
    """Fold a reviewed synonym to its canonical action token.

    Exact match only, after case and separator normalisation. Anything not in
    the closed map is returned unchanged so the Literal can reject it.
    """
    text = " ".join(str(value or "").strip().casefold().split())
    text = text.replace("-", "_").replace(" ", "_")
    return _DESKTOP_ACTION_ALIASES.get(text, text)


class DesktopControlRequest(ToolRequest):
    """Control the Windows desktop through the bounded Part 12.1 action set.

    `target` is an application ALIAS resolved against the deterministic
    registry -- never a path, never a command line. That is the whole reason
    this capability cannot become arbitrary code execution, so the validator
    below rejects anything path-shaped or shell-shaped before resolution is
    even attempted. `shortcut` is likewise an allowlisted id, not keystrokes.
    """

    action: DesktopActionName
    target: str = Field(default="", max_length=48)
    shortcut: str = Field(default="", max_length=48)
    control_name: str = Field(default="", max_length=120)
    control_type: str = Field(default="", max_length=40)
    max_depth: int = Field(default=4, ge=1, le=4)
    max_nodes: int = Field(default=120, ge=1, le=120)
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=30.0)

    @field_validator("action", mode="before")
    @classmethod
    def _canonical_action(cls, value: Any) -> Any:
        """Fold a reviewed synonym before the Literal is checked.

        Runs only for this capability, so it can refine an action inside an
        already-chosen desktop_control turn and can never cause one. An unknown
        token is passed straight through to fail closed.
        """
        return canonical_desktop_action(value) if isinstance(value, str) else value

    @field_validator("target")
    @classmethod
    def _alias_only(cls, value: str) -> str:
        """An application alias, never a path or a command.

        Mirrors FileSearchRequest.root_scope: the model may name a thing, it may
        never name a location or an executable.
        """
        text = " ".join(str(value or "").strip().casefold().split())
        if not text:
            return ""
        # Each token must START with a letter/digit, so switch-shaped input
        # ("powershell -Command ...", "app /quiet") cannot masquerade as an
        # alias even though its characters are individually harmless.
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:[ _-][a-z0-9]+)*", text):
            raise ValueError(
                "target must be an application alias, never a path or command line"
            )
        return text

    @field_validator("shortcut")
    @classmethod
    def _shortcut_id_only(cls, value: str) -> str:
        text = str(value or "").strip().casefold().replace(" ", "_").replace("+", "_")
        if not text:
            return ""
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", text):
            raise ValueError("shortcut must be an allowlisted shortcut id")
        return text

    @field_validator("control_type")
    @classmethod
    def _control_type_token(cls, value: str) -> str:
        text = str(value or "").strip()
        if text and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,39}", text):
            raise ValueError("control_type must be a UI Automation control-type name")
        return text

    @model_validator(mode="after")
    def _action_needs_its_target(self) -> "DesktopControlRequest":
        """A target-requiring action must actually carry a target.

        This is what makes a rejected target fail the TURN rather than quietly
        default to empty. build_request drops an invalid OPTIONAL field and
        retries, so without this check "open C:/Windows/System32/cmd.exe" would
        become a targetless open_app instead of a refusal. A model-level error
        has no single offending field name, so build_request treats it as
        blocking -- which is exactly the intent.
        """
        if self.action in _DESKTOP_ACTIONS_NEEDING_TARGET and not self.target:
            raise ValueError(
                f"action {self.action!r} requires a registered application alias "
                "as 'target' (paths and command lines are never accepted)"
            )
        if self.action == "safe_shortcut" and not self.shortcut:
            raise ValueError("action 'safe_shortcut' requires an allowlisted shortcut id")
        return self

    @property
    def tool_name(self) -> str:
        return "desktop_control"

    def audit_arguments(self) -> dict[str, Any]:
        payload = super().audit_arguments()
        # A control name is user/UI text; record its shape, not its content.
        name = payload.pop("control_name", "")
        if name:
            payload["control_name_chars"] = len(str(name))
        return payload


REQUEST_MODELS: Final[Mapping[str, type[ToolRequest]]] = {
    "general_answer": GeneralAnswerRequest,
    "gmail_read": GmailReadRequest,
    "gmail_compose": GmailComposeRequest,
    "gmail_reply": GmailReplyRequest,
    "calendar_read": CalendarReadRequest,
    "calendar_create": CalendarCreateRequest,
    "cross_tool_read": CrossToolReadRequest,
    "file_search": FileSearchRequest,
    "desktop_control": DesktopControlRequest,
}

# Requests that can produce an external side effect (always via an approval).
WRITE_REQUEST_NAMES: Final[frozenset[str]] = frozenset(
    {"gmail_compose", "gmail_reply", "calendar_create"}
)
READ_REQUEST_NAMES: Final[frozenset[str]] = frozenset(
    {"gmail_read", "calendar_read", "cross_tool_read", "file_search"}
)
# Part 12.1: desktop_control spans observe and local-modify actions, so it is
# neither a pure read nor an external write. Its per-action risk is decided by
# the controller and the application registry, not by membership here.


class ToolRequestValidationError(ValueError):
    """Raised when Brain arguments cannot form a valid request for the chosen tool."""


def request_model_for(tool_name: str) -> type[ToolRequest]:
    try:
        return REQUEST_MODELS[tool_name]
    except KeyError as exc:
        raise ToolRequestValidationError(f"Unknown Bunnelby tool: {tool_name}") from exc


def build_request(
    tool_name: str,
    raw_message: str,
    arguments: Mapping[str, Any] | None = None,
) -> ToolRequest:
    """Validate Brain-supplied arguments into a typed request.

    Three tiers of tolerance, in increasing strictness:

    1. UNKNOWN keys are dropped. A model that invents an extra hint must not be
       able to break the turn.
    2. An invalid value for an OPTIONAL field is dropped, falling back to the
       schema default. A live Gemini run supplied freshness="latest" for a
       gmail_read; refusing the whole turn over a defaulted, non-targeting field
       is a worse failure than ignoring it. The dropped value is logged.
    3. An invalid or missing REQUIRED field fails closed. For every write
       capability the required fields are exactly the safety-critical ones
       (gmail_compose.recipient_hint, calendar_create.title), so nothing that
       determines an external target is ever silently discarded.
    """
    model = request_model_for(tool_name)
    fields = model.model_fields
    required = {
        name for name, spec in fields.items() if spec.is_required() and name != "raw_message"
    }

    payload: dict[str, Any] = {"raw_message": raw_message}
    for key, value in (arguments or {}).items():
        if key in fields and key != "raw_message" and value is not None:
            payload[key] = value

    try:
        return model(**payload)
    except ValidationError as exc:
        offenders = {
            str(error["loc"][0])
            for error in exc.errors()
            if error.get("loc") and isinstance(error["loc"][0], str)
        }
        blocking = offenders & required
        droppable = {name for name in offenders - required if name in payload}
        if blocking or not droppable:
            raise ToolRequestValidationError(
                f"{tool_name} arguments failed validation: {exc}"
            ) from exc

        logger.warning(
            "Dropping invalid optional %s argument(s) %s from the brain; using defaults.",
            tool_name,
            sorted(droppable),
        )
        for name in droppable:
            payload.pop(name, None)
        try:
            return model(**payload)
        except Exception as retry_exc:
            raise ToolRequestValidationError(
                f"{tool_name} arguments failed validation: {retry_exc}"
            ) from retry_exc
    except Exception as exc:
        raise ToolRequestValidationError(
            f"{tool_name} arguments failed validation: {exc}"
        ) from exc


def is_write_request(request: ToolRequest) -> bool:
    return request.tool_name in WRITE_REQUEST_NAMES


def json_schema_for(tool_name: str) -> dict[str, Any]:
    """JSON schema for one tool's arguments, minus the plumbing field.

    Phase H generates the Brain's tool catalog from these so the routing prompt
    stops being a hand-maintained prose list.
    """
    schema = request_model_for(tool_name).model_json_schema()
    properties = dict(schema.get("properties", {}))
    properties.pop("raw_message", None)
    required = [name for name in schema.get("required", []) if name != "raw_message"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def argument_names_for(tool_name: str) -> tuple[str, ...]:
    return tuple(
        name for name in request_model_for(tool_name).model_fields if name != "raw_message"
    )


__all__ = [
    "CalendarCreateRequest",
    "CalendarReadRequest",
    "CrossToolReadRequest",
    "GeneralAnswerRequest",
    "FileSearchRequest",
    "GmailComposeRequest",
    "GmailReadRequest",
    "GmailReplyRequest",
    "READ_REQUEST_NAMES",
    "REQUEST_MODELS",
    "WRITE_REQUEST_NAMES",
    "ToolRequest",
    "ToolRequestValidationError",
    "argument_names_for",
    "build_request",
    "is_write_request",
    "json_schema_for",
    "request_model_for",
]
