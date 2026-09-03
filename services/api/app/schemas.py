from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ApprovalStatus = Literal["pending", "approved", "rejected"]
ExecutionState = Literal["not_started", "executing", "completed", "failed", "unknown"]

MAX_CHAT_MESSAGE_CHARS = 8000


class ApprovalResponse(BaseModel):
    id: int
    task_type: str
    preview_content: str
    target: str
    status: ApprovalStatus
    execution_state: ExecutionState
    recipient: str | None = None
    subject: str | None = None
    title: str | None = None
    start: str | None = None
    end: str | None = None
    timezone: str | None = None
    attendees: list[str] | None = None
    calendar_id: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    executed_at: datetime | None = None
    result_message: str | None = None


class ApprovalDecisionResponse(BaseModel):
    approval: ApprovalResponse
    outcome: Literal[
        "sent",
        "created",
        "rejected",
        "already_sent",
        "already_created",
        "already_processing",
        "failed",
        "unknown",
    ]
    message: str
    spoken_reply: str | None = None
    spoken_language: Literal["en", "hi"] | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    # Part 10.2 Phase D. Optional on purpose: a caller that predates sessions
    # still works and simply gets a fresh isolated session for that turn.
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("message contains an invalid NUL character")
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must contain non-whitespace text")
        return normalized


class ChatResponse(BaseModel):
    reply: str
    # Echoed so a client that did not supply one can reuse the session Bunnelby
    # minted, keeping a conversation continuous across turns.
    session_id: str | None = None
    turn_id: str | None = None
    spoken_reply: str | None = None
    spoken_ack: str | None = None
    spoken_language: Literal["en", "hi"] | None = None
    action_type: str | None = None
    approval: ApprovalResponse | None = None
    latency_ms: dict[str, float] | None = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    language: Literal["en", "hi"]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("text contains an invalid NUL character")
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must contain non-whitespace content")
        return normalized


class STTResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration_seconds: float
