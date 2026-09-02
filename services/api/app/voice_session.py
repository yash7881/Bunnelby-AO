from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


DEFAULT_FOLLOW_UP_SECONDS = 10.0


class VoiceState(str, Enum):
    STANDBY = "standby"
    WAKE_CANDIDATE = "wake_candidate"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"
    ERROR_RECOVERY = "error_recovery"
    STOPPING = "stopping"


class InvalidVoiceTransition(RuntimeError):
    """Raised when runtime code attempts an impossible voice-state transition."""


@dataclass(frozen=True)
class VoiceTransition:
    previous: VoiceState
    current: VoiceState
    at_monotonic: float
    reason: str


_ALLOWED_TRANSITIONS: dict[VoiceState, frozenset[VoiceState]] = {
    VoiceState.STANDBY: frozenset(
        {
            VoiceState.WAKE_CANDIDATE,
            VoiceState.WAKE_DETECTED,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.WAKE_CANDIDATE: frozenset(
        {
            VoiceState.STANDBY,
            VoiceState.WAKE_DETECTED,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.WAKE_DETECTED: frozenset(
        {VoiceState.LISTENING, VoiceState.ERROR_RECOVERY, VoiceState.STOPPING}
    ),
    VoiceState.LISTENING: frozenset(
        {
            VoiceState.TRANSCRIBING,
            VoiceState.STANDBY,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.TRANSCRIBING: frozenset(
        {
            VoiceState.THINKING,
            VoiceState.STANDBY,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.THINKING: frozenset(
        {
            VoiceState.SPEAKING,
            VoiceState.FOLLOW_UP,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.SPEAKING: frozenset(
        {
            VoiceState.FOLLOW_UP,
            VoiceState.LISTENING,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.FOLLOW_UP: frozenset(
        {
            VoiceState.LISTENING,
            VoiceState.STANDBY,
            VoiceState.ERROR_RECOVERY,
            VoiceState.STOPPING,
        }
    ),
    VoiceState.ERROR_RECOVERY: frozenset(
        {VoiceState.STANDBY, VoiceState.STOPPING}
    ),
    VoiceState.STOPPING: frozenset(),
}


@dataclass
class VoiceSessionController:
    """Deterministic state and follow-up deadline for one persistent voice runtime.

    Audio capture, network dispatch, and playback stay outside this class. That keeps the
    product-critical timing rule independently testable: the follow-up deadline is created
    only when real playback completes, or at the explicit no-TTS/failure fallback point.
    """

    follow_up_seconds: float = DEFAULT_FOLLOW_UP_SECONDS
    clock: Callable[[], float] = time.monotonic
    state: VoiceState = VoiceState.STANDBY
    follow_up_deadline: float | None = None
    history: list[VoiceTransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1.0 <= float(self.follow_up_seconds) <= 60.0:
            raise ValueError("follow_up_seconds must be between 1 and 60 seconds")
        self.follow_up_seconds = float(self.follow_up_seconds)

    def transition(
        self,
        target: VoiceState,
        reason: str,
        *,
        at: float | None = None,
    ) -> VoiceTransition:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidVoiceTransition(
                f"Voice state cannot move from {self.state.value} to {target.value}."
            )
        timestamp = self.clock() if at is None else float(at)
        transition = VoiceTransition(self.state, target, timestamp, reason)
        self.state = target
        if target is not VoiceState.FOLLOW_UP:
            self.follow_up_deadline = None
        self.history.append(transition)
        return transition

    def wake_detected(self, *, at: float | None = None) -> VoiceTransition:
        return self.transition(VoiceState.WAKE_DETECTED, "strict wake phrase", at=at)

    def begin_listening(self, reason: str = "wake accepted", *, at: float | None = None) -> VoiceTransition:
        return self.transition(VoiceState.LISTENING, reason, at=at)

    def utterance_completed(self, *, at: float | None = None) -> VoiceTransition:
        return self.transition(VoiceState.TRANSCRIBING, "utterance endpoint", at=at)

    def transcription_completed(self, *, at: float | None = None) -> VoiceTransition:
        return self.transition(VoiceState.THINKING, "transcript ready", at=at)

    def speaking_started(self, *, at: float | None = None) -> VoiceTransition:
        return self.transition(VoiceState.SPEAKING, "audio playback started", at=at)

    def _enter_follow_up(self, reason: str, *, at: float | None = None) -> VoiceTransition:
        timestamp = self.clock() if at is None else float(at)
        transition = self.transition(VoiceState.FOLLOW_UP, reason, at=timestamp)
        self.follow_up_deadline = timestamp + self.follow_up_seconds
        return transition

    def playback_completed(self, *, at: float | None = None) -> VoiceTransition:
        return self._enter_follow_up("audio playback completed", at=at)

    def playback_failed(self, *, at: float | None = None) -> VoiceTransition:
        return self._enter_follow_up("TTS/playback failed safely", at=at)

    def response_completed_without_tts(self, *, at: float | None = None) -> VoiceTransition:
        return self._enter_follow_up("response completed without TTS", at=at)

    def accept_follow_up_speech(self, speech_started_at: float) -> bool:
        if self.state is not VoiceState.FOLLOW_UP or self.follow_up_deadline is None:
            raise InvalidVoiceTransition("Follow-up speech can only start inside FOLLOW_UP.")
        started = float(speech_started_at)
        if started <= self.follow_up_deadline:
            self.transition(VoiceState.LISTENING, "follow-up speech began", at=started)
            return True
        self.transition(VoiceState.STANDBY, "follow-up speech began after deadline", at=started)
        return False

    def expire_follow_up(self, *, at: float | None = None) -> bool:
        if self.state is not VoiceState.FOLLOW_UP or self.follow_up_deadline is None:
            raise InvalidVoiceTransition("Only FOLLOW_UP can expire.")
        timestamp = self.clock() if at is None else float(at)
        if timestamp < self.follow_up_deadline:
            return False
        self.transition(VoiceState.STANDBY, "follow-up deadline expired", at=timestamp)
        return True

    def barge_in(self, *, speech_started_at: float) -> VoiceTransition:
        return self.transition(
            VoiceState.LISTENING,
            "barge-in speech began",
            at=float(speech_started_at),
        )

    def recover(self, reason: str, *, at: float | None = None) -> None:
        timestamp = self.clock() if at is None else float(at)
        if self.state is VoiceState.STOPPING:
            return
        if self.state is not VoiceState.ERROR_RECOVERY:
            self.transition(VoiceState.ERROR_RECOVERY, reason, at=timestamp)
        self.transition(VoiceState.STANDBY, "recovery complete", at=timestamp)

    def stop(self, *, at: float | None = None) -> VoiceTransition | None:
        if self.state is VoiceState.STOPPING:
            return None
        return self.transition(VoiceState.STOPPING, "runtime stopping", at=at)

    def follow_up_remaining(self, *, at: float | None = None) -> float:
        if self.state is not VoiceState.FOLLOW_UP or self.follow_up_deadline is None:
            return 0.0
        timestamp = self.clock() if at is None else float(at)
        return max(0.0, self.follow_up_deadline - timestamp)
