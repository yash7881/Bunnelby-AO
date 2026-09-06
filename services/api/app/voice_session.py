from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


DEFAULT_FOLLOW_UP_SECONDS = 10.0

#: How many recent activation ids are remembered for duplicate suppression.
#: Small on purpose: this defends against a repeated callback for the SAME
#: activation, not against a long replay history.
MAX_REMEMBERED_ACTIVATIONS = 64

#: Consecutive turns allowed WITHOUT a wake phrase before Bunnelby insists on
#: one again. Follow-up is convenient precisely because it skips the wake word,
#: which is also what lets ambient speech feed turn after turn; this bounds it.
DEFAULT_MAX_CONSECUTIVE_FOLLOW_UPS = 2


class ConversationPhase(str, Enum):
    """Whether speech currently counts as a command.

    THE MICROPHONE IS ALWAYS ON AND THE WAKE DETECTOR IS ALWAYS LISTENING.
    None of these phases turn hardware off, stop PortAudio, or disable wake
    recognition. They describe one thing only: whether audio the runtime hears
    is permitted to become a command turn.
    """

    #: Wake-only. The stream is live and the detector is listening, but normal
    #: speech -- conversation, a song, a television -- is not a command and
    #: cannot reach capture, STT or /chat. Only the wake detector may leave
    #: this phase.
    STANDBY_WAKE = "standby_wake"
    #: A lease exists. Capture, STT, dispatch and the spoken reply are allowed.
    ACTIVE_TURN = "active_turn"
    #: The assistant has finished speaking and a bounded window is open in
    #: which the user may continue WITHOUT repeating the wake phrase.
    FOLLOW_UP = "follow_up"


class VoiceActivationDenied(RuntimeError):
    """A voice turn was refused before any capture, STT, dispatch or speech.

    Carrying the reason as a field (rather than only in the message) keeps the
    audit line machine-readable: `voice_turn rejected reason=no_active_lease`.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True)
class ConversationLease:
    """Permission to run exactly one command turn.

    A lease is minted by a wake activation, or by an accepted follow-up inside
    the window a completed response opened. It carries the authority
    `generation` it was minted under, so returning to standby invalidates every
    outstanding lease at once and a stale one can never be replayed later.
    """

    activation_id: str
    generation: int
    source: str  # "wake" | "follow_up"
    created_at: float


@dataclass
class VoiceConversationAuthority:
    """Decides whether heard speech may become a command turn.

    THE INVARIANT. Outside an ACTIVE_TURN lease, speech is never a command.
    Standby is not silence and not a disabled microphone: the wake detector is
    listening the whole time, and it is the only thing that can leave standby.
    After a response, one bounded follow-up window allows continuing without
    the wake phrase; when it expires, the wake phrase is required again.

    This object knows nothing about wake recognition. The detector may emit
    whatever it emits; this decides what the system is allowed to do next.
    That separation is what lets the wake implementation stay untouched.
    """

    max_consecutive_follow_ups: int = DEFAULT_MAX_CONSECUTIVE_FOLLOW_UPS
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._counter = itertools.count(1)
        self._seen: list[str] = []
        self._phase = ConversationPhase.STANDBY_WAKE
        self._active: ConversationLease | None = None
        self._consecutive_follow_ups = 0

    # -- observation --------------------------------------------------------- #

    @property
    def phase(self) -> ConversationPhase:
        with self._lock:
            return self._phase

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def consecutive_follow_ups(self) -> int:
        with self._lock:
            return self._consecutive_follow_ups

    def may_capture_command(self) -> bool:
        """True only while a live lease authorises a command turn.

        This is the single question the runtime asks before treating audio as
        a command. In STANDBY_WAKE it is always False, which is what makes
        ambient speech inert.
        """
        with self._lock:
            return (
                self._phase is ConversationPhase.ACTIVE_TURN
                and self._active is not None
                and self._active.generation == self._generation
            )

    def lease_is_live(self, lease: ConversationLease | None) -> bool:
        if lease is None:
            return False
        with self._lock:
            return (
                lease.generation == self._generation
                and self._active is not None
                and self._active.activation_id == lease.activation_id
            )

    # -- transitions --------------------------------------------------------- #

    def _mint(self, activation_id: str | None, source: str) -> ConversationLease:
        identifier = (
            str(activation_id)
            if activation_id
            else f"act-{self._generation}-{next(self._counter)}"
        )
        if identifier in self._seen:
            raise VoiceActivationDenied(
                "duplicate_activation",
                f"Activation {identifier} already started a turn.",
            )
        self._seen.append(identifier)
        if len(self._seen) > MAX_REMEMBERED_ACTIVATIONS:
            del self._seen[:-MAX_REMEMBERED_ACTIVATIONS]
        lease = ConversationLease(identifier, self._generation, source, self.clock())
        self._active = lease
        self._phase = ConversationPhase.ACTIVE_TURN
        return lease

    def begin_wake_turn(self, activation_id: str | None = None) -> ConversationLease:
        """Start a turn from a wake activation.

        Always permitted -- the wake phrase IS the user's consent, and the
        microphone was never off. The only refusal is a duplicate delivery of
        one physical activation, which must not become two turns.
        """
        with self._lock:
            lease = self._mint(activation_id, "wake")
            # A fresh wake is a deliberate new conversation, so the no-wake
            # budget starts over.
            self._consecutive_follow_ups = 0
            return lease

    def open_follow_up_window(self) -> None:
        """Called once the assistant's response is fully complete."""
        with self._lock:
            self._phase = ConversationPhase.FOLLOW_UP
            self._active = None

    def may_begin_follow_up(self, *, playback_blocked: bool = False) -> bool:
        """Whether a no-wake follow-up may start right now.

        `playback_blocked` is the existing external playback gate: Bunnelby
        must never hear its own speech (or its room echo) as the user's next
        command.
        """
        with self._lock:
            if self._phase is not ConversationPhase.FOLLOW_UP:
                return False
            if self._consecutive_follow_ups >= self.max_consecutive_follow_ups:
                return False
            return not playback_blocked

    def begin_follow_up_turn(
        self, activation_id: str | None = None, *, playback_blocked: bool = False
    ) -> ConversationLease:
        """Start a turn without a wake phrase, inside the open window."""
        with self._lock:
            if self._phase is not ConversationPhase.FOLLOW_UP:
                raise VoiceActivationDenied(
                    "no_follow_up_window",
                    "A follow-up may only start inside an open follow-up window.",
                )
            if self._consecutive_follow_ups >= self.max_consecutive_follow_ups:
                raise VoiceActivationDenied(
                    "follow_up_chain_limit",
                    "Consecutive no-wake follow-ups are exhausted; wake is required.",
                )
            if playback_blocked:
                raise VoiceActivationDenied(
                    "assistant_speaking",
                    "The assistant is still speaking; that audio is not a command.",
                )
            lease = self._mint(activation_id, "follow_up")
            self._consecutive_follow_ups += 1
            return lease

    def return_to_standby(self, reason: str = "conversation complete") -> str:
        """Back to wake-only. Every outstanding lease dies here.

        The microphone stream and the wake detector are untouched: this expires
        PERMISSION for no-wake conversation, nothing else.
        """
        with self._lock:
            self._generation += 1
            self._phase = ConversationPhase.STANDBY_WAKE
            self._active = None
            self._consecutive_follow_ups = 0
            return reason

    def require_live(self, lease: ConversationLease | None, stage: str) -> None:
        """Checkpoint. Raises rather than letting a dead turn continue."""
        if not self.lease_is_live(lease):
            raise VoiceActivationDenied(
                "no_active_lease",
                f"No live conversation lease before {stage}.",
            )


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
