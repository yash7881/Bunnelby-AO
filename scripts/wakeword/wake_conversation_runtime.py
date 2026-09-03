from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx

# Windows terminals may default to a legacy code page. Voice transcripts can
# contain Hindi/Hinglish/Indic Unicode, so console logging must never crash
# the persistent runtime.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.audio_playback import (
    AudioPlaybackError,
    PlaybackHandle,
    PlaybackResult,
    PlaybackStatus,
    SoundDeviceWavPlayer,
)
from services.api.app.session_service import new_session_id
from services.api.app.stt_service import (
    STTServiceError,
    TranscriptionResult,
    stt_hindi_hotwords,
    stt_runtime_profile,
    transcribe_samples,
)
from services.api.app.tts_service import (
    edge_voice_name,
    preferred_provider,
    voice_name,
)
from services.api.app.voice_session import (
    DEFAULT_FOLLOW_UP_SECONDS,
    VoiceSessionController,
    VoiceState,
    VoiceTransition,
)

if __package__:
    from .always_on_wake_listener import (
        MAX_WAKE_CANDIDATE_SECONDS,
        MIN_WAKE_CANDIDATE_SECONDS,
        SAMPLE_RATE,
        WAKE_ASR_MODEL,
        create_vad,
        default_microphone,
        ensure_silero_vad_model,
        load_wake_asr,
        transcribe_candidate,
        wake_match,
    )
else:  # Direct script execution adds this directory to sys.path.
    from always_on_wake_listener import (
        MAX_WAKE_CANDIDATE_SECONDS,
        MIN_WAKE_CANDIDATE_SECONDS,
        SAMPLE_RATE,
        WAKE_ASR_MODEL,
        create_vad,
        default_microphone,
        ensure_silero_vad_model,
        load_wake_asr,
        transcribe_candidate,
        wake_match,
    )

DEFAULT_COMMAND_WAIT_SECONDS = 8.0
DEFAULT_MAX_UTTERANCE_SECONDS = 60.0
DEFAULT_CONVERSATION_SILENCE_SECONDS = 1.35
DEFAULT_CONVERSATION_MIN_SPEECH_SECONDS = 0.15
DEFAULT_API_URL = "http://127.0.0.1:8000/chat"
DEFAULT_TTS_URL = "http://127.0.0.1:8000/tts"
DEFAULT_CHAT_TIMEOUT_SECONDS = 90.0
DEFAULT_TTS_TIMEOUT_SECONDS = 30.0
DEFAULT_BARGE_IN_GRACE_SECONDS = 0.35
DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS = 0.18
DEFAULT_BARGE_IN_ECHO_MARGIN = 2.0
DEFAULT_BARGE_IN_MIN_RMS = 0.008
READ_SAMPLES = 512

# Barge-in onset detection. Bunnelby knows exactly what it is playing, so the expected
# level of its own speaker leakage is predicted from the live playback reference and the
# measured speaker-to-microphone coupling instead of being guessed or correlated.
BARGE_IN_PREROLL_SECONDS = 0.40
BARGE_IN_REFERENCE_LOOKBACK_SECONDS = 0.30
BARGE_IN_MISS_TOLERANCE_FRAMES = 3
BARGE_IN_MIN_CALIBRATION_FRAMES = 12
BARGE_IN_COUPLING_RISE = 0.25
BARGE_IN_COUPLING_RISE_CEILING = 1.35
BARGE_IN_COUPLING_DRIFT_CEILING = 1.5
BARGE_IN_COUPLING_FALL = 0.005
BARGE_IN_INITIAL_COUPLING = 1.0
BARGE_IN_MAX_COUPLING = 4.0
BARGE_IN_REFERENCE_FLOOR = 1e-3
BARGE_IN_VAD_MIN_SPEECH_SECONDS = 0.05
# Small output blocks keep the cancellation check close to the audio device so a
# cancelled reply stops within roughly one block instead of one 1024-frame write.
BARGE_IN_PLAYBACK_BLOCK_FRAMES = 256
_INDIC_RESCUE_LANGUAGE_CODES = frozenset(
    {"bn", "gu", "kn", "ml", "mr", "pa", "ta", "te", "ur"}
)


def _sounddevice():
    """Import PortAudio only when opening the real microphone."""
    try:
        import sounddevice
    except Exception as exc:
        raise RuntimeError("sounddevice/PortAudio is unavailable for live microphone use.") from exc
    return sounddevice


UI_EVENT_PREFIX = "BUNNELBY_UI_EVENT "


def _emit_ui_event(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    print(
        UI_EVENT_PREFIX
        + json.dumps(message, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


@dataclass
class RuntimeStats:
    wake_candidates: int = 0
    wake_events: int = 0
    conversation_turns: int = 0
    empty_turns: int = 0
    stt_failures: int = 0
    chat_failures: int = 0
    tts_failures: int = 0
    playback_failures: int = 0
    barge_ins: int = 0
    follow_up_timeouts: int = 0
    microphone_overflows: int = 0
    fatal_runtime_failures: int = 0


@dataclass(frozen=True)
class CapturedUtterance:
    samples: np.ndarray
    speech_started_at: float
    completed_at: float
    overflows: int = 0

    @property
    def speech_seconds(self) -> float:
        return float(self.samples.size / SAMPLE_RATE)

    @property
    def endpoint_delay_ms(self) -> float:
        elapsed = max(0.0, self.completed_at - self.speech_started_at)
        return max(0.0, elapsed - self.speech_seconds) * 1000.0


@dataclass
class TurnMetrics:
    turn: int
    wake_asr_ms: float | None = None
    speech_capture_seconds: float = 0.0
    endpoint_delay_ms: float = 0.0
    stt_ms: float = 0.0
    backend_ms: float = 0.0
    tts_prepare_ms: float = 0.0
    tts_first_audio_ms: float | None = None
    tts_playback_ms: float = 0.0
    turn_total_ms: float = 0.0


def _state(controller: VoiceSessionController, transition: VoiceTransition) -> None:
    print(f"State: {transition.current.name} - {transition.reason}")
    _emit_ui_event(
        "state",
        state=transition.current.name.lower(),
        reason=transition.reason,
    )


def create_conversation_vad(
    model_path: Path,
    *,
    min_silence_seconds: float,
    max_utterance_seconds: float,
    min_speech_seconds: float = DEFAULT_CONVERSATION_MIN_SPEECH_SECONDS,
):
    """Create VAD tuned for post-wake natural commands, not keyword spotting."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_path)
    config.silero_vad.threshold = 0.35
    config.silero_vad.min_silence_duration = min_silence_seconds
    config.silero_vad.min_speech_duration = min_speech_seconds
    config.silero_vad.max_speech_duration = max_utterance_seconds
    config.silero_vad.window_size = READ_SAMPLES
    config.sample_rate = SAMPLE_RATE
    detector = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=max(90.0, max_utterance_seconds + 10.0),
    )
    return detector, int(config.silero_vad.window_size)


def _feed_vad(vad, window_size: int, pending: np.ndarray, samples: np.ndarray) -> np.ndarray:
    pending = np.concatenate((pending, np.asarray(samples, dtype=np.float32).reshape(-1)))
    while pending.size >= window_size:
        vad.accept_waveform(pending[:window_size])
        pending = pending[window_size:]
    return pending


def _read_microphone(stream, window_size: int) -> tuple[np.ndarray, bool]:
    samples, overflowed = stream.read(window_size)
    return np.asarray(samples, dtype=np.float32).reshape(-1), bool(overflowed)


def _discard_buffered_microphone(stream) -> int:
    """Discard frames captured before a new interpretation state begins.

    PortAudio keeps filling the one persistent input stream while wake ASR, STT, backend,
    or an interactive benchmark prompt runs. A snapshot prevents old speech from becoming
    the first utterance in the next state while avoiding an open-ended drain that could
    chase newly arriving live audio forever.
    """
    try:
        available = int(getattr(stream, "read_available", 0))
    except (TypeError, ValueError, OSError):
        return 0
    remaining = max(0, min(available, SAMPLE_RATE * 120))
    discarded = 0
    while remaining:
        frame_count = min(remaining, SAMPLE_RATE)
        try:
            stream.read(frame_count)
        except (OSError, RuntimeError):
            break
        discarded += frame_count
        remaining -= frame_count
    return discarded


def _pop_segments(vad) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    while not vad.empty():
        segment = np.asarray(vad.front.samples, dtype=np.float32).reshape(-1).copy()
        vad.pop()
        if segment.size:
            segments.append(segment)
    return segments


def wait_for_wake(stream, model_path: Path, wake_model, args, stats: RuntimeStats):
    discarded = _discard_buffered_microphone(stream)
    if discarded:
        print(f"Discarded {discarded / SAMPLE_RATE:.2f}s of pre-standby microphone buffer")
    wake_vad, window_size = create_vad(model_path)
    pending = np.empty(0, dtype=np.float32)

    while True:
        samples, overflowed = _read_microphone(stream, window_size)
        if overflowed:
            stats.microphone_overflows += 1
            print("WARNING: microphone input overflow during standby")
        pending = _feed_vad(wake_vad, window_size, pending, samples)
        for segment in _pop_segments(wake_vad):
            seconds = segment.size / SAMPLE_RATE
            if not (MIN_WAKE_CANDIDATE_SECONDS <= seconds <= MAX_WAKE_CANDIDATE_SECONDS):
                continue

            transcript, latency = transcribe_candidate(wake_model, segment)
            stats.wake_candidates += 1
            matched = wake_match(transcript)
            if args.debug_wake_transcripts:
                print(
                    f"[{'WAKE' if matched else 'speech'}] {transcript!r} "
                    f"| segment={seconds:.2f}s | asr={latency:.2f}s"
                )
            if matched:
                stats.wake_events += 1
                return transcript, latency


def capture_conversation_turn(
    stream,
    model_path: Path,
    args,
    *,
    wait_seconds: float,
    initial_samples: np.ndarray | None = None,
    initial_speech_started_at: float | None = None,
) -> CapturedUtterance | None:
    """Capture one utterance in RAM; the timeout applies only to speech onset."""
    vad, window_size = create_conversation_vad(
        model_path,
        min_silence_seconds=args.conversation_silence,
        max_utterance_seconds=args.max_utterance,
    )
    pending = np.empty(0, dtype=np.float32)
    waiting_started = time.monotonic()
    speech_started_at = initial_speech_started_at
    overflows = 0

    if initial_samples is not None and initial_samples.size:
        pending = _feed_vad(vad, window_size, pending, initial_samples)
    else:
        discarded = _discard_buffered_microphone(stream)
        if discarded:
            print(f"Discarded {discarded / SAMPLE_RATE:.2f}s of pre-listening microphone buffer")

    while True:
        samples, overflowed = _read_microphone(stream, window_size)
        if overflowed:
            overflows += 1
            print("WARNING: microphone input overflow during conversation")
        pending = _feed_vad(vad, window_size, pending, samples)
        now = time.monotonic()

        if speech_started_at is None and vad.is_speech_detected():
            speech_started_at = now
            print("Speech detected - keep talking naturally...")

        completed = _pop_segments(vad)
        if completed:
            started = speech_started_at if speech_started_at is not None else now
            return CapturedUtterance(
                np.concatenate(completed).astype(np.float32, copy=False),
                started,
                now,
                overflows,
            )

        if speech_started_at is None:
            if now - waiting_started >= max(0.0, wait_seconds):
                return None
            continue

        # Once speech begins, the onset deadline is cancelled. A user who starts at 9.8
        # seconds can finish the complete command up to the separate max-utterance bound.
        if now - speech_started_at >= args.max_utterance:
            vad.flush()
            completed = _pop_segments(vad)
            if completed:
                return CapturedUtterance(
                    np.concatenate(completed).astype(np.float32, copy=False),
                    speech_started_at,
                    time.monotonic(),
                    overflows,
                )
            return None


def dispatch_to_chat(
    api_url: str,
    transcript: str,
    timeout: float,
    session_id: str | None = None,
) -> dict[str, object]:
    """POST one transcript to /chat.

    Part 10.2 Phase D: session_id ties every turn of one wake -> follow-up
    conversation together, so a later conversation never inherits this one as
    its active topic. Omitting it keeps the pre-10.2 behavior (the backend mints
    a fresh isolated session).
    """
    body_payload: dict[str, object] = {"message": transcript}
    if session_id:
        body_payload["session_id"] = session_id
    payload = json.dumps(body_payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not reach Bunnelby API at {api_url}: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Bunnelby API returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Bunnelby API returned an unexpected response.")
    return decoded


def request_tts(tts_url: str, text: str, language: str, timeout: float) -> bytes:
    payload = json.dumps({"text": text, "language": language}).encode("utf-8")
    request = urllib.request.Request(
        tts_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            wav_bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Bunnelby TTS request failed safely: {exc}") from exc
    if len(wav_bytes) <= 44 or not wav_bytes.startswith(b"RIFF"):
        raise RuntimeError("Bunnelby TTS returned invalid WAV audio.")
    return wav_bytes


def _print_backend_latency(response: dict[str, object], round_trip_seconds: float) -> None:
    print(f"Local /chat round-trip: {round_trip_seconds:.2f}s")
    timings = response.get("latency_ms")
    if not isinstance(timings, dict) or not timings:
        return
    print("Backend latency breakdown:")
    for key, value in timings.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {float(value):.0f} ms")


def _await_playback_start(handle: PlaybackHandle, timeout: float = 3.0) -> PlaybackStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = handle.status
        if status is not PlaybackStatus.QUEUED:
            return status
        time.sleep(0.005)
    return handle.status


@dataclass(frozen=True)
class BargeInOnset:
    """One accepted barge-in onset plus every microphone frame that produced it.

    ``detected_at`` is the first frame that cleared the self-echo prediction and is the
    correct anchor for cancellation-latency measurement. ``speech_started_at`` is that
    moment backdated over the retained pre-roll, so it matches the first sample of
    ``samples`` and keeps the captured utterance's own timing honest.
    """

    speech_started_at: float
    detected_at: float
    accepted_at: float
    samples: np.ndarray
    coupling: float


@dataclass
class BargeInOutcome:
    """Authoritative record that the runtime, not the audio device, ended playback."""

    utterance: CapturedUtterance | None
    speech_started_at: float
    cancellation_latency_ms: float
    detection_latency_ms: float
    coupling: float


@dataclass
class BargeInDetector:
    """Detect real user speech during SPEAKING without triggering on Bunnelby's own audio.

    The discriminator is level-based, not correlation-based. Bunnelby always knows the
    waveform it is currently playing, so the expected microphone level of its own speaker
    leakage is ``coupling * reference_level``. ``coupling`` is the speaker-to-microphone
    gain, learned online from frames that were *not* treated as user speech, so the user's
    own voice can never raise the bar against itself. A frame only counts as an
    interruption when it clears that prediction by ``echo_margin``, clears an absolute
    noise floor, and the VAD agrees it is speech.

    Every observed frame is kept in a short pre-roll ring so the audio that occurred while
    the onset was still being confirmed is carried into the captured utterance. Without it
    the first syllable of "Wait" or "Actually" is lost.
    """

    frame_samples: int = READ_SAMPLES
    sample_rate: int = SAMPLE_RATE
    min_speech_seconds: float = DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS
    min_rms: float = DEFAULT_BARGE_IN_MIN_RMS
    echo_margin: float = DEFAULT_BARGE_IN_ECHO_MARGIN
    preroll_seconds: float = BARGE_IN_PREROLL_SECONDS
    miss_tolerance: int = BARGE_IN_MISS_TOLERANCE_FRAMES
    min_calibration_frames: int = BARGE_IN_MIN_CALIBRATION_FRAMES
    coupling: float = BARGE_IN_INITIAL_COUPLING

    _preroll: deque = field(init=False, repr=False)
    _candidate: list = field(default_factory=list, init=False, repr=False)
    _candidate_started_at: float | None = field(default=None, init=False, repr=False)
    _candidate_leading_seconds: float = field(default=0.0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)
    _calibration_frames: int = field(default=0, init=False, repr=False)
    _calibrated_coupling: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.frame_samples = max(1, int(self.frame_samples))
        frames = max(
            1,
            int(round(self.preroll_seconds * self.sample_rate / self.frame_samples)),
        )
        self._preroll = deque(maxlen=frames)

    @property
    def frame_seconds(self) -> float:
        return self.frame_samples / float(self.sample_rate)

    @property
    def calibrated(self) -> bool:
        return self._calibration_frames >= self.min_calibration_frames

    def threshold(self, reference_level: float) -> float:
        predicted_echo = max(0.0, float(reference_level)) * self.coupling
        return max(self.min_rms, predicted_echo * self.echo_margin)

    def _reset_candidate(self) -> None:
        self._candidate = []
        self._candidate_started_at = None
        self._candidate_leading_seconds = 0.0
        self._misses = 0

    def _learn_coupling(self, rms: float, reference_level: float) -> None:
        """Track the speaker-to-microphone gain from frames that are not user speech.

        A fresh detector is built for every reply, so calibration happens during the
        grace window at the start of playback: the loudest self-echo frame seen there
        seeds the estimate. That is far quicker than converging from a pessimistic
        constant, and it happens while barge-in is disarmed anyway.

        After calibration the estimate drifts only gently, and it refuses to learn from
        frames far above the current estimate. Unbounded upward adaptation is
        self-defeating: the user's own voice is by construction the loudest thing in the
        microphone, so learning from it would ratchet the bar up at exactly the moment it
        needs to hold still, and the interruption would be swallowed.
        """
        if reference_level < BARGE_IN_REFERENCE_FLOOR:
            return
        ratio = min(rms / reference_level, BARGE_IN_MAX_COUPLING)
        if self._calibration_frames == 0:
            self.coupling = ratio
        elif not self.calibrated:
            self.coupling = max(self.coupling, ratio)
        elif ratio < self.coupling:
            self.coupling += BARGE_IN_COUPLING_FALL * (ratio - self.coupling)
        elif ratio <= self.coupling * BARGE_IN_COUPLING_RISE_CEILING:
            self.coupling += BARGE_IN_COUPLING_RISE * (ratio - self.coupling)
        ceiling = BARGE_IN_MAX_COUPLING
        if self._calibrated_coupling is not None:
            # Post-calibration drift is bounded. Calibration ran while the user was not
            # yet interrupting, so it is the trustworthy anchor; without this bound a
            # loud interruption can creep the estimate upward one frame at a time until
            # it hides itself.
            ceiling = min(ceiling, self._calibrated_coupling * BARGE_IN_COUPLING_DRIFT_CEILING)
        self.coupling = min(ceiling, max(0.0, self.coupling))
        self._calibration_frames += 1
        if self._calibrated_coupling is None and self.calibrated:
            self._calibrated_coupling = self.coupling

    def observe(
        self,
        frame: np.ndarray,
        *,
        reference_level: float,
        vad_speech: bool,
        now: float,
        armed: bool,
    ) -> BargeInOnset | None:
        chunk = np.asarray(frame, dtype=np.float32).reshape(-1).copy()
        rms = (
            float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
            if chunk.size
            else 0.0
        )
        eligible = (
            armed
            and self.calibrated
            and bool(vad_speech)
            and rms >= self.threshold(reference_level)
        )

        onset: BargeInOnset | None = None
        if eligible:
            if self._candidate_started_at is None:
                self._candidate_started_at = now
                self._candidate = list(self._preroll)
                self._candidate_leading_seconds = len(self._candidate) * self.frame_seconds
            self._candidate.append(chunk)
            self._misses = 0
            if now - self._candidate_started_at >= self.min_speech_seconds:
                samples = (
                    np.concatenate(self._candidate).astype(np.float32, copy=False)
                    if self._candidate
                    else np.empty(0, dtype=np.float32)
                )
                onset = BargeInOnset(
                    speech_started_at=self._candidate_started_at
                    - self._candidate_leading_seconds,
                    detected_at=self._candidate_started_at,
                    accepted_at=now,
                    samples=samples,
                    coupling=self.coupling,
                )
                self._reset_candidate()
        elif self._candidate_started_at is not None:
            # A stop closure inside a real word ("wait", "no") drops below threshold for
            # one or two frames. Tolerate a bounded dropout and keep the audio, instead of
            # discarding a genuine interruption mid-word.
            self._misses += 1
            self._candidate.append(chunk)
            if self._misses > self.miss_tolerance:
                self._reset_candidate()
        else:
            self._learn_coupling(rms, reference_level)

        self._preroll.append(chunk)
        return onset


def _playback_reference_level(handle: PlaybackHandle, frames: int) -> float:
    """Read the current self-audio level, degrading to 0.0 for handles without it."""
    getter = getattr(handle, "reference_level", None)
    if getter is None:
        return 0.0
    try:
        return max(0.0, float(getter(frames)))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _cancelled_playback_result(
    queued_at: float,
    started_at: float,
    finished_at: float,
) -> PlaybackResult:
    return PlaybackResult(
        status=PlaybackStatus.CANCELLED,
        queued_at=queued_at,
        started_at=started_at,
        first_audio_at=None,
        finished_at=finished_at,
        frames_played=0,
        error="playback thread did not confirm cancellation",
    )


def _monitor_playback_for_barge_in(
    stream,
    model_path: Path,
    args,
    handle: PlaybackHandle,
    stats: RuntimeStats,
    *,
    on_accepted=None,
    clock=time.monotonic,
) -> tuple[BargeInOutcome | None, PlaybackResult]:
    if not args.barge_in:
        result = handle.wait(args.tts_playback_timeout)
        if result is None:
            handle.cancel()
            result = handle.wait(5.0)
        if result is None:
            raise RuntimeError("TTS playback did not stop after cancellation.")
        return None, result

    # PortAudio kept filling the persistent input stream while STT, /chat, and TTS ran.
    # Monitoring must start from live audio, otherwise onset detection is evaluated on
    # seconds-old frames that cannot possibly line up with what is playing right now.
    _discard_buffered_microphone(stream)

    vad, window_size = create_conversation_vad(
        model_path,
        min_silence_seconds=args.conversation_silence,
        max_utterance_seconds=args.max_utterance,
        min_speech_seconds=BARGE_IN_VAD_MIN_SPEECH_SECONDS,
    )
    detector = BargeInDetector(
        frame_samples=window_size,
        sample_rate=SAMPLE_RATE,
        min_speech_seconds=args.barge_in_min_speech,
        min_rms=args.barge_in_min_rms,
        echo_margin=args.barge_in_echo_margin,
    )
    pending = np.empty(0, dtype=np.float32)
    reference_frames = max(1, int(BARGE_IN_REFERENCE_LOOKBACK_SECONDS * SAMPLE_RATE))
    playback_started_at = clock()

    while not handle.done:
        samples, overflowed = _read_microphone(stream, window_size)
        if overflowed:
            stats.microphone_overflows += 1
            print("WARNING: microphone input overflow while monitoring barge-in")
        pending = _feed_vad(vad, window_size, pending, samples)
        now = clock()
        onset = detector.observe(
            samples,
            reference_level=_playback_reference_level(handle, reference_frames),
            vad_speech=bool(vad.is_speech_detected()),
            now=now,
            armed=(now - playback_started_at) >= args.barge_in_grace,
        )

        # Completed echo segments are deliberately discarded. Wake matching is never active
        # during SPEAKING, so Bunnelby's own TTS cannot produce a wake event.
        _pop_segments(vad)

        if onset is None:
            continue

        # Stop the reply first, then move the state machine, then finish capturing. The
        # cancellation request itself is what makes the outcome authoritative: whatever
        # terminal status the playback thread publishes afterwards, this was a barge-in.
        requested_at = clock()
        handle.cancel()
        if on_accepted is not None:
            on_accepted(onset.speech_started_at)

        result = handle.wait(5.0)
        if result is None:
            print("WARNING: TTS playback thread did not confirm cancellation within 5s.")
            result = _cancelled_playback_result(
                playback_started_at,
                playback_started_at,
                clock(),
            )
        cancellation_latency_ms = max(
            0.0, (result.finished_at - requested_at) * 1000.0
        )
        detection_latency_ms = max(0.0, (requested_at - onset.detected_at) * 1000.0)

        captured = capture_conversation_turn(
            stream,
            model_path,
            args,
            wait_seconds=args.command_wait,
            initial_samples=onset.samples,
            initial_speech_started_at=onset.speech_started_at,
        )
        return (
            BargeInOutcome(
                utterance=captured,
                speech_started_at=onset.speech_started_at,
                cancellation_latency_ms=cancellation_latency_ms,
                detection_latency_ms=detection_latency_ms,
                coupling=onset.coupling,
            ),
            result,
        )

    result = handle.wait(1.0)
    if result is None:
        raise RuntimeError("TTS playback ended without a terminal lifecycle result.")
    return None, result


def _emit_metrics(metrics: TurnMetrics) -> None:
    rounded: dict[str, object] = {}
    for key, value in asdict(metrics).items():
        rounded[key] = round(value, 2) if isinstance(value, float) else value
    print("BUNNELBY_TURN_METRICS " + json.dumps(rounded, separators=(",", ":")))


def _lexical_character_count(text: str) -> int:
    return sum(
        1
        for character in text
        if unicodedata.category(character)[:1] in {"L", "N"}
    )


def _transcript_units(text: str) -> list[str]:
    return [
        item.casefold()
        for item in re.findall(
            r"[A-Za-z0-9@._%+\-]+|[\u0900-\u097f]+",
            text,
        )
        if item.strip()
    ]


def _is_pathological_transcript(text: str) -> bool:
    """Reject obvious ASR hallucination/repetition before it reaches /chat."""
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return True

    non_space = [character for character in normalized if not character.isspace()]
    lexical = _lexical_character_count(normalized)

    if not non_space:
        return True

    # Examples:
    #   "? ? ? ? ?..."
    #   "? ? ? ? ?..."
    if lexical == 0 and len(non_space) >= 3:
        return True

    lexical_ratio = lexical / len(non_space)

    # Real words followed by dozens of punctuation hallucinations.
    if len(non_space) >= 10 and lexical_ratio < 0.45:
        return True

    units = _transcript_units(normalized)
    if len(units) >= 6:
        frequencies: dict[str, int] = {}
        for unit in units:
            frequencies[unit] = frequencies.get(unit, 0) + 1

        dominant = max(frequencies.values(), default=0) / len(units)
        unique_ratio = len(frequencies) / len(units)

        if dominant >= 0.70 and unique_ratio <= 0.35:
            return True

    return False


def _transcription_quality_score(result: TranscriptionResult) -> float:
    text = str(result.text or "").strip()

    if not text:
        return -10000.0

    lexical = _lexical_character_count(text)
    units = _transcript_units(text)
    unique = len(set(units))

    score = (
        lexical
        + unique * 3.0
        + float(result.language_probability) * 5.0
    )

    if _is_pathological_transcript(text):
        score -= 10000.0

    return score


def _devanagari_count(text: str) -> int:
    return sum("\u0900" <= character <= "\u097f" for character in text)


def _needs_hindi_rescue(result: TranscriptionResult, language_mode: str) -> bool:
    if language_mode != "auto" or result.language == "hi" or not result.text.strip():
        return False
    if _devanagari_count(result.text) >= 2 and result.language_probability < 0.75:
        return True
    return (
        result.language in _INDIC_RESCUE_LANGUAGE_CODES
        and result.language_probability < 0.50
    )


def _prefer_hindi_rescue(
    automatic: TranscriptionResult,
    hindi: TranscriptionResult,
) -> bool:
    if not hindi.text.strip() or hindi.language != "hi":
        return False
    if hindi.language_probability <= automatic.language_probability:
        return False
    return _devanagari_count(hindi.text) >= _devanagari_count(automatic.text)


def _transcribe_conversation(samples: np.ndarray, language_mode: str) -> TranscriptionResult:
    automatic = transcribe_samples(
        samples,
        sample_rate=SAMPLE_RATE,
        language=language_mode,
    )

    # Do not ever send obvious Whisper hallucination/repetition directly to Brain.
    if _is_pathological_transcript(automatic.text):
        print(
            "Suspicious STT transcript detected; running bounded multilingual rescue."
        )

        candidates = [automatic]

        if language_mode == "auto":
            hindi = transcribe_samples(
                samples,
                sample_rate=SAMPLE_RATE,
                language="hi",
                hotwords_override=stt_hindi_hotwords(),
            )
            candidates.append(hindi)

            english = transcribe_samples(
                samples,
                sample_rate=SAMPLE_RATE,
                language="en",
            )
            candidates.append(english)

        selected = max(candidates, key=_transcription_quality_score)

        if not _is_pathological_transcript(selected.text):
            print(
                "STT rescue selected a higher-quality transcript "
                f"(language={selected.language})."
            )
            return selected

        print(
            "All STT candidates failed transcript-quality checks; "
            "nothing will be dispatched to Brain."
        )

        return TranscriptionResult(
            text="",
            language=automatic.language,
            language_probability=automatic.language_probability,
            duration_seconds=automatic.duration_seconds,
        )

    # Preserve the established low-confidence Indic rescue path.
    if not _needs_hindi_rescue(automatic, language_mode):
        return automatic

    print(
        "Low-confidence Indic auto-detection; running one bounded Hindi rescue inference."
    )

    hindi = transcribe_samples(
        samples,
        sample_rate=SAMPLE_RATE,
        language="hi",
        hotwords_override=stt_hindi_hotwords(),
    )

    if _prefer_hindi_rescue(automatic, hindi):
        print("Hindi rescue selected by language-confidence/script policy.")
        return hindi

    print("Hindi rescue rejected; retaining the original auto transcript.")
    return automatic

def _env_follow_up_seconds() -> float:
    raw = os.getenv("FOLLOW_UP_SECONDS", "").strip()
    if not raw:
        return DEFAULT_FOLLOW_UP_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_FOLLOW_UP_SECONDS
    return max(1.0, min(value, 60.0))


def _standby_after_failure(
    controller: VoiceSessionController,
    reason: str,
) -> None:
    controller.recover(reason)
    print(f"Recovered safely: {reason}")
    print("State: STANDBY - say 'Hey Bunnelby' or 'Hello Bunnelby'")
    _emit_ui_event("runtime_error", message=reason)
    _emit_ui_event("state", state="standby", reason="safe recovery")


def run(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    wake_model = load_wake_asr()
    device_index, device = default_microphone(args.device)
    stats = RuntimeStats()
    controller = VoiceSessionController(follow_up_seconds=args.follow_up_seconds)
    player = SoundDeviceWavPlayer(
        output_device=args.output_device,
        block_frames=BARGE_IN_PLAYBACK_BLOCK_FRAMES,
    )

    def accept_barge_in(speech_started_at: float) -> None:
        """Move SPEAKING -> LISTENING the moment cancellation is requested."""
        stats.barge_ins += 1
        _state(controller, controller.barge_in(speech_started_at=speech_started_at))
        print("Barge-in accepted; TTS playback cancelled.")

    stt_profile = stt_runtime_profile()

    print()
    print("=" * 78)
    print("BUNNELBY PERSISTENT VOICE RUNTIME - PART 10 / 10.1")
    print("=" * 78)
    print(f"Microphone: [{device_index}] {device['name']}")
    print(f"Wake: model={WAKE_ASR_MODEL} device=cpu compute=int8")
    print(
        "Conversation STT: "
        f"model={stt_profile.model} device={stt_profile.device} "
        f"compute={stt_profile.compute_type} beam={stt_profile.beam_size} "
        f"language={args.language} context={'enabled' if stt_profile.hotwords else 'disabled'}"
    )
    print(
        f"TTS: enabled={'YES' if args.tts else 'NO'} provider={preferred_provider()} "
        f"English={edge_voice_name('en') if preferred_provider() == 'edge' else voice_name('en')} "
        f"Hindi={edge_voice_name('hi') if preferred_provider() == 'edge' else voice_name('hi')}"
    )
    print(f"Follow-up window: {args.follow_up_seconds:.1f}s after playback completes")
    print(
        f"Barge-in monitor: {'ENABLED' if args.barge_in else 'DISABLED'} "
        f"grace={args.barge_in_grace:.2f}s confirm={args.barge_in_min_speech:.2f}s "
        f"echo-margin={args.barge_in_echo_margin:.2f}x floor={args.barge_in_min_rms:.3f}"
    )
    print(f"Post-wake trailing silence: {args.conversation_silence:.2f}s")
    print(f"Maximum one-turn speech: {args.max_utterance:.0f}s")
    print(f"Dispatch to /chat: {'YES' if args.dispatch else 'NO'}")
    print("Privacy: raw microphone audio saved = NO")
    print("State: STANDBY - say 'Hey Bunnelby' or 'Hello Bunnelby'")
    _emit_ui_event(
        "runtime_ready",
        wake_phrases=["Hey Bunnelby", "Hello Bunnelby"],
        raw_audio_saved=False,
        tts_provider=preferred_provider(),
    )
    _emit_ui_event("state", state="standby", reason="runtime ready")
    print()

    next_audio: CapturedUtterance | None = None
    pending_wake_latency_ms: float | None = None
    exit_code = 0

    try:
        sd = _sounddevice()
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=READ_SAMPLES,
        ) as microphone:
            conversation_session_id: str | None = None
            while args.turns == 0 or stats.conversation_turns < args.turns:
                if controller.state is VoiceState.STANDBY:
                    try:
                        wake_text, wake_latency = wait_for_wake(
                            microphone,
                            model_path,
                            wake_model,
                            args,
                            stats,
                        )
                    except Exception as exc:
                        _standby_after_failure(controller, f"wake path failed: {exc}")
                        time.sleep(0.25)
                        continue
                    print()
                    print(f"WAKE DETECTED: {wake_text!r} ({wake_latency:.2f}s ASR)")
                    _emit_ui_event(
                        "wake_detected",
                        transcript=wake_text,
                        latency_seconds=round(wake_latency, 4),
                    )
                    _state(controller, controller.wake_detected())
                    _state(controller, controller.begin_listening())
                    # A fresh wake starts a new conversation. Follow-up turns
                    # below reuse this id; returning to STANDBY and waking again
                    # mints another one.
                    conversation_session_id = new_session_id()
                    print(f"Conversation session: {conversation_session_id}")
                    pending_wake_latency_ms = wake_latency * 1000.0
                    next_audio = capture_conversation_turn(
                        microphone,
                        model_path,
                        args,
                        wait_seconds=args.command_wait,
                    )
                    if next_audio is None:
                        stats.empty_turns += 1
                        controller.transition(VoiceState.STANDBY, "no command after wake")
                        print("No command began before timeout.")
                        print("State: STANDBY - say 'Hey Bunnelby' or 'Hello Bunnelby'")
                        _emit_ui_event(
                            "state",
                            state="standby",
                            reason="no command after wake",
                        )
                        continue

                if controller.state is VoiceState.FOLLOW_UP:
                    remaining = controller.follow_up_remaining()
                    print(f"Follow-up listening: {remaining:.1f}s; wake phrase not required")
                    next_audio = capture_conversation_turn(
                        microphone,
                        model_path,
                        args,
                        wait_seconds=remaining,
                    )
                    if next_audio is None:
                        stats.follow_up_timeouts += 1
                        controller.expire_follow_up(at=time.monotonic())
                        print("Follow-up window expired.")
                        print(
                            "State: STANDBY - next interaction requires "
                            "'Hey Bunnelby' or 'Hello Bunnelby'"
                        )
                        _emit_ui_event(
                            "state",
                            state="standby",
                            reason="follow-up window expired",
                        )
                        pending_wake_latency_ms = None
                        continue
                    if not controller.accept_follow_up_speech(next_audio.speech_started_at):
                        print("Speech began after the follow-up deadline and was not executed.")
                        next_audio = None
                        pending_wake_latency_ms = None
                        continue
                    print("Follow-up accepted without wake phrase.")
                    pending_wake_latency_ms = None

                if controller.state is not VoiceState.LISTENING or next_audio is None:
                    continue

                stats.microphone_overflows += next_audio.overflows
                turn_started_at = next_audio.speech_started_at
                metrics = TurnMetrics(
                    turn=stats.conversation_turns + 1,
                    wake_asr_ms=pending_wake_latency_ms,
                    speech_capture_seconds=next_audio.speech_seconds,
                    endpoint_delay_ms=next_audio.endpoint_delay_ms,
                )
                print(f"Utterance captured in RAM: {next_audio.speech_seconds:.2f}s")
                _state(controller, controller.utterance_completed(at=next_audio.completed_at))

                try:
                    stt_started = time.perf_counter()
                    result = _transcribe_conversation(next_audio.samples, args.language)
                    metrics.stt_ms = (time.perf_counter() - stt_started) * 1000.0
                except STTServiceError as exc:
                    stats.stt_failures += 1
                    next_audio = None
                    _standby_after_failure(controller, f"conversation STT failed: {exc}")
                    continue

                transcript = result.text.strip()
                if not transcript:
                    stats.empty_turns += 1
                    next_audio = None
                    _standby_after_failure(controller, "conversation STT returned no text")
                    continue

                stats.conversation_turns += 1
                metrics.turn = stats.conversation_turns
                print()
                print("BUNNELBY CONVERSATION TRANSCRIPT")
                print(f"Text: {transcript}")
                print(
                    f"Language: {result.language} "
                    f"(confidence={result.language_probability:.3f})"
                )
                print("Raw audio saved: NO")
                _emit_ui_event(
                    "user_transcript",
                    text=transcript,
                    language=result.language,
                    language_probability=round(result.language_probability, 4),
                )
                _state(controller, controller.transcription_completed())
                next_audio = None

                if not args.dispatch:
                    controller.response_completed_without_tts()
                    controller.expire_follow_up(at=controller.follow_up_deadline)
                    metrics.turn_total_ms = (time.monotonic() - turn_started_at) * 1000.0
                    _emit_metrics(metrics)
                    print("Diagnostic transcription-only turn complete; returning to STANDBY.")
                    continue

                try:
                    dispatch_started = time.perf_counter()
                    response = dispatch_to_chat(
                        args.api_url,
                        transcript,
                        args.chat_timeout,
                        session_id=conversation_session_id,
                    )
                    dispatch_seconds = time.perf_counter() - dispatch_started
                    metrics.backend_ms = dispatch_seconds * 1000.0
                    _print_backend_latency(response, dispatch_seconds)
                except RuntimeError as exc:
                    stats.chat_failures += 1
                    _standby_after_failure(controller, f"chat failed; transcript preserved: {exc}")
                    metrics.turn_total_ms = (time.monotonic() - turn_started_at) * 1000.0
                    _emit_metrics(metrics)
                    continue

                reply = str(response.get("reply") or "").strip()
                spoken = str(
                    response.get("spoken_reply") or response.get("spoken_ack") or ""
                ).strip()
                spoken_language = str(response.get("spoken_language") or "").casefold()
                if spoken_language not in {"en", "hi"}:
                    spoken_language = "hi" if result.language == "hi" else "en"
                print(f"Assistant reply: {reply or '[empty]'}")
                print(f"Spoken reply: {spoken or '[empty]'}")
                _emit_ui_event(
                    "assistant_response",
                    reply=reply,
                    spoken_reply=spoken,
                    spoken_language=spoken_language,
                    approval=response.get("approval"),
                    action_type=response.get("action_type"),
                )

                if not args.tts or not spoken:
                    transition = controller.response_completed_without_tts(at=time.monotonic())
                    _state(controller, transition)
                    print("TTS disabled/empty; follow-up timer uses response completion fallback.")
                else:
                    tts_started_at = time.monotonic()
                    try:
                        wav_bytes = request_tts(
                            args.tts_url,
                            spoken,
                            spoken_language,
                            args.tts_timeout,
                        )
                        metrics.tts_prepare_ms = (time.monotonic() - tts_started_at) * 1000.0
                        handle = player.start(wav_bytes)
                    except (RuntimeError, AudioPlaybackError) as exc:
                        stats.tts_failures += 1
                        _state(controller, controller.playback_failed(at=time.monotonic()))
                        print(f"TTS failed safely; screen reply remains available: {exc}")
                    else:
                        start_status = _await_playback_start(handle)
                        if start_status is PlaybackStatus.STARTED:
                            _state(controller, controller.speaking_started())
                            try:
                                barge_outcome, playback = _monitor_playback_for_barge_in(
                                    microphone,
                                    model_path,
                                    args,
                                    handle,
                                    stats,
                                    on_accepted=accept_barge_in,
                                )
                            except RuntimeError as exc:
                                handle.cancel()
                                playback = handle.wait(5.0)
                                stats.playback_failures += 1
                                failure_at = (
                                    playback.finished_at if playback else time.monotonic()
                                )
                                if controller.state is VoiceState.SPEAKING:
                                    _state(
                                        controller,
                                        controller.playback_failed(at=failure_at),
                                    )
                                print(
                                    "Playback monitoring failed safely; screen reply remains "
                                    f"available: {exc}"
                                )
                                metrics.turn_total_ms = (
                                    time.monotonic() - turn_started_at
                                ) * 1000.0
                                _emit_metrics(metrics)
                                pending_wake_latency_ms = None
                                continue
                            metrics.tts_playback_ms = playback.playback_ms
                            if playback.first_audio_at is not None:
                                metrics.tts_first_audio_ms = max(
                                    0.0,
                                    (playback.first_audio_at - tts_started_at) * 1000.0,
                                )

                            # The barge-in outcome, not the playback status, decides the
                            # next state. A "completed" result that lands after the runtime
                            # already cancelled is a late callback from a race the user has
                            # already won, and must never rewrite LISTENING into FOLLOW_UP.
                            if barge_outcome is not None:
                                print(
                                    "Barge-in cancellation latency: "
                                    f"{barge_outcome.cancellation_latency_ms:.0f} ms "
                                    f"(onset to cancel {barge_outcome.detection_latency_ms:.0f} ms, "
                                    f"self-echo coupling {barge_outcome.coupling:.2f})"
                                )
                                next_audio = barge_outcome.utterance
                                if next_audio is None:
                                    controller.transition(
                                        VoiceState.STANDBY,
                                        "barge-in produced no utterance",
                                    )
                                    stats.empty_turns += 1
                                    print("Barge-in produced no usable utterance.")
                                    print(
                                        "State: STANDBY - say 'Hey Bunnelby' or "
                                        "'Hello Bunnelby'"
                                    )
                                    _emit_ui_event(
                                        "state",
                                        state="standby",
                                        reason="barge-in produced no utterance",
                                    )
                            elif controller.state is not VoiceState.SPEAKING:
                                print(
                                    "Late playback result ignored; runtime already left "
                                    f"SPEAKING ({playback.status.value})."
                                )
                            elif playback.status is PlaybackStatus.COMPLETED:
                                _state(
                                    controller,
                                    controller.playback_completed(at=playback.finished_at),
                                )
                                print("TTS playback completed; fresh follow-up window started.")
                            else:
                                stats.playback_failures += 1
                                _state(
                                    controller,
                                    controller.playback_failed(at=playback.finished_at),
                                )
                                print(
                                    "Playback failed safely; screen reply remains available: "
                                    f"{playback.error or playback.status.value}"
                                )
                        else:
                            handle.cancel()
                            playback = handle.wait(1.0)
                            stats.playback_failures += 1
                            failure_at = playback.finished_at if playback else time.monotonic()
                            _state(controller, controller.playback_failed(at=failure_at))
                            print("TTS playback could not start; follow-up fallback timer started.")

                metrics.turn_total_ms = (time.monotonic() - turn_started_at) * 1000.0
                _emit_metrics(metrics)
                pending_wake_latency_ms = None

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping Bunnelby voice runtime.")
    except Exception as exc:
        print(f"Voice runtime stopped safely after an unrecoverable device/runtime error: {exc}")
        stats.fatal_runtime_failures += 1
        exit_code = 1
    finally:
        controller.stop()

    print()
    print("=" * 78)
    print("BUNNELBY VOICE RUNTIME RESULT")
    print("=" * 78)
    print(f"Wake candidates: {stats.wake_candidates}")
    print(f"Wake events: {stats.wake_events}")
    print(f"Completed conversation turns: {stats.conversation_turns}")
    print(f"Empty/timed-out turns: {stats.empty_turns}")
    print(f"STT failures: {stats.stt_failures}")
    print(f"Chat failures: {stats.chat_failures}")
    print(f"TTS failures: {stats.tts_failures}")
    print(f"Playback failures: {stats.playback_failures}")
    print(f"Barge-ins: {stats.barge_ins}")
    print(f"Follow-up timeouts: {stats.follow_up_timeouts}")
    print(f"Microphone overflows: {stats.microphone_overflows}")
    print(f"Fatal runtime failures: {stats.fatal_runtime_failures}")
    print("Raw audio saved: NO")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent Bunnelby wake, conversation, TTS, and follow-up runtime."
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Completed turns before exit; default 0 runs persistently.",
    )
    parser.add_argument(
        "--command-wait",
        type=float,
        default=DEFAULT_COMMAND_WAIT_SECONDS,
        help="Seconds to wait for speech after wake detection.",
    )
    parser.add_argument(
        "--follow-up-seconds",
        type=float,
        default=_env_follow_up_seconds(),
        help="Speech-onset window after actual TTS completion (default: 10 seconds).",
    )
    parser.add_argument(
        "--max-utterance",
        type=float,
        default=DEFAULT_MAX_UTTERANCE_SECONDS,
        help="Maximum duration for one user command.",
    )
    parser.add_argument(
        "--conversation-silence",
        type=float,
        default=DEFAULT_CONVERSATION_SILENCE_SECONDS,
        help="Trailing silence that ends a conversational command.",
    )
    parser.add_argument("--language", choices=("auto", "en", "hi"), default="auto")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--tts-url", default=None)
    parser.add_argument("--chat-timeout", type=float, default=DEFAULT_CHAT_TIMEOUT_SECONDS)
    parser.add_argument("--tts-timeout", type=float, default=DEFAULT_TTS_TIMEOUT_SECONDS)
    parser.add_argument("--tts-playback-timeout", type=float, default=180.0)
    parser.set_defaults(dispatch=True, tts=True, barge_in=True)
    parser.add_argument("--no-dispatch", dest="dispatch", action="store_false")
    parser.add_argument("--dispatch", dest="dispatch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-tts", dest="tts", action="store_false")
    parser.add_argument("--no-barge-in", dest="barge_in", action="store_false")
    parser.add_argument("--barge-in-grace", type=float, default=DEFAULT_BARGE_IN_GRACE_SECONDS)
    parser.add_argument(
        "--barge-in-min-speech",
        type=float,
        default=DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS,
    )
    parser.add_argument(
        "--barge-in-echo-margin",
        type=float,
        default=DEFAULT_BARGE_IN_ECHO_MARGIN,
        help=(
            "How far above Bunnelby's measured self-echo level a microphone frame must "
            "rise before it counts as a real interruption."
        ),
    )
    parser.add_argument("--barge-in-min-rms", type=float, default=DEFAULT_BARGE_IN_MIN_RMS)
    parser.add_argument("--debug-wake-transcripts", action="store_true")
    args = parser.parse_args(argv)
    args.turns = max(0, min(int(args.turns), 1000))
    args.command_wait = max(3.0, min(float(args.command_wait), 30.0))
    args.follow_up_seconds = max(1.0, min(float(args.follow_up_seconds), 60.0))
    args.max_utterance = max(5.0, min(float(args.max_utterance), 120.0))
    args.conversation_silence = max(0.5, min(float(args.conversation_silence), 2.0))
    args.chat_timeout = max(5.0, min(float(args.chat_timeout), 180.0))
    args.tts_timeout = max(2.0, min(float(args.tts_timeout), 60.0))
    args.tts_playback_timeout = max(10.0, min(float(args.tts_playback_timeout), 300.0))
    args.barge_in_grace = max(0.0, min(float(args.barge_in_grace), 2.0))
    args.barge_in_min_speech = max(0.10, min(float(args.barge_in_min_speech), 1.0))
    args.barge_in_echo_margin = max(1.2, min(float(args.barge_in_echo_margin), 10.0))
    args.barge_in_min_rms = max(0.001, min(float(args.barge_in_min_rms), 0.25))
    if args.tts_url is None:
        args.tts_url = (
            args.api_url[: -len("/chat")] + "/tts"
            if args.api_url.endswith("/chat")
            else DEFAULT_TTS_URL
        )
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
