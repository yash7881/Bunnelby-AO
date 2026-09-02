from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

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
from services.api.app.stt_service import (
    STTServiceError,
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
DEFAULT_CONVERSATION_SILENCE_SECONDS = 1.0
DEFAULT_CONVERSATION_MIN_SPEECH_SECONDS = 0.15
DEFAULT_API_URL = "http://127.0.0.1:8000/chat"
DEFAULT_TTS_URL = "http://127.0.0.1:8000/tts"
DEFAULT_CHAT_TIMEOUT_SECONDS = 90.0
DEFAULT_TTS_TIMEOUT_SECONDS = 30.0
DEFAULT_BARGE_IN_GRACE_SECONDS = 0.35
DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS = 0.20
DEFAULT_BARGE_IN_ECHO_THRESHOLD = 0.32
DEFAULT_BARGE_IN_MIN_RMS = 0.008
READ_SAMPLES = 512


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


def create_conversation_vad(
    model_path: Path,
    *,
    min_silence_seconds: float,
    max_utterance_seconds: float,
):
    """Create VAD tuned for post-wake natural commands, not keyword spotting."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_path)
    config.silero_vad.threshold = 0.35
    config.silero_vad.min_silence_duration = min_silence_seconds
    config.silero_vad.min_speech_duration = DEFAULT_CONVERSATION_MIN_SPEECH_SECONDS
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


def _pop_segments(vad) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    while not vad.empty():
        segment = np.asarray(vad.front.samples, dtype=np.float32).reshape(-1).copy()
        vad.pop()
        if segment.size:
            segments.append(segment)
    return segments


def wait_for_wake(stream, model_path: Path, wake_model, args, stats: RuntimeStats):
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


def dispatch_to_chat(api_url: str, transcript: str, timeout: float) -> dict[str, object]:
    payload = json.dumps({"message": transcript}).encode("utf-8")
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


def _monitor_playback_for_barge_in(
    stream,
    model_path: Path,
    args,
    handle: PlaybackHandle,
    stats: RuntimeStats,
) -> tuple[CapturedUtterance | None, PlaybackResult]:
    if not args.barge_in:
        result = handle.wait(args.tts_playback_timeout)
        if result is None:
            handle.cancel()
            result = handle.wait(5.0)
        if result is None:
            raise RuntimeError("TTS playback did not stop after cancellation.")
        return None, result

    vad, window_size = create_conversation_vad(
        model_path,
        min_silence_seconds=args.conversation_silence,
        max_utterance_seconds=args.max_utterance,
    )
    pending = np.empty(0, dtype=np.float32)
    unexplained_chunks: list[np.ndarray] = []
    unexplained_started_at: float | None = None
    playback_started_at = time.monotonic()

    while not handle.done:
        samples, overflowed = _read_microphone(stream, window_size)
        if overflowed:
            stats.microphone_overflows += 1
            print("WARNING: microphone input overflow while monitoring barge-in")
        pending = _feed_vad(vad, window_size, pending, samples)
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        echo_score = handle.echo_score(samples)
        eligible = (
            now - playback_started_at >= args.barge_in_grace
            and vad.is_speech_detected()
            and rms >= args.barge_in_min_rms
            and echo_score < args.barge_in_echo_threshold
        )

        if eligible:
            if unexplained_started_at is None:
                unexplained_started_at = now
                unexplained_chunks.clear()
            unexplained_chunks.append(samples.copy())
            if now - unexplained_started_at >= args.barge_in_min_speech:
                handle.cancel()
                result = handle.wait(5.0)
                if result is None:
                    raise RuntimeError("Barge-in cancellation did not stop TTS playback.")
                initial = np.concatenate(unexplained_chunks).astype(np.float32, copy=False)
                captured = capture_conversation_turn(
                    stream,
                    model_path,
                    args,
                    wait_seconds=args.command_wait,
                    initial_samples=initial,
                    initial_speech_started_at=unexplained_started_at,
                )
                return captured, result
        else:
            unexplained_chunks.clear()
            unexplained_started_at = None

        # Completed echo segments are deliberately discarded. Wake matching is never active
        # during SPEAKING, so Bunnelby's own TTS cannot produce a wake event.
        _pop_segments(vad)

    result = handle.wait(1.0)
    if result is None:
        raise RuntimeError("TTS playback ended without a terminal lifecycle result.")
    return None, result


def _emit_metrics(metrics: TurnMetrics) -> None:
    rounded: dict[str, object] = {}
    for key, value in asdict(metrics).items():
        rounded[key] = round(value, 2) if isinstance(value, float) else value
    print("BUNNELBY_TURN_METRICS " + json.dumps(rounded, separators=(",", ":")))


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
    print("State: STANDBY - say 'Hey Bunnelby'")


def run(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    wake_model = load_wake_asr()
    device_index, device = default_microphone(args.device)
    stats = RuntimeStats()
    controller = VoiceSessionController(follow_up_seconds=args.follow_up_seconds)
    player = SoundDeviceWavPlayer(output_device=args.output_device)
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
        f"language={args.language}"
    )
    print(
        f"TTS: enabled={'YES' if args.tts else 'NO'} provider={preferred_provider()} "
        f"English={edge_voice_name('en') if preferred_provider() == 'edge' else voice_name('en')} "
        f"Hindi={edge_voice_name('hi') if preferred_provider() == 'edge' else voice_name('hi')}"
    )
    print(f"Follow-up window: {args.follow_up_seconds:.1f}s after playback completes")
    print(f"Barge-in monitor: {'ENABLED' if args.barge_in else 'DISABLED'}")
    print(f"Post-wake trailing silence: {args.conversation_silence:.2f}s")
    print(f"Maximum one-turn speech: {args.max_utterance:.0f}s")
    print(f"Dispatch to /chat: {'YES' if args.dispatch else 'NO'}")
    print("Privacy: raw microphone audio saved = NO")
    print("State: STANDBY - say 'Hey Bunnelby'")
    print()

    next_audio: CapturedUtterance | None = None
    pending_wake_latency_ms: float | None = None
    exit_code = 0

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=READ_SAMPLES,
        ) as microphone:
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
                    _state(controller, controller.wake_detected())
                    _state(controller, controller.begin_listening())
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
                        print("State: STANDBY - say 'Hey Bunnelby'")
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
                        print("State: STANDBY - next interaction requires 'Hey Bunnelby'")
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
                    result = transcribe_samples(
                        next_audio.samples,
                        sample_rate=SAMPLE_RATE,
                        language=args.language,
                    )
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
                                barge_audio, playback = _monitor_playback_for_barge_in(
                                    microphone,
                                    model_path,
                                    args,
                                    handle,
                                    stats,
                                )
                            except RuntimeError as exc:
                                handle.cancel()
                                playback = handle.wait(5.0)
                                stats.playback_failures += 1
                                failure_at = (
                                    playback.finished_at if playback else time.monotonic()
                                )
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

                            if barge_audio is not None and playback.status is PlaybackStatus.CANCELLED:
                                stats.barge_ins += 1
                                _state(
                                    controller,
                                    controller.barge_in(
                                        speech_started_at=barge_audio.speech_started_at
                                    ),
                                )
                                print("Barge-in accepted; TTS playback cancelled.")
                                next_audio = barge_audio
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
        "--barge-in-echo-threshold",
        type=float,
        default=DEFAULT_BARGE_IN_ECHO_THRESHOLD,
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
    args.barge_in_echo_threshold = max(
        0.05, min(float(args.barge_in_echo_threshold), 0.95)
    )
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
