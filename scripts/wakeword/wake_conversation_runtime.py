from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.stt_service import STTServiceError, transcribe_samples

from always_on_wake_listener import (
    MAX_WAKE_CANDIDATE_SECONDS,
    MIN_WAKE_CANDIDATE_SECONDS,
    SAMPLE_RATE,
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
READ_SAMPLES = 512


@dataclass
class RuntimeStats:
    wake_candidates: int = 0
    wake_events: int = 0
    conversation_turns: int = 0
    empty_turns: int = 0
    stt_failures: int = 0


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


def _read_into_vad(stream, vad, window_size: int, pending: np.ndarray) -> np.ndarray:
    samples, overflowed = stream.read(window_size)
    if overflowed:
        print("WARNING: microphone input overflow")
    pending = np.concatenate((pending, np.asarray(samples, dtype=np.float32).reshape(-1)))
    while pending.size >= window_size:
        vad.accept_waveform(pending[:window_size])
        pending = pending[window_size:]
    return pending


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
        pending = _read_into_vad(stream, wake_vad, window_size, pending)
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


def capture_conversation_turn(stream, model_path: Path, args) -> np.ndarray | None:
    """Capture one post-wake utterance entirely in RAM.

    Speech may continue for up to max_utterance seconds. Natural pauses shorter than
    conversation_silence remain inside the same turn. Once that trailing-silence threshold
    is crossed, Silero emits the completed utterance and the turn is finalized.
    """
    vad, window_size = create_conversation_vad(
        model_path,
        min_silence_seconds=args.conversation_silence,
        max_utterance_seconds=args.max_utterance,
    )
    pending = np.empty(0, dtype=np.float32)
    waiting_started = time.monotonic()
    speech_started_at: float | None = None

    while True:
        pending = _read_into_vad(stream, vad, window_size, pending)
        now = time.monotonic()

        if speech_started_at is None and vad.is_speech_detected():
            speech_started_at = now
            print("Speech detected — keep talking naturally...")

        completed = _pop_segments(vad)
        if completed:
            return np.concatenate(completed).astype(np.float32, copy=False)

        if speech_started_at is None:
            if now - waiting_started >= args.command_wait:
                return None
            continue

        if now - speech_started_at >= args.max_utterance:
            vad.flush()
            completed = _pop_segments(vad)
            if completed:
                return np.concatenate(completed).astype(np.float32, copy=False)
            return None


def dispatch_to_chat(api_url: str, transcript: str) -> dict[str, object]:
    payload = json.dumps({"message": transcript}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Bunnelby API at {api_url}: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Bunnelby API returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Bunnelby API returned an unexpected response.")
    return decoded


def _print_backend_latency(response: dict[str, object], round_trip_seconds: float) -> None:
    print(f"Local /chat round-trip: {round_trip_seconds:.2f}s")
    timings = response.get("latency_ms")
    if not isinstance(timings, dict) or not timings:
        return

    print("Backend latency breakdown:")
    preferred_order = (
        "planner_ms",
        "tool_gmail.read_ms",
        "tool_calendar.read_ms",
        "tools_wall_ms",
        "synthesis_ms",
        "cross_tool_total_ms",
    )
    shown: set[str] = set()
    for key in preferred_order:
        value = timings.get(key)
        if isinstance(value, (int, float)):
            print(f"  {key}: {float(value):.0f} ms")
            shown.add(key)
    for key, value in timings.items():
        if key in shown or not isinstance(value, (int, float)):
            continue
        print(f"  {key}: {float(value):.0f} ms")


def run(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    wake_model = load_wake_asr()
    device_index, device = default_microphone(args.device)
    stats = RuntimeStats()

    print()
    print("=" * 78)
    print("BUNNELBY WAKE -> LONG CONVERSATION RUNTIME")
    print("=" * 78)
    print(f"Microphone: [{device_index}] {device['name']}")
    print("Standby wake engine: Silero VAD -> faster-whisper base.en -> strict matcher")
    print("Conversation STT: faster-whisper small multilingual CPU/int8")
    print(f"Post-wake trailing silence: {args.conversation_silence:.2f}s")
    print(f"Maximum one-turn speech: {args.max_utterance:.0f}s")
    print(f"Wait for command after wake: {args.command_wait:.0f}s")
    print(f"Dispatch to /chat: {'YES' if args.dispatch else 'NO'}")
    print("Raw microphone audio saved: NO")
    print("State: STANDBY — say 'Hey Bunnelby'")
    print()

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=READ_SAMPLES,
        ) as microphone:
            while args.turns == 0 or stats.conversation_turns < args.turns:
                wake_text, wake_latency = wait_for_wake(
                    microphone,
                    model_path,
                    wake_model,
                    args,
                    stats,
                )
                print()
                print(f"WAKE DETECTED: {wake_text!r} ({wake_latency:.2f}s ASR)")
                print("State: LISTENING — speak your full command now.")

                audio = capture_conversation_turn(microphone, model_path, args)
                if audio is None or audio.size == 0:
                    stats.empty_turns += 1
                    print("No command completed before timeout. Returning to STANDBY.")
                    print("State: STANDBY — say 'Hey Bunnelby'")
                    continue

                captured_seconds = audio.size / SAMPLE_RATE
                print(f"Utterance captured in RAM: {captured_seconds:.2f}s")
                print("State: TRANSCRIBING")

                try:
                    started = time.perf_counter()
                    result = transcribe_samples(
                        audio,
                        sample_rate=SAMPLE_RATE,
                        language=args.language,
                    )
                    stt_latency = time.perf_counter() - started
                except STTServiceError as exc:
                    stats.stt_failures += 1
                    print(f"Conversation STT failed safely: {exc}")
                    print("State: STANDBY — say 'Hey Bunnelby'")
                    continue

                transcript = result.text.strip()
                if not transcript:
                    stats.empty_turns += 1
                    print("Conversation STT returned no text. Returning to STANDBY.")
                    print("State: STANDBY — say 'Hey Bunnelby'")
                    continue

                stats.conversation_turns += 1
                print()
                print("BUNNELBY CONVERSATION TRANSCRIPT")
                print(f"Text: {transcript}")
                print(
                    f"Language: {result.language} "
                    f"(confidence={result.language_probability:.3f})"
                )
                print(f"Captured: {captured_seconds:.2f}s | STT latency: {stt_latency:.2f}s")
                print("Raw audio saved: NO")

                if args.dispatch:
                    print("State: THINKING — dispatching transcript to existing /chat pipeline")
                    try:
                        dispatch_started = time.perf_counter()
                        response = dispatch_to_chat(args.api_url, transcript)
                        dispatch_latency = time.perf_counter() - dispatch_started
                        _print_backend_latency(response, dispatch_latency)
                        print(f"Post-speech processing so far: {stt_latency + dispatch_latency:.2f}s")
                        reply = str(response.get("reply") or "").strip()
                        spoken = str(response.get("spoken_reply") or response.get("spoken_ack") or "").strip()
                        print(f"Assistant reply: {reply or '[empty]'}")
                        if spoken:
                            print(f"Spoken reply: {spoken}")
                    except RuntimeError as exc:
                        print(f"Dispatch failed without losing transcript: {exc}")

                print()
                print("State: STANDBY — say 'Hey Bunnelby'")

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping Bunnelby voice runtime.")

    print()
    print("=" * 78)
    print("BUNNELBY VOICE RUNTIME RESULT")
    print("=" * 78)
    print(f"Wake candidates: {stats.wake_candidates}")
    print(f"Wake events: {stats.wake_events}")
    print(f"Completed conversation turns: {stats.conversation_turns}")
    print(f"Empty/timed-out turns: {stats.empty_turns}")
    print(f"STT failures: {stats.stt_failures}")
    print("Raw audio saved: NO")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bunnelby wake phrase to long natural-command listening runtime."
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="Completed conversation turns before exit; 0 runs continuously.",
    )
    parser.add_argument(
        "--command-wait",
        type=float,
        default=DEFAULT_COMMAND_WAIT_SECONDS,
        help="Seconds to wait for speech after wake detection.",
    )
    parser.add_argument(
        "--max-utterance",
        type=float,
        default=DEFAULT_MAX_UTTERANCE_SECONDS,
        help="Maximum continuous duration for one user command.",
    )
    parser.add_argument(
        "--conversation-silence",
        type=float,
        default=DEFAULT_CONVERSATION_SILENCE_SECONDS,
        help="Trailing silence that ends a post-wake command.",
    )
    parser.add_argument("--language", choices=("auto", "en", "hi"), default="auto")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--debug-wake-transcripts", action="store_true")
    args = parser.parse_args()
    args.turns = max(0, min(int(args.turns), 1000))
    args.command_wait = max(3.0, min(float(args.command_wait), 30.0))
    args.max_utterance = max(5.0, min(float(args.max_utterance), 120.0))
    args.conversation_silence = max(0.5, min(float(args.conversation_silence), 2.0))
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
