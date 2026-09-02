from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
READ_SECONDS = 0.10
WAKE_ASR_MODEL = "base.en"
WAKE_HOTWORDS = "Hey Bunnelby Bonilby Bonnellby"


def _sounddevice():
    """Import PortAudio only at the real microphone boundary, not during CI imports."""
    try:
        import sounddevice
    except Exception as exc:
        raise RuntimeError("sounddevice/PortAudio is unavailable for live microphone use.") from exc
    return sounddevice

# Safe acoustic variants observed in the user's real recordings.
# Deliberately excludes bare "Bunnelby" and common phrases such as
# "hey but there'll be" because recall must not be bought with unsafe FP risk.
SAFE_WAKE_FORMS = {
    "hey bunnelby",
    "hey bonilby",
    "hey bonnellby",
    "hey bundle b",
}

# Use an immutable upstream Silero revision as the primary bootstrap source.
# A sherpa-onnx release asset is retained as a fallback. Every downloaded
# candidate is actually loaded by sherpa-onnx before it is promoted into the
# Bunnelby runtime model path, so an HTML/error response can never be accepted
# just because it has a plausible filename.
SILERO_VAD_SOURCES = (
    (
        "Silero upstream pinned model",
        "https://raw.githubusercontent.com/snakers4/silero-vad/"
        "867c2aa692646a1f1de3e94a15c9dd9f614c0acb/"
        "src/silero_vad/data/silero_vad.onnx",
    ),
    (
        "sherpa-onnx V5 release model",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "vad-models/silero_vad_v5.onnx",
    ),
)
MIN_SILERO_MODEL_BYTES = 1_000_000

VAD_THRESHOLD = 0.35
VAD_MIN_SILENCE_SECONDS = 0.35
VAD_MIN_SPEECH_SECONDS = 0.10
VAD_MAX_SPEECH_SECONDS = 4.0
VAD_BUFFER_SECONDS = 30.0

MIN_WAKE_CANDIDATE_SECONDS = 0.30
MAX_WAKE_CANDIDATE_SECONDS = 3.25
DEFAULT_COOLDOWN_SECONDS = 3.0


@dataclass
class RuntimeStats:
    speech_segments: int = 0
    asr_calls: int = 0
    skipped_duration: int = 0
    skipped_cooldown: int = 0
    wake_events: int = 0
    asr_latency_total: float = 0.0


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def wake_match(text: str) -> bool:
    return normalize_text(text) in SAFE_WAKE_FORMS


def _runtime_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "Bunnelby"
    return Path.home() / ".bunnelby"


def vad_model_path() -> Path:
    return _runtime_root() / "models" / "vad" / "silero_vad.onnx"


def wake_asr_model_root() -> Path:
    return _runtime_root() / "models" / "wake-asr"


def create_vad(model_path: Path):
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_path)
    config.silero_vad.threshold = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_SECONDS
    config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_SECONDS
    config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_SECONDS
    config.silero_vad.window_size = 512
    config.sample_rate = SAMPLE_RATE

    detector = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=VAD_BUFFER_SECONDS,
    )
    return detector, int(config.silero_vad.window_size)


def _validate_vad_model(candidate: Path) -> None:
    if not candidate.is_file():
        raise RuntimeError("Silero VAD candidate file was not created.")
    size = candidate.stat().st_size
    if size < MIN_SILERO_MODEL_BYTES:
        raise RuntimeError(
            f"Silero VAD candidate is unexpectedly small ({size:,} bytes)."
        )

    # File-size checks alone can accept HTML/error payloads. Loading the model
    # through the exact runtime API is the compatibility gate.
    detector, window_size = create_vad(candidate)
    if detector is None or window_size <= 0:
        raise RuntimeError("Silero VAD candidate failed runtime validation.")


def ensure_silero_vad_model() -> Path:
    target = vad_model_path()

    if target.is_file():
        try:
            _validate_vad_model(target)
            return target
        except Exception as exc:
            print(f"Existing Silero VAD runtime asset is invalid: {exc}")
            print("A verified replacement will be downloaded atomically.")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    errors: list[str] = []

    for source_name, url in SILERO_VAD_SOURCES:
        try:
            if partial.exists():
                partial.unlink()

            print(f"Downloading Silero VAD from: {source_name}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Bunnelby-Wake-Runtime/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)

            _validate_vad_model(partial)
            partial.replace(target)
            print(
                "Silero VAD bootstrap: PASS "
                f"({target.stat().st_size:,} bytes, runtime validated)"
            )
            return target
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            print(f"Silero VAD source failed validation: {source_name}: {exc}")
        finally:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass

    detail = "\n  - ".join(errors)
    raise RuntimeError(
        "Could not bootstrap a valid Silero VAD model from verified sources."
        + (f"\n  - {detail}" if detail else "")
    )


def load_wake_asr() -> WhisperModel:
    root = wake_asr_model_root()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Loading wake ASR: {WAKE_ASR_MODEL} (CPU/int8)")
    print("First run may download the model once.")
    return WhisperModel(
        WAKE_ASR_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
        download_root=str(root),
    )


def transcribe_candidate(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    started = time.perf_counter()

    # External Silero VAD already produced this speech segment, so we intentionally
    # disable faster-whisper's second VAD pass. This is the one controlled refinement:
    # avoid trimming the leading "Hey", which caused two baseline misses.
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        hotwords=WAKE_HOTWORDS,
        word_timestamps=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    latency = time.perf_counter() - started
    return text, latency


def default_microphone(device_index: int | None):
    sd = _sounddevice()
    devices = sd.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No audio devices detected by PortAudio.")

    index = int(sd.default.device[0]) if device_index is None else int(device_index)
    if index < 0 or index >= len(devices):
        raise RuntimeError(f"Invalid input device index: {index}")

    device = devices[index]
    sd.check_input_settings(
        device=index,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    )
    return index, device


def emit_wake_event(*, transcript: str, latency: float, candidate_seconds: float) -> None:
    payload = {
        "event": "wake_detected",
        "wake_phrase": "Hey Bunnelby",
        "transcript": transcript,
        "latency_seconds": round(latency, 4),
        "candidate_seconds": round(candidate_seconds, 4),
        "timestamp_monotonic": round(time.monotonic(), 4),
    }
    print("BUNNELBY_WAKE_EVENT " + json.dumps(payload, ensure_ascii=True))


def run_self_test() -> int:
    expected_true = [
        "Hey Bunnelby",
        "HEY BONILBY!",
        "Hey, Bonnellby.",
        "hey bundle b",
    ]
    expected_false = [
        "Bunnelby",
        "hey but there'll be",
        "hey buddy",
        "bundle b",
        "hello bunnelby",
        "",
    ]

    failures: list[str] = []
    for phrase in expected_true:
        if not wake_match(phrase):
            failures.append(f"expected PASS: {phrase!r}")
    for phrase in expected_false:
        if wake_match(phrase):
            failures.append(f"expected REJECT: {phrase!r}")

    if failures:
        print("SELF TEST FAILED")
        for failure in failures:
            print(" -", failure)
        return 1

    print("BUNNELBY WAKE MATCHER SELF TEST: PASS")
    print("Bare 'Bunnelby': REJECTED")
    print("Common confusable phrase: REJECTED")
    return 0


def run_listener(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    vad, window_size = create_vad(model_path)
    model = load_wake_asr()

    device_index, device = default_microphone(args.device)
    print()
    print("=" * 72)
    print("BUNNELBY ALWAYS-ON WAKE LISTENER")
    print("=" * 72)
    print(f"Microphone: [{device_index}] {device['name']}")
    print(f"VAD: Silero ONNX | threshold={VAD_THRESHOLD}")
    print(f"ASR: {WAKE_ASR_MODEL} CPU/int8")
    print("Matcher: strict safe forms only")
    print(f"Cooldown: {args.cooldown:.1f}s")
    print("Audio saved: NO")
    print("Pre-wake audio leaves device: NO")
    if args.duration > 0:
        print(f"Run duration: {args.duration:.0f}s")
    else:
        print("Run duration: until Ctrl+C")
    print("Listening...")

    stats = RuntimeStats()
    read_size = int(READ_SECONDS * SAMPLE_RATE)
    pending = np.empty(0, dtype=np.float32)
    started = time.monotonic()
    last_wake_time = -1e9

    try:
        sd = _sounddevice()
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        ) as microphone:
            while args.duration <= 0 or time.monotonic() - started < args.duration:
                samples, overflowed = microphone.read(read_size)
                if overflowed:
                    print("WARNING: microphone input overflow")

                chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
                pending = np.concatenate((pending, chunk))

                while pending.size >= window_size:
                    frame = pending[:window_size]
                    pending = pending[window_size:]
                    vad.accept_waveform(frame)

                while not vad.empty():
                    segment = np.asarray(vad.front.samples, dtype=np.float32).reshape(-1)
                    vad.pop()
                    stats.speech_segments += 1

                    candidate_seconds = float(segment.size / SAMPLE_RATE)
                    if not (
                        MIN_WAKE_CANDIDATE_SECONDS
                        <= candidate_seconds
                        <= MAX_WAKE_CANDIDATE_SECONDS
                    ):
                        stats.skipped_duration += 1
                        if args.debug_transcripts:
                            print(
                                f"[skip duration] {candidate_seconds:.2f}s "
                                f"outside {MIN_WAKE_CANDIDATE_SECONDS:.2f}-"
                                f"{MAX_WAKE_CANDIDATE_SECONDS:.2f}s"
                            )
                        continue

                    now = time.monotonic()
                    if now - last_wake_time < args.cooldown:
                        stats.skipped_cooldown += 1
                        continue

                    transcript, latency = transcribe_candidate(model, segment)
                    stats.asr_calls += 1
                    stats.asr_latency_total += latency
                    matched = wake_match(transcript)

                    if args.debug_transcripts:
                        mark = "WAKE" if matched else "speech"
                        print(
                            f"[{mark}] {transcript!r} | "
                            f"segment={candidate_seconds:.2f}s | asr={latency:.2f}s"
                        )

                    if matched:
                        last_wake_time = time.monotonic()
                        stats.wake_events += 1
                        print()
                        print(
                            f"WAKE DETECTED #{stats.wake_events}: "
                            f"{transcript!r} ({latency:.2f}s ASR)"
                        )
                        emit_wake_event(
                            transcript=transcript,
                            latency=latency,
                            candidate_seconds=candidate_seconds,
                        )
                        print()
                        if args.exit_on_wake:
                            return 0

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping listener.")

    elapsed = time.monotonic() - started
    average_latency = (
        stats.asr_latency_total / stats.asr_calls if stats.asr_calls else 0.0
    )

    print()
    print("=" * 72)
    print("BUNNELBY ALWAYS-ON WAKE RESULT")
    print("=" * 72)
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Speech segments: {stats.speech_segments}")
    print(f"ASR calls: {stats.asr_calls}")
    print(f"Skipped by duration gate: {stats.skipped_duration}")
    print(f"Skipped by cooldown: {stats.skipped_cooldown}")
    print(f"Wake detections: {stats.wake_events}")
    if args.expected_wakes is not None:
        expected = max(0, int(args.expected_wakes))
        recall = (stats.wake_events / expected * 100.0) if expected else 0.0
        print(f"Expected spoken wakes: {expected}")
        print(f"Observed recall: {recall:.1f}%")
    print(f"Average candidate ASR latency: {average_latency:.2f}s")
    print("Audio saved: NO")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local always-on Bunnelby wake listener (Silero VAD -> faster-whisper)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Seconds to listen; use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--expected-wakes",
        type=int,
        default=None,
        help="Optional number of wake phrases you intend to speak for recall reporting.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Minimum seconds between accepted wake events.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input-device index; default uses Windows default mic.",
    )
    parser.add_argument(
        "--debug-transcripts",
        action="store_true",
        help="Print non-wake candidate transcripts. Off by default for privacy.",
    )
    parser.add_argument(
        "--exit-on-wake",
        action="store_true",
        help="Exit immediately after the first accepted wake event.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate strict matcher safety without loading mic/models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.duration = max(0.0, float(args.duration))
    args.cooldown = max(0.5, min(float(args.cooldown), 30.0))

    if args.self_test:
        return run_self_test()
    return run_listener(args)


if __name__ == "__main__":
    raise SystemExit(main())
