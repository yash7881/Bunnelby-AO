from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path
from urllib.parse import urlsplit

# When this file is launched directly (python services/api/scripts/...), Python puts only
# the script directory on sys.path. Add the repository root so `services.api...` imports
# work consistently from the documented repo-root command.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from services.api.app.stt_service import transcribe_audio
from services.api.app.vad_service import create_voice_activity_detector, vad_model_path, vad_settings

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
SILERO_VAD_HOST = "github.com"
SILERO_VAD_PATH = "/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"


def _validated_silero_url() -> str:
    parsed = urlsplit(SILERO_VAD_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != SILERO_VAD_HOST
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != SILERO_VAD_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Silero VAD source failed the fixed HTTPS allowlist.")
    return SILERO_VAD_URL


def ensure_vad_model() -> Path:
    target = vad_model_path()
    if target.is_file() and target.stat().st_size > 100_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading Silero VAD model once to: {target}")
    source_url = _validated_silero_url()
    try:
        # The source is a compile-time constant validated for exact HTTPS host/path above.
        with urllib.request.urlopen(  # nosec B310
            source_url,
            timeout=60,
        ) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if partial.stat().st_size < 100_000:
            raise RuntimeError("Downloaded VAD model is unexpectedly small.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def float_samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _resolve_input_device(sd, requested: str | None):
    devices = sd.query_devices()
    if requested:
        if requested.isdigit():
            return int(requested)
        needle = requested.casefold()
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0 and needle in str(device.get("name", "")).casefold():
                return index
        raise RuntimeError(f"No input device matched: {requested}")

    default_input = sd.default.device[0]
    if isinstance(default_input, int) and default_input >= 0:
        return default_input

    preferred = ("microphone array", "microphone")
    for phrase in preferred:
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0 and phrase in str(device.get("name", "")).casefold():
                return index
    raise RuntimeError("No microphone input device found.")


def capture_one_utterance(*, device_hint: str | None, wait_seconds: float) -> np.ndarray:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is unavailable.") from exc

    ensure_vad_model()
    detector = create_voice_activity_detector()
    settings = vad_settings()
    device_index = _resolve_input_device(sd, device_hint)
    device = sd.query_devices(device_index)
    print(f"Microphone: {device['name']}")
    print("Listening — speak one command, then stop speaking...")

    deadline = time.monotonic() + wait_seconds
    buffer = np.empty(0, dtype=np.float32)

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=settings.sample_rate,
            blocksize=settings.window_size,
        ) as stream:
            while time.monotonic() < deadline:
                samples, overflowed = stream.read(settings.window_size)
                if overflowed:
                    print("Warning: microphone input overflowed once; continuing.")
                buffer = np.concatenate((buffer, samples.reshape(-1)))
                while len(buffer) >= settings.window_size:
                    detector.accept_waveform(buffer[: settings.window_size])
                    buffer = buffer[settings.window_size :]
                if not detector.empty():
                    segment = np.asarray(detector.front.samples, dtype=np.float32).copy()
                    detector.pop()
                    return segment
    except Exception as exc:
        raise RuntimeError(f"Microphone/VAD capture failed: {exc}") from exc

    raise TimeoutError(f"No completed speech segment detected within {wait_seconds:.0f} seconds.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bunnelby live VAD → local Whisper STT probe")
    parser.add_argument(
        "--device",
        default=os.getenv("BUNNELBY_MIC_DEVICE", ""),
        help="Optional microphone device index or case-insensitive name fragment.",
    )
    parser.add_argument("--language", choices=("auto", "en", "hi"), default="auto")
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        samples = capture_one_utterance(
            device_hint=args.device.strip() or None,
            wait_seconds=max(5.0, min(args.wait_seconds, 60.0)),
        )
        capture_seconds = len(samples) / vad_settings().sample_rate
        print(f"Speech segment: {capture_seconds:.2f}s")

        wav_bytes = float_samples_to_wav(samples, vad_settings().sample_rate)
        stt_started = time.perf_counter()
        result = transcribe_audio(
            wav_bytes,
            content_type="audio/wav",
            language=args.language,
        )
        stt_seconds = time.perf_counter() - stt_started
        print(f"Transcript: {result.text}")
        print(
            "STT: "
            f"language={result.language} confidence={result.language_probability:.3f} "
            f"latency={stt_seconds:.2f}s total={time.perf_counter() - started:.2f}s"
        )
        return 0 if result.text else 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
