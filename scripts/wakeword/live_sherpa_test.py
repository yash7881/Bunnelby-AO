from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.1
TEST_SECONDS = 60
EXPECTED_ATTEMPTS = 10

# Permissive diagnostic settings. These are not production thresholds.
KEYWORD_SCORE = 3.0
KEYWORD_THRESHOLD = 0.10
NUM_TRAILING_BLANKS = 1


def wake_paths() -> dict[str, Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is unavailable. This diagnostic targets Windows.")

    wake_root = Path(local_app_data) / "Bunnelby" / "wakeword"
    model_dir = wake_root / "models" / MODEL_NAME

    return {
        "encoder": model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "decoder": model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "joiner": model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "tokens": model_dir / "tokens.txt",
        "keywords": wake_root / "keywords" / "hey_bunnelby.txt",
    }


def validate_assets(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Wake-word assets are missing:\n  " + "\n  ".join(missing))


def get_default_input_device():
    devices = sd.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No audio devices detected by PortAudio.")

    default_input = sd.default.device[0]
    if default_input is None or int(default_input) < 0:
        raise RuntimeError("Windows has no default microphone configured.")

    device_index = int(default_input)
    device = devices[device_index]

    sd.check_input_settings(
        device=device_index,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    )
    return device_index, device


def create_spotter(paths: dict[str, Path]) -> sherpa_onnx.KeywordSpotter:
    return sherpa_onnx.KeywordSpotter(
        tokens=str(paths["tokens"]),
        encoder=str(paths["encoder"]),
        decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]),
        num_threads=2,
        max_active_paths=4,
        keywords_file=str(paths["keywords"]),
        keywords_score=KEYWORD_SCORE,
        keywords_threshold=KEYWORD_THRESHOLD,
        num_trailing_blanks=NUM_TRAILING_BLANKS,
        provider="cpu",
    )


def main() -> int:
    print("=" * 62)
    print("BUNNELBY LIVE WAKE-WORD DIAGNOSTIC")
    print("=" * 62)

    paths = wake_paths()
    validate_assets(paths)

    print(f"Sherpa: {sherpa_onnx.__version__}")
    print("Provider: CPU")
    print(f"Keyword tokens: {paths['keywords'].read_text(encoding='utf-8').strip()}")

    device_index, device = get_default_input_device()
    print(f"Microphone: [{device_index}] {device['name']}")
    print("Microphone 16 kHz mono check: PASS")
    print(
        "Diagnostic config: "
        f"score={KEYWORD_SCORE}, threshold={KEYWORD_THRESHOLD}, "
        f"trailing_blanks={NUM_TRAILING_BLANKS}"
    )

    spotter = create_spotter(paths)
    stream = spotter.create_stream()

    samples_per_read = int(CHUNK_SECONDS * SAMPLE_RATE)
    detections = 0
    blocks = 0
    rms_values: list[float] = []
    started = time.monotonic()

    print()
    print(f"Speak 'Hey Bunnelby' {EXPECTED_ATTEMPTS} times in the next {TEST_SECONDS} seconds.")
    print("Use your normal voice and keep roughly 2 seconds between attempts.")
    print("Listening...")

    with sd.InputStream(
        device=device_index,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    ) as microphone:
        while time.monotonic() - started < TEST_SECONDS:
            samples, overflowed = microphone.read(samples_per_read)
            if overflowed:
                print("WARNING: microphone input overflow")

            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            blocks += 1
            rms_values.append(float(np.sqrt(np.mean(np.square(samples)))))

            stream.accept_waveform(SAMPLE_RATE, samples)

            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
                result = spotter.get_result(stream)
                if result:
                    detections += 1
                    elapsed = time.monotonic() - started
                    print(f"DETECTED #{detections} at {elapsed:.1f}s: {result}")
                    spotter.reset_stream(stream)

    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    peak_rms = float(np.max(rms_values)) if rms_values else 0.0

    print()
    print("=" * 62)
    print("BUNNELBY LIVE DIAGNOSTIC RESULT")
    print("=" * 62)
    print(f"Detections: {detections}")
    print(f"Expected spoken attempts: {EXPECTED_ATTEMPTS}")
    print(f"Mean RMS: {mean_rms:.6f}")
    print(f"Peak RMS: {peak_rms:.6f}")
    print(f"Blocks processed: {blocks}")

    if detections >= 7:
        print("RESULT: STRONG ACOUSTIC SIGNAL")
    elif detections >= 3:
        print("RESULT: PARTIAL SIGNAL - representation/tuning needs work")
    elif detections >= 1:
        print("RESULT: VERY WEAK SIGNAL - do not run a large benchmark yet")
    else:
        print("RESULT: ZERO DETECTIONS - move to pronunciation/phonetic diagnosis")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
