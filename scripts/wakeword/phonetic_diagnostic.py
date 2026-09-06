from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
ATTEMPTS = 10
RECORD_SECONDS = 3.5
MODEL_NAME = "base.en"


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


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split()).strip(" .,!?:;\"'")


def transcribe(model: WhisperModel, audio: np.ndarray) -> tuple[str, str | None, float | None]:
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    return text, language, language_probability


def main() -> int:
    print("=" * 64)
    print("BUNNELBY WAKE-PHRASE PHONETIC DIAGNOSTIC")
    print("=" * 64)
    print("Purpose: learn what a local ASR model hears when you say 'Hey Bunnelby'.")
    print("Audio is processed in memory and is NOT saved to disk.")
    print()

    device_index, device = get_default_input_device()
    print(f"Microphone: [{device_index}] {device['name']}")
    print("Microphone 16 kHz mono check: PASS")
    print()

    print(f"Loading local Whisper model: {MODEL_NAME} (CPU/int8)")
    print("First run may download the model once.")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    print("Whisper model load: PASS")

    results: list[dict[str, object]] = []

    for attempt in range(1, ATTEMPTS + 1):
        print()
        print(f"Attempt {attempt}/{ATTEMPTS}")
        print("Say exactly: Hey Bunnelby")
        time.sleep(0.5)
        print("RECORDING...")

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0

        text, language, language_probability = transcribe(model, samples)
        normalized = normalize_text(text)

        print(f"ASR heard: {text!r}")
        print(f"RMS={rms:.6f}  Peak={peak:.6f}")

        results.append(
            {
                "attempt": attempt,
                "transcript": text,
                "normalized": normalized,
                "rms": rms,
                "peak": peak,
                "language": language,
                "language_probability": language_probability,
            }
        )

    counts = Counter(
        str(row["normalized"])
        for row in results
        if row["normalized"]
    )

    print()
    print("=" * 64)
    print("BUNNELBY PHONETIC DIAGNOSTIC RESULT")
    print("=" * 64)
    print("Transcripts:")
    for row in results:
        print(f"  {row['attempt']:02d}. {row['transcript']!r}")

    print()
    print("Most common normalized interpretations:")
    if counts:
        for phrase, count in counts.most_common():
            print(f"  {count}x  {phrase}")
    else:
        print("  No speech transcription produced.")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        report_dir = Path(local_app_data) / "Bunnelby" / "wakeword" / "logs"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        report_path = report_dir / f"phonetic_diagnostic_{stamp}.json"
        report = {
            "target_phrase": "Hey Bunnelby",
            "model": MODEL_NAME,
            "audio_saved": False,
            "attempts": results,
            "counts": dict(counts),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"Transcript-only report: {report_path}")
        print("Audio saved: NO")

    print()
    print("Next decision: build/test Sherpa keyword aliases from the observed pronunciation, or drop Sherpa if the acoustic path remains poor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
