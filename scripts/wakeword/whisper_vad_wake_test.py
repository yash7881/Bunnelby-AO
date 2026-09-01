from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
ATTEMPTS = 10
RECORD_SECONDS = 3.5
MODEL_NAME = "base.en"

# Safe acoustic variants observed from the user's real recordings.
# We intentionally exclude common phrases such as "hey but there'll be"
# because they would create unacceptable false-positive risk.
SAFE_WAKE_FORMS = {
    "hey bunnelby",
    "hey bonilby",
    "hey bonnellby",
    "hey bundle b",
}


@dataclass
class AttemptResult:
    attempt: int
    transcript: str
    normalized: str
    vad_speech_seconds: float
    matched: bool
    latency_seconds: float
    rms: float
    peak: float


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
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def wake_match(normalized: str) -> bool:
    # Exact membership is deliberate. We do not use broad fuzzy matching here:
    # wake-word false activations matter more than accepting every ASR spelling.
    return normalized in SAFE_WAKE_FORMS


def transcribe_wake(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 180,
        },
        hotwords="Hey Bunnelby Bonilby Bonnellby",
    )

    segment_list = list(segments)
    text = " ".join(segment.text.strip() for segment in segment_list).strip()
    speech_seconds = sum(max(0.0, segment.end - segment.start) for segment in segment_list)
    latency = time.perf_counter() - started
    return text, speech_seconds, latency


def main() -> int:
    print("=" * 68)
    print("BUNNELBY WHISPER + SILERO VAD WAKE TEST")
    print("=" * 68)
    print("Engine: faster-whisper base.en on CPU/int8")
    print("VAD: built-in Silero VAD")
    print("Matcher: strict safe acoustic aliases only")
    print("Audio saved: NO")
    print()

    device_index, device = get_default_input_device()
    print(f"Microphone: [{device_index}] {device['name']}")
    print("Microphone 16 kHz mono check: PASS")
    print()

    print(f"Loading local Whisper model: {MODEL_NAME} (CPU/int8)")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    print("Model load: PASS")

    results: list[AttemptResult] = []

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

        transcript, speech_seconds, latency = transcribe_wake(model, samples)
        normalized = normalize_text(transcript)
        matched = wake_match(normalized)

        results.append(
            AttemptResult(
                attempt=attempt,
                transcript=transcript,
                normalized=normalized,
                vad_speech_seconds=speech_seconds,
                matched=matched,
                latency_seconds=latency,
                rms=rms,
                peak=peak,
            )
        )

        state = "DETECTED" if matched else "MISS"
        print(f"ASR: {transcript!r}")
        print(f"Normalized: {normalized!r}")
        print(f"Wake: {state}")
        print(f"ASR latency: {latency:.2f}s")

    detections = sum(1 for row in results if row.matched)
    average_latency = float(np.mean([row.latency_seconds for row in results]))
    mean_rms = float(np.mean([row.rms for row in results]))
    peak_level = float(np.max([row.peak for row in results]))

    print()
    print("=" * 68)
    print("BUNNELBY WHISPER-VAD WAKE RESULT")
    print("=" * 68)
    print(f"Detections: {detections}/{ATTEMPTS}")
    print(f"Recall: {detections / ATTEMPTS * 100:.1f}%")
    print(f"Average ASR latency: {average_latency:.2f}s")
    print(f"Mean recording RMS: {mean_rms:.6f}")
    print(f"Peak recording level: {peak_level:.6f}")
    print("Audio saved: NO")
    print()
    print("Transcripts:")
    for row in results:
        mark = "PASS" if row.matched else "MISS"
        print(f"  {row.attempt:02d}. [{mark}] {row.transcript!r}")

    print()
    if detections >= 8:
        print("Decision: VIABLE - build the always-on VAD-gated listener next.")
    elif detections >= 5:
        print("Decision: PROMISING - one controlled matcher/ASR refinement only.")
    else:
        print("Decision: NOT GOOD ENOUGH - do not ship this wake path as-is.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
