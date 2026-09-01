from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    import pvporcupine
except ImportError as exc:
    raise SystemExit(
        "pvporcupine is not installed. Run: python -m pip install pvporcupine"
    ) from exc

TEST_SECONDS = 60
EXPECTED_ATTEMPTS = 10
DEFAULT_SENSITIVITY = 0.60


def default_keyword_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is unavailable. This diagnostic targets Windows.")
    return (
        Path(local_app_data)
        / "Bunnelby"
        / "wakeword"
        / "porcupine"
        / "hey_bunnelby_windows.ppn"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Windows Porcupine diagnostic for the Hey Bunnelby wake phrase."
    )
    parser.add_argument(
        "--keyword-path",
        type=Path,
        default=default_keyword_path(),
        help="Path to the Windows Porcupine .ppn model.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=DEFAULT_SENSITIVITY,
        help="Porcupine sensitivity in [0, 1]. Higher means fewer misses but more false alarms.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=TEST_SECONDS,
        help="Live diagnostic duration.",
    )
    return parser.parse_args()


def get_access_key() -> str:
    key = os.environ.get("PICOVOICE_ACCESS_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PICOVOICE_ACCESS_KEY is not set. Keep the key private and set it only in your local shell."
        )
    return key


def choose_input_device(sample_rate: int) -> tuple[int, object]:
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
        dtype="int16",
        samplerate=sample_rate,
    )
    return device_index, device


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.sensitivity <= 1.0:
        raise ValueError("--sensitivity must be between 0 and 1.")
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive.")

    keyword_path = args.keyword_path.expanduser().resolve()
    if not keyword_path.is_file():
        raise FileNotFoundError(
            "Porcupine keyword model not found:\n"
            f"  {keyword_path}\n"
            "Download a Windows custom model for the phrase 'Hey Bunnelby' and place it at this path."
        )

    access_key = get_access_key()

    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[str(keyword_path)],
        sensitivities=[args.sensitivity],
    )

    try:
        device_index, device = choose_input_device(porcupine.sample_rate)

        print("=" * 66)
        print("BUNNELBY PORCUPINE LIVE WAKE-WORD DIAGNOSTIC")
        print("=" * 66)
        print(f"Engine version: {pvporcupine.__version__}")
        print(f"Sample rate: {porcupine.sample_rate} Hz")
        print(f"Frame length: {porcupine.frame_length}")
        print(f"Sensitivity: {args.sensitivity:.2f}")
        print(f"Keyword model: {keyword_path}")
        print(f"Microphone: [{device_index}] {device['name']}")
        print("Audio saved: NO")
        print()
        print(
            f"Say 'Hey Bunnelby' {EXPECTED_ATTEMPTS} times during the next {args.seconds} seconds."
        )
        print("Use your normal voice and keep roughly 2 seconds between attempts.")
        print("Listening...")

        detections = 0
        processed_frames = 0
        started = time.monotonic()
        last_detection_at = -999.0
        cooldown_seconds = 1.25

        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="int16",
            samplerate=porcupine.sample_rate,
            blocksize=porcupine.frame_length,
        ) as microphone:
            while time.monotonic() - started < args.seconds:
                samples, overflowed = microphone.read(porcupine.frame_length)
                if overflowed:
                    print("WARNING: microphone input overflow")

                pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
                if pcm.size != porcupine.frame_length:
                    continue

                keyword_index = porcupine.process(pcm.tolist())
                processed_frames += 1

                if keyword_index >= 0:
                    now = time.monotonic()
                    if now - last_detection_at < cooldown_seconds:
                        continue
                    last_detection_at = now
                    detections += 1
                    print(f"DETECTED #{detections} at {now - started:.1f}s")

        print()
        print("=" * 66)
        print("BUNNELBY PORCUPINE LIVE DIAGNOSTIC RESULT")
        print("=" * 66)
        print(f"Detections: {detections}")
        print(f"Expected spoken attempts: {EXPECTED_ATTEMPTS}")
        print(f"Frames processed: {processed_frames}")
        print(f"Sensitivity: {args.sensitivity:.2f}")
        print("Audio saved: NO")

        if detections >= 8:
            print("Decision: STRONG CANDIDATE - proceed to hard-negative and long false-positive testing.")
        elif detections >= 5:
            print("Decision: PROMISING - run a small sensitivity sweep before negative testing.")
        elif detections >= 1:
            print("Decision: WEAK - one controlled sensitivity check only; do not brute-force.")
        else:
            print("Decision: FAIL - do not invest further in this Porcupine model.")

        return 0
    finally:
        porcupine.delete()


if __name__ == "__main__":
    raise SystemExit(main())
