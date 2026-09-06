from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
SAMPLE_RATE = 16_000
RECORD_SECONDS = 3.5
ATTEMPTS = 10
CHUNK_SAMPLES = 1_600  # 100 ms
TAIL_SILENCE_SECONDS = 0.8

# Permissive diagnostic settings only. These are NOT production thresholds.
KEYWORD_SCORE = 3.0
KEYWORD_THRESHOLD = 0.10
NUM_TRAILING_BLANKS = 1

# These aliases come from the user's actual local Whisper diagnostic.
# We intentionally exclude common-speech interpretations such as
# "HEY BUT THERE'LL BE" because they would be high false-positive risks.
ALIASES = [
    "HEY BUNNELBY",      # intended spelling / baseline
    "HEY BONILBY",       # observed by Whisper
    "HEY BONNELLBY",     # observed by Whisper
    "HEY BUNDLE B",      # observed by Whisper
    "HEY BONO V",        # observed acoustic approximation
    "HEY ONE L B",       # observed acoustic approximation
]


def runtime_paths() -> dict[str, Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is unavailable. This diagnostic targets Windows.")

    root = Path(local_app_data) / "Bunnelby" / "wakeword"
    model_dir = root / "models" / MODEL_NAME
    return {
        "root": root,
        "model_dir": model_dir,
        "encoder": model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "decoder": model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "joiner": model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "tokens": model_dir / "tokens.txt",
        "bpe": model_dir / "bpe.model",
        "alias_dir": root / "keywords" / "alias_diagnostic",
        "logs": root / "logs",
    }


def validate_assets(paths: dict[str, Path]) -> None:
    required = ["encoder", "decoder", "joiner", "tokens", "bpe"]
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing wake-word assets:\n  " + "\n  ".join(missing))


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


def build_keyword_files(paths: dict[str, Path]) -> list[dict[str, object]]:
    alias_dir = paths["alias_dir"]
    alias_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, alias in enumerate(ALIASES, start=1):
        encoded = sherpa_onnx.text2token(
            [alias],
            tokens=str(paths["tokens"]),
            tokens_type="bpe",
            bpe_model=str(paths["bpe"]),
            lexicon="",
        )
        if len(encoded) != 1 or not encoded[0]:
            raise RuntimeError(f"Could not tokenize alias: {alias}")

        token_line = " ".join(encoded[0])
        keyword_file = alias_dir / f"alias_{index:02d}.txt"
        keyword_file.write_text(token_line + "\n", encoding="utf-8")
        results.append(
            {
                "alias": alias,
                "tokens": token_line,
                "keyword_file": keyword_file,
            }
        )
    return results


def create_spotter(paths: dict[str, Path], keyword_file: Path) -> sherpa_onnx.KeywordSpotter:
    return sherpa_onnx.KeywordSpotter(
        tokens=str(paths["tokens"]),
        encoder=str(paths["encoder"]),
        decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]),
        num_threads=2,
        max_active_paths=4,
        keywords_file=str(keyword_file),
        keywords_score=KEYWORD_SCORE,
        keywords_threshold=KEYWORD_THRESHOLD,
        num_trailing_blanks=NUM_TRAILING_BLANKS,
        provider="cpu",
    )


def detect_audio(spotter: sherpa_onnx.KeywordSpotter, audio: np.ndarray) -> bool:
    stream = spotter.create_stream()
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)

    for start in range(0, len(samples), CHUNK_SAMPLES):
        chunk = samples[start : start + CHUNK_SAMPLES]
        stream.accept_waveform(SAMPLE_RATE, chunk)
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            if spotter.get_result(stream):
                return True

    tail = np.zeros(int(SAMPLE_RATE * TAIL_SILENCE_SECONDS), dtype=np.float32)
    for start in range(0, len(tail), CHUNK_SAMPLES):
        chunk = tail[start : start + CHUNK_SAMPLES]
        stream.accept_waveform(SAMPLE_RATE, chunk)
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            if spotter.get_result(stream):
                return True

    return False


def main() -> int:
    print("=" * 68)
    print("BUNNELBY SHERPA PHONETIC-ALIAS DIAGNOSTIC")
    print("=" * 68)
    print("One 10-attempt recording session; the same RAM-only audio is replayed")
    print("against several safe phonetic aliases. Audio is NOT saved to disk.")
    print()

    paths = runtime_paths()
    validate_assets(paths)
    aliases = build_keyword_files(paths)

    device_index, device = get_default_input_device()
    print(f"Microphone: [{device_index}] {device['name']}")
    print("Microphone 16 kHz mono check: PASS")
    print(f"Sherpa: {sherpa_onnx.__version__} | provider=cpu")
    print(
        "Diagnostic config: "
        f"score={KEYWORD_SCORE}, threshold={KEYWORD_THRESHOLD}, "
        f"trailing_blanks={NUM_TRAILING_BLANKS}"
    )
    print()
    print("Aliases under test:")
    for row in aliases:
        print(f"  - {row['alias']}: {row['tokens']}")

    recordings: list[np.ndarray] = []
    rms_values: list[float] = []
    peak_values: list[float] = []

    print()
    print("RECORDING PHASE")
    print("Say exactly 'Hey Bunnelby' once on each attempt, in your normal voice.")

    for attempt in range(1, ATTEMPTS + 1):
        print()
        print(f"Attempt {attempt}/{ATTEMPTS} - get ready...")
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
        samples = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        recordings.append(samples)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms_values.append(rms)
        peak_values.append(peak)
        print(f"Captured in RAM. RMS={rms:.6f} Peak={peak:.6f}")

    print()
    print("EVALUATION PHASE")
    print("No more speaking needed.")

    alias_results: list[dict[str, object]] = []
    for row in aliases:
        alias = str(row["alias"])
        keyword_file = Path(row["keyword_file"])
        spotter = create_spotter(paths, keyword_file)
        attempt_hits: list[bool] = []

        for audio in recordings:
            attempt_hits.append(detect_audio(spotter, audio))

        hits = sum(attempt_hits)
        result = {
            "alias": alias,
            "tokens": str(row["tokens"]),
            "hits": hits,
            "attempts": ATTEMPTS,
            "recall": hits / ATTEMPTS,
            "attempt_hits": attempt_hits,
        }
        alias_results.append(result)
        hit_map = "".join("Y" if hit else "." for hit in attempt_hits)
        print(f"{alias:<18} {hits:>2}/{ATTEMPTS}  [{hit_map}]")

    ranked = sorted(alias_results, key=lambda r: (-int(r["hits"]), str(r["alias"])))
    best = ranked[0]

    print()
    print("=" * 68)
    print("BUNNELBY SHERPA ALIAS DIAGNOSTIC RESULT")
    print("=" * 68)
    print(f"Best alias: {best['alias']}")
    print(f"Best detections: {best['hits']}/{ATTEMPTS}")
    print(f"Mean recording RMS: {float(np.mean(rms_values)):.6f}")
    print(f"Peak recording level: {float(np.max(peak_values)):.6f}")

    best_hits = int(best["hits"])
    if best_hits >= 7:
        decision = "SHERPA_ALIAS_VIABLE"
        print("Decision: SHERPA ALIAS PATH IS VIABLE - proceed to controlled negative testing.")
    elif best_hits >= 3:
        decision = "SHERPA_ALIAS_PARTIAL"
        print("Decision: PARTIAL SIGNAL - one narrow refinement step only; no brute-force grid.")
    else:
        decision = "DROP_SHERPA"
        print("Decision: DROP SHERPA FOR BUNNELBY - acoustic match remains too weak.")

    paths["logs"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_path = paths["logs"] / f"sherpa_alias_diagnostic_{stamp}.json"
    report = {
        "target_phrase": "Hey Bunnelby",
        "audio_saved": False,
        "sherpa_version": sherpa_onnx.__version__,
        "config": {
            "keyword_score": KEYWORD_SCORE,
            "keyword_threshold": KEYWORD_THRESHOLD,
            "num_trailing_blanks": NUM_TRAILING_BLANKS,
        },
        "aliases": alias_results,
        "best": best,
        "decision": decision,
        "recording_metrics": {
            "mean_rms": float(np.mean(rms_values)),
            "max_peak": float(np.max(peak_values)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Transcript-free diagnostic report: {report_path}")
    print("Audio saved: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
