from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd

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

DEFAULT_POSITIVE_ATTEMPTS = 20
DEFAULT_RECORD_SECONDS = 3.2
DEFAULT_NEGATIVE_SECONDS = 300.0
DEFAULT_DEBOUNCE_SECONDS = 3.0
TARGET_MIN_RECALL = 0.80
TARGET_PREFERRED_RECALL = 0.90
TARGET_MAX_FP_PER_HOUR = 0.50
CONFIDENCE = 0.95


@dataclass
class CandidateResult:
    transcript: str
    matched: bool
    candidate_seconds: float
    asr_latency_seconds: float


@dataclass
class PositiveAttempt:
    attempt: int
    detected: bool
    rms: float
    peak: float
    speech_segments: int
    candidates: list[CandidateResult]


@dataclass
class NegativeStats:
    elapsed_seconds: float = 0.0
    speech_segments: int = 0
    asr_calls: int = 0
    skipped_duration: int = 0
    raw_match_events: int = 0
    debounced_wake_events: int = 0
    asr_latency_total: float = 0.0


def _runtime_root() -> Path:
    import os

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "Bunnelby"
    return Path.home() / ".bunnelby"


def _report_path(kind: str) -> Path:
    folder = _runtime_root() / "wakeword" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    return folder / f"wake_certification_{kind}_{stamp}.json"


def _save_report(kind: str, payload: dict[str, object]) -> Path:
    path = _report_path(kind)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _capture_audio(device_index: int, seconds: float) -> np.ndarray:
    frames = max(1, int(round(seconds * SAMPLE_RATE)))
    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_index,
        blocking=True,
    )
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _drain_vad(vad) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    while not vad.empty():
        segment = np.asarray(vad.front.samples, dtype=np.float32).reshape(-1).copy()
        vad.pop()
        if segment.size:
            segments.append(segment)
    return segments


def _speech_segments_from_audio(audio: np.ndarray, model_path: Path) -> list[np.ndarray]:
    vad, window_size = create_vad(model_path)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    offset = 0

    while offset + window_size <= samples.size:
        vad.accept_waveform(samples[offset : offset + window_size])
        offset += window_size

    if offset < samples.size:
        tail = samples[offset:]
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: tail.size] = tail
        vad.accept_waveform(padded)

    vad.flush()
    return _drain_vad(vad)


def _evaluate_segments(model, segments: list[np.ndarray]) -> tuple[list[CandidateResult], int]:
    candidates: list[CandidateResult] = []
    skipped = 0

    for segment in segments:
        candidate_seconds = float(segment.size / SAMPLE_RATE)
        if not (
            MIN_WAKE_CANDIDATE_SECONDS
            <= candidate_seconds
            <= MAX_WAKE_CANDIDATE_SECONDS
        ):
            skipped += 1
            continue

        transcript, latency = transcribe_candidate(model, segment)
        candidates.append(
            CandidateResult(
                transcript=transcript,
                matched=wake_match(transcript),
                candidate_seconds=candidate_seconds,
                asr_latency_seconds=latency,
            )
        )

    return candidates, skipped


def _poisson_upper_mean(events: int, confidence: float = CONFIDENCE) -> float:
    """Exact one-sided Poisson upper confidence bound for the event mean."""
    if events < 0:
        raise ValueError("events must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    target_cdf = 1.0 - confidence
    if events == 0:
        return -math.log(target_cdf)

    def poisson_cdf(k: int, lam: float) -> float:
        term = math.exp(-lam)
        total = term
        for i in range(1, k + 1):
            term *= lam / i
            total += term
        return total

    low = 0.0
    high = max(1.0, float(events + 1))
    while poisson_cdf(events, high) > target_cdf:
        high *= 2.0
        if high > 1_000_000:
            raise RuntimeError("Could not bracket Poisson confidence bound.")

    for _ in range(100):
        mid = (low + high) / 2.0
        if poisson_cdf(events, mid) > target_cdf:
            low = mid
        else:
            high = mid
    return high


def _fp_metrics(events: int, elapsed_seconds: float) -> tuple[float, float]:
    hours = max(elapsed_seconds / 3600.0, 1e-12)
    observed = events / hours
    upper95 = _poisson_upper_mean(events, CONFIDENCE) / hours
    return observed, upper95


def run_self_test() -> int:
    failures: list[str] = []

    if wake_match("Bunnelby"):
        failures.append("bare Bunnelby must remain rejected")
    if not wake_match("Hey Bunnelby"):
        failures.append("canonical wake phrase must match")

    expected_zero_upper = -math.log(0.05) / 6.0
    observed, upper = _fp_metrics(0, 6 * 3600)
    if abs(observed) > 1e-12:
        failures.append("zero-event observed FP rate should be zero")
    if abs(upper - expected_zero_upper) > 1e-9:
        failures.append("Poisson zero-event upper bound calculation mismatch")

    observed, _ = _fp_metrics(10, 10 * 3600)
    if observed <= TARGET_MAX_FP_PER_HOUR:
        failures.append("FP target gate arithmetic is incorrect")

    if failures:
        print("BUNNELBY WAKE CERTIFICATION SELF TEST: FAILED")
        for failure in failures:
            print(" -", failure)
        return 1

    print("BUNNELBY WAKE CERTIFICATION SELF TEST: PASS")
    print("Attempt-level recall is bounded at 100%: one hit per prompted attempt.")
    print("Negative certification reports raw and debounced false-positive rates separately.")
    print("95% one-sided Poisson confidence bound: PASS")
    return 0


def run_positive(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    model = load_wake_asr()
    device_index, device = default_microphone(args.device)

    print()
    print("=" * 76)
    print("BUNNELBY WAKE CERTIFICATION — POSITIVE RECALL")
    print("=" * 76)
    print(f"Microphone: [{device_index}] {device['name']}")
    print(f"Attempts: {args.attempts}")
    print(f"Per-attempt recording: {args.record_seconds:.1f}s")
    print("Say exactly: Hey Bunnelby")
    print("One prompted recording = one recall trial.")
    print("Audio saved: NO")
    print()

    attempts: list[PositiveAttempt] = []
    total_latency = 0.0
    total_asr_calls = 0
    skipped_duration = 0

    for attempt_no in range(1, args.attempts + 1):
        input(f"[{attempt_no:02d}/{args.attempts:02d}] Press Enter, then say 'Hey Bunnelby'...")
        audio = _capture_audio(device_index, args.record_seconds)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0

        segments = _speech_segments_from_audio(audio, model_path)
        candidates, skipped = _evaluate_segments(model, segments)
        skipped_duration += skipped
        detected = any(candidate.matched for candidate in candidates)

        total_asr_calls += len(candidates)
        total_latency += sum(candidate.asr_latency_seconds for candidate in candidates)

        attempts.append(
            PositiveAttempt(
                attempt=attempt_no,
                detected=detected,
                rms=rms,
                peak=peak,
                speech_segments=len(segments),
                candidates=candidates,
            )
        )

        best_text = " | ".join(
            candidate.transcript for candidate in candidates if candidate.transcript
        )
        print(
            f"  {'PASS' if detected else 'MISS'}"
            f" | speech_segments={len(segments)}"
            f" | transcript={best_text!r}"
        )

    hits = sum(1 for attempt in attempts if attempt.detected)
    recall = hits / args.attempts if args.attempts else 0.0
    avg_latency = total_latency / total_asr_calls if total_asr_calls else 0.0

    if recall >= TARGET_PREFERRED_RECALL:
        decision = "POSITIVE_STRONG_PASS"
    elif recall >= TARGET_MIN_RECALL:
        decision = "POSITIVE_MINIMUM_PASS"
    else:
        decision = "POSITIVE_FAIL"

    payload = {
        "schema": "bunnelby.wake-certification.positive.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wake_phrase": "Hey Bunnelby",
        "engine": "Silero VAD -> faster-whisper base.en CPU/int8 -> strict matcher",
        "attempts": args.attempts,
        "hits": hits,
        "recall": recall,
        "preferred_recall_target": TARGET_PREFERRED_RECALL,
        "minimum_recall_target": TARGET_MIN_RECALL,
        "average_asr_latency_seconds": avg_latency,
        "skipped_by_duration_gate": skipped_duration,
        "audio_saved": False,
        "results": [
            {
                **asdict(attempt),
                "candidates": [asdict(candidate) for candidate in attempt.candidates],
            }
            for attempt in attempts
        ],
        "decision": decision,
    }
    report = _save_report("positive", payload)

    print()
    print("=" * 76)
    print("BUNNELBY WAKE CERTIFICATION POSITIVE RESULT")
    print("=" * 76)
    print(f"Detected attempts: {hits}/{args.attempts}")
    print(f"Attempt-level recall: {recall * 100.0:.1f}%")
    print(f"Average candidate ASR latency: {avg_latency:.2f}s")
    print(f"Skipped by duration gate: {skipped_duration}")
    print(f"Decision: {decision}")
    print(f"Report: {report}")
    print("Audio saved: NO")
    return 0 if recall >= TARGET_MIN_RECALL else 2


def run_negative_live(args: argparse.Namespace) -> int:
    model_path = ensure_silero_vad_model()
    vad, window_size = create_vad(model_path)
    model = load_wake_asr()
    device_index, device = default_microphone(args.device)

    print()
    print("=" * 76)
    print("BUNNELBY WAKE CERTIFICATION — NEGATIVE LIVE")
    print("=" * 76)
    print(f"Microphone: [{device_index}] {device['name']}")
    print(f"Duration: {args.duration:.0f}s ({args.duration / 3600.0:.3f}h)")
    print("IMPORTANT: Do NOT say the wake phrase during this run.")
    print("Use normal English/Hindi/Hinglish speech, commands, media, and room audio.")
    print("Certification ASRs every qualifying speech segment even during debounce.")
    print("Audio saved: NO")
    print("Only actual false-trigger transcripts are persisted for diagnosis.")
    print()

    stats = NegativeStats()
    false_triggers: list[dict[str, object]] = []
    read_size = int(0.10 * SAMPLE_RATE)
    pending = np.empty(0, dtype=np.float32)
    started = time.monotonic()
    last_debounced_wake = -1e9

    def process_ready_segments() -> None:
        nonlocal last_debounced_wake
        while not vad.empty():
            segment = np.asarray(vad.front.samples, dtype=np.float32).reshape(-1).copy()
            vad.pop()
            stats.speech_segments += 1

            candidate_seconds = float(segment.size / SAMPLE_RATE)
            if not (
                MIN_WAKE_CANDIDATE_SECONDS
                <= candidate_seconds
                <= MAX_WAKE_CANDIDATE_SECONDS
            ):
                stats.skipped_duration += 1
                continue

            transcript, latency = transcribe_candidate(model, segment)
            stats.asr_calls += 1
            stats.asr_latency_total += latency
            matched = wake_match(transcript)

            if args.debug_transcripts:
                print(
                    f"[{'FALSE-WAKE' if matched else 'speech'}] {transcript!r}"
                    f" | segment={candidate_seconds:.2f}s | asr={latency:.2f}s"
                )

            if not matched:
                continue

            stats.raw_match_events += 1
            now = time.monotonic()
            debounced = now - last_debounced_wake >= args.cooldown
            if debounced:
                stats.debounced_wake_events += 1
                last_debounced_wake = now

            false_triggers.append(
                {
                    "elapsed_seconds": round(now - started, 3),
                    "transcript": transcript,
                    "candidate_seconds": round(candidate_seconds, 4),
                    "asr_latency_seconds": round(latency, 4),
                    "debounced_event": debounced,
                }
            )
            print(
                f"FALSE WAKE #{stats.raw_match_events}: {transcript!r}"
                f" | debounced={'YES' if debounced else 'NO'}"
            )

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        ) as microphone:
            while time.monotonic() - started < args.duration:
                samples, overflowed = microphone.read(read_size)
                if overflowed:
                    print("WARNING: microphone input overflow")

                chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
                pending = np.concatenate((pending, chunk))

                while pending.size >= window_size:
                    vad.accept_waveform(pending[:window_size])
                    pending = pending[window_size:]

                process_ready_segments()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Finalizing partial negative run.")

    if pending.size:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: pending.size] = pending
        vad.accept_waveform(padded)
    vad.flush()
    process_ready_segments()

    stats.elapsed_seconds = max(0.0, time.monotonic() - started)
    avg_latency = stats.asr_latency_total / stats.asr_calls if stats.asr_calls else 0.0
    raw_rate, raw_upper95 = _fp_metrics(stats.raw_match_events, stats.elapsed_seconds)
    debounced_rate, debounced_upper95 = _fp_metrics(
        stats.debounced_wake_events, stats.elapsed_seconds
    )

    enough_confidence = raw_upper95 <= TARGET_MAX_FP_PER_HOUR
    observed_pass = raw_rate <= TARGET_MAX_FP_PER_HOUR

    if observed_pass and enough_confidence:
        decision = "NEGATIVE_CERTIFIED_PASS"
    elif stats.raw_match_events == 0:
        decision = "NEGATIVE_QUICK_PASS_NOT_YET_CERTIFIED"
    else:
        decision = "NEGATIVE_FAIL_OR_NEEDS_REFINEMENT"

    payload = {
        "schema": "bunnelby.wake-certification.negative-live.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wake_phrase_present_by_instruction": False,
        "engine": "Silero VAD -> faster-whisper base.en CPU/int8 -> strict matcher",
        "duration_seconds": stats.elapsed_seconds,
        "duration_hours": stats.elapsed_seconds / 3600.0,
        "speech_segments": stats.speech_segments,
        "asr_calls": stats.asr_calls,
        "skipped_by_duration_gate": stats.skipped_duration,
        "raw_false_matches": stats.raw_match_events,
        "debounced_false_wakes": stats.debounced_wake_events,
        "raw_fp_per_hour": raw_rate,
        "raw_fp_per_hour_95pct_upper": raw_upper95,
        "debounced_fp_per_hour": debounced_rate,
        "debounced_fp_per_hour_95pct_upper": debounced_upper95,
        "target_max_fp_per_hour": TARGET_MAX_FP_PER_HOUR,
        "average_asr_latency_seconds": avg_latency,
        "cooldown_seconds_for_debounced_metric_only": args.cooldown,
        "audio_saved": False,
        "false_triggers": false_triggers,
        "decision": decision,
    }
    report = _save_report("negative_live", payload)

    print()
    print("=" * 76)
    print("BUNNELBY WAKE CERTIFICATION NEGATIVE RESULT")
    print("=" * 76)
    print(f"Elapsed: {stats.elapsed_seconds:.1f}s ({stats.elapsed_seconds / 3600.0:.3f}h)")
    print(f"Speech segments: {stats.speech_segments}")
    print(f"ASR calls: {stats.asr_calls}")
    print(f"Raw false matches: {stats.raw_match_events}")
    print(f"Debounced false wake events: {stats.debounced_wake_events}")
    print(f"Raw FP/hour: {raw_rate:.3f}")
    print(f"Raw FP/hour 95% upper bound: {raw_upper95:.3f}")
    print(f"Debounced FP/hour: {debounced_rate:.3f}")
    print(f"Average candidate ASR latency: {avg_latency:.2f}s")
    print(f"Target: <= {TARGET_MAX_FP_PER_HOUR:.2f} false positives/hour")
    print(f"Decision: {decision}")
    print(f"Report: {report}")
    print("Audio saved: NO")
    return 0 if observed_pass else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bunnelby wake-word certification harness."
    )
    parser.add_argument(
        "--mode",
        choices=("positive", "negative-live"),
        help="Certification mode.",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_POSITIVE_ATTEMPTS,
        help="Prompted positive attempts.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=DEFAULT_RECORD_SECONDS,
        help="Seconds recorded into RAM per positive attempt.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_NEGATIVE_SECONDS,
        help="Negative live-test duration in seconds.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_DEBOUNCE_SECONDS,
        help="Debounce window used only for the user-visible event metric.",
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--debug-transcripts",
        action="store_true",
        help="Print all negative candidate transcripts; off by default for privacy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.self_test:
        return run_self_test()

    if not args.mode:
        raise SystemExit("Choose --mode positive or --mode negative-live.")

    args.attempts = max(1, min(int(args.attempts), 100))
    args.record_seconds = max(1.5, min(float(args.record_seconds), 10.0))
    args.duration = max(10.0, min(float(args.duration), 24 * 3600.0))
    args.cooldown = max(0.5, min(float(args.cooldown), 30.0))

    if args.mode == "positive":
        return run_positive(args)
    return run_negative_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
