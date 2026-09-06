from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.stt_service import transcribe_samples, unload_stt_model

if __package__:
    from .always_on_wake_listener import SAMPLE_RATE, default_microphone, ensure_silero_vad_model
    from .wake_conversation_runtime import READ_SAMPLES, capture_conversation_turn
else:
    from always_on_wake_listener import SAMPLE_RATE, default_microphone, ensure_silero_vad_model
    from wake_conversation_runtime import READ_SAMPLES, capture_conversation_turn


@dataclass(frozen=True)
class BenchmarkPrompt:
    identifier: str
    language: str
    text: str


@dataclass(frozen=True)
class BenchmarkResult:
    identifier: str
    transcript: str
    latency_seconds: float
    word_error_rate: float


PROMPTS = (
    BenchmarkPrompt("en_short", "en", "Check tomorrow's calendar."),
    BenchmarkPrompt("en_medium", "en", "Read my latest unread emails."),
    BenchmarkPrompt(
        "en_long",
        "en",
        "Check tomorrow's calendar and read my latest unread emails, then tell me what I should focus on first.",
    ),
    BenchmarkPrompt("hi_short", "hi", "कल का कैलेंडर चेक करो।"),
    BenchmarkPrompt("hinglish_medium", "auto", "Mere latest unread emails batao."),
    BenchmarkPrompt(
        "hinglish_long",
        "auto",
        "Kal ka calendar aur latest unread emails check karke batao ki mujhe sabse pehle kis cheez par focus karna chahiye.",
    ),
)

BENCHMARK_HOTWORDS = "Bunnelby Gmail calendar latest unread email emails"

PROFILES = (
    ("CPU/int8+context", "cpu", "int8", BENCHMARK_HOTWORDS),
    ("GPU/int8_float16+context", "cuda", "int8_float16", BENCHMARK_HOTWORDS),
    ("GPU/int8_float16+no-context", "cuda", "int8_float16", ""),
)


def _sounddevice():
    """Import PortAudio only for a live benchmark, not while CI imports helpers."""
    try:
        import sounddevice
    except Exception as exc:
        raise RuntimeError("sounddevice/PortAudio is unavailable for live microphone use.") from exc
    return sounddevice


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _tokens(reference)
    actual = _tokens(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_token in enumerate(expected, start=1):
        current = [row]
        for column, actual_token in enumerate(actual, start=1):
            substitution = previous[column - 1] + (expected_token != actual_token)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(expected)


def select_profile(
    cpu: list[BenchmarkResult],
    gpu: list[BenchmarkResult],
) -> str:
    """Choose GPU only when the same complete corpus proves a useful safe win."""
    if not cpu or len(cpu) != len(gpu):
        return "CPU/int8"
    cpu_median = statistics.median(result.latency_seconds for result in cpu)
    gpu_median = statistics.median(result.latency_seconds for result in gpu)
    cpu_wer = statistics.fmean(result.word_error_rate for result in cpu)
    gpu_wer = statistics.fmean(result.word_error_rate for result in gpu)
    if gpu_median < cpu_median * 0.85 and gpu_wer <= cpu_wer + 0.05:
        return "GPU/int8_float16"
    return "CPU/int8"


def select_context_policy(
    contextual: list[BenchmarkResult],
    baseline: list[BenchmarkResult],
) -> str:
    """Enable context only when the same corpus improves mean WER without a bad outlier."""
    if not contextual or len(contextual) != len(baseline):
        return "disabled"
    contextual_mean = statistics.fmean(result.word_error_rate for result in contextual)
    baseline_mean = statistics.fmean(result.word_error_rate for result in baseline)
    worst_regression = max(
        context.word_error_rate - plain.word_error_rate
        for context, plain in zip(contextual, baseline, strict=True)
    )
    if contextual_mean < baseline_mean and worst_regression <= 0.25:
        return "enabled"
    return "disabled"


@contextmanager
def _stt_profile(device: str, compute_type: str, hotwords: str):
    names = (
        "STT_MODEL",
        "STT_DEVICE",
        "STT_COMPUTE_TYPE",
        "STT_BEAM_SIZE",
        "STT_HOTWORDS",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(
        {
            "STT_MODEL": "small",
            "STT_DEVICE": device,
            "STT_COMPUTE_TYPE": compute_type,
            "STT_BEAM_SIZE": "5",
            "STT_HOTWORDS": hotwords,
        }
    )
    unload_stt_model()
    try:
        yield
    finally:
        unload_stt_model()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _benchmark_profile(
    label: str,
    device: str,
    compute_type: str,
    hotwords: str,
    corpus: list[tuple[BenchmarkPrompt, np.ndarray]],
) -> list[BenchmarkResult]:
    print()
    print(f"PROFILE: {label} | model=small | beam=5")
    with _stt_profile(device, compute_type, hotwords):
        warm_started = time.perf_counter()
        transcribe_samples(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            language="en",
        )
        print(f"Warm-up/load: {time.perf_counter() - warm_started:.2f}s")

        results: list[BenchmarkResult] = []
        for prompt, samples in corpus:
            started = time.perf_counter()
            result = transcribe_samples(
                samples,
                sample_rate=SAMPLE_RATE,
                language=prompt.language,
            )
            latency = time.perf_counter() - started
            error_rate = word_error_rate(prompt.text, result.text)
            results.append(BenchmarkResult(prompt.identifier, result.text, latency, error_rate))
            print(
                f"  {prompt.identifier}: {latency:.2f}s | WER={error_rate:.3f} | "
                f"{result.text!r}"
            )
        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAM-only CPU vs GPU benchmark for Bunnelby multilingual conversation STT."
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--prompt-count", type=int, default=len(PROMPTS))
    parser.add_argument("--command-wait", type=float, default=20.0)
    parser.add_argument("--conversation-silence", type=float, default=1.0)
    parser.add_argument("--max-utterance", type=float, default=60.0)
    args = parser.parse_args()
    prompt_count = max(1, min(int(args.prompt_count), len(PROMPTS)))
    args.command_wait = max(5.0, min(float(args.command_wait), 60.0))
    args.conversation_silence = max(0.5, min(float(args.conversation_silence), 2.0))
    args.max_utterance = max(5.0, min(float(args.max_utterance), 120.0))

    model_path = ensure_silero_vad_model()
    device_index, device = default_microphone(args.device)
    print("=" * 78)
    print("BUNNELBY STT CPU VS GPU BENCHMARK")
    print("=" * 78)
    print(f"Microphone: [{device_index}] {device['name']}")
    print(f"Prompts: {prompt_count}")
    print("Audio storage: RAM ONLY; nothing is written to disk")

    corpus: list[tuple[BenchmarkPrompt, np.ndarray]] = []
    try:
        sd = _sounddevice()
        with sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=READ_SAMPLES,
        ) as microphone:
            for index, prompt in enumerate(PROMPTS[:prompt_count], start=1):
                print()
                print(f"PROMPT {index}/{prompt_count} [{prompt.identifier}]")
                print(prompt.text)
                input("Press Enter, then speak the sentence exactly once: ")
                captured = capture_conversation_turn(
                    microphone,
                    model_path,
                    SimpleNamespace(
                        conversation_silence=args.conversation_silence,
                        max_utterance=args.max_utterance,
                    ),
                    wait_seconds=args.command_wait,
                )
                if captured is None:
                    raise RuntimeError(f"No completed speech detected for {prompt.identifier}.")
                corpus.append((prompt, captured.samples.copy()))
                print(f"Captured {captured.speech_seconds:.2f}s in RAM")

        all_results: dict[str, list[BenchmarkResult]] = {}
        for label, device_name, compute_type, hotwords in PROFILES:
            try:
                all_results[label] = _benchmark_profile(
                    label,
                    device_name,
                    compute_type,
                    hotwords,
                    corpus,
                )
            except Exception as exc:
                print(f"PROFILE FAILED: {label}: {exc}")

        print()
        print("SUMMARY")
        for label, results in all_results.items():
            latencies = [result.latency_seconds for result in results]
            mean_wer = statistics.fmean(result.word_error_rate for result in results)
            print(
                f"{label}: median={statistics.median(latencies):.2f}s "
                f"worst={max(latencies):.2f}s mean_WER={mean_wer:.3f}"
            )

        if len(all_results) == len(PROFILES):
            selected = select_profile(
                all_results["CPU/int8+context"],
                all_results["GPU/int8_float16+context"],
            )
            print(f"MEASURED PROFILE RECOMMENDATION: {selected}")
            context_policy = select_context_policy(
                all_results["GPU/int8_float16+context"],
                all_results["GPU/int8_float16+no-context"],
            )
            print(f"MEASURED CONTEXT RECOMMENDATION: {context_policy}")
            if context_policy == "enabled":
                print(f"STT_HOTWORDS={BENCHMARK_HOTWORDS}")
        else:
            print("MEASURED PROFILE RECOMMENDATION: CPU/int8 (GPU corpus gate incomplete)")
            print("MEASURED CONTEXT RECOMMENDATION: disabled (comparison incomplete)")
        print("Audio storage: RAM ONLY; captured corpus released at process exit")
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark cancelled; RAM audio will be released.")
        return 130
    except Exception as exc:
        print(f"BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        corpus.clear()
        unload_stt_model()


if __name__ == "__main__":
    raise SystemExit(main())
