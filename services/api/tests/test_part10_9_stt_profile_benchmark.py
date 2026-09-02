from __future__ import annotations

import unittest

from scripts.wakeword.stt_profile_benchmark import BenchmarkResult, select_profile, word_error_rate


def _result(identifier: str, latency: float, error_rate: float) -> BenchmarkResult:
    return BenchmarkResult(identifier, "transcript", latency, error_rate)


class STTProfileBenchmarkTests(unittest.TestCase):
    def test_word_error_rate_handles_exact_substitution_insertion_and_empty(self) -> None:
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)
        self.assertEqual(word_error_rate("hello world", "hello there"), 0.5)
        self.assertEqual(word_error_rate("hello", "hello there"), 1.0)
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "unexpected"), 1.0)

    def test_gpu_requires_complete_corpus_and_at_least_fifteen_percent_speedup(self) -> None:
        cpu = [_result("a", 1.0, 0.1), _result("b", 2.0, 0.1)]
        fast_gpu = [_result("a", 0.7, 0.1), _result("b", 1.0, 0.1)]
        marginal_gpu = [_result("a", 1.0, 0.1), _result("b", 1.6, 0.1)]

        self.assertEqual(select_profile(cpu, fast_gpu), "GPU/int8_float16")
        self.assertEqual(select_profile(cpu, marginal_gpu), "CPU/int8")
        self.assertEqual(select_profile(cpu, fast_gpu[:1]), "CPU/int8")

    def test_gpu_is_rejected_when_mean_error_regresses_materially(self) -> None:
        cpu = [_result("a", 1.0, 0.10), _result("b", 2.0, 0.10)]
        gpu = [_result("a", 0.5, 0.16), _result("b", 0.8, 0.16)]

        self.assertEqual(select_profile(cpu, gpu), "CPU/int8")


if __name__ == "__main__":
    unittest.main()
