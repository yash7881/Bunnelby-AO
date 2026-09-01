from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from services.api.app import stt_service


class ConversationSTTTests(unittest.TestCase):
    def tearDown(self) -> None:
        stt_service._reset_model_cache_for_tests()

    def test_ram_samples_transcribe_without_temp_file_or_second_vad(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(text=" Check tomorrow's calendar,"),
                    SimpleNamespace(text=" then read my latest emails."),
                ]
            ),
            SimpleNamespace(language="en", language_probability=0.99, duration=8.4),
        )
        samples = np.zeros(16_000 * 9, dtype=np.float32)

        with (
            patch.object(stt_service, "_load_model", return_value=model),
            patch("services.api.app.stt_service.tempfile.NamedTemporaryFile") as named_temp,
        ):
            result = stt_service.transcribe_samples(samples, language="auto")

        named_temp.assert_not_called()
        self.assertEqual(
            result.text,
            "Check tomorrow's calendar, then read my latest emails.",
        )
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.language_probability, 0.99)
        kwargs = model.transcribe.call_args.kwargs
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertIsNone(kwargs["language"])
        waveform = model.transcribe.call_args.args[0]
        self.assertIsInstance(waveform, np.ndarray)
        self.assertTrue(waveform.flags["C_CONTIGUOUS"])

    def test_ram_samples_reject_wrong_sample_rate_before_model_load(self) -> None:
        with patch.object(stt_service, "_load_model") as load_model:
            with self.assertRaises(stt_service.STTAudioError):
                stt_service.transcribe_samples(
                    np.ones(8000, dtype=np.float32),
                    sample_rate=8000,
                )
        load_model.assert_not_called()

    def test_ram_samples_reject_non_finite_audio(self) -> None:
        samples = np.array([0.0, np.nan, 0.1], dtype=np.float32)
        with patch.object(stt_service, "_load_model") as load_model:
            with self.assertRaises(stt_service.STTAudioError):
                stt_service.transcribe_samples(samples)
        load_model.assert_not_called()

    def test_ram_samples_enforce_single_turn_duration_bound(self) -> None:
        samples = np.zeros(16_000 * 6, dtype=np.float32)
        with (
            patch.dict(os.environ, {"STT_MAX_SAMPLE_SECONDS": "5"}),
            patch.object(stt_service, "_load_model") as load_model,
        ):
            with self.assertRaises(stt_service.STTAudioError):
                stt_service.transcribe_samples(samples)
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
