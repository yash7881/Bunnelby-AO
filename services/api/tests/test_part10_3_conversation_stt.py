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
            patch.dict(os.environ, {"STT_HOTWORDS": ""}),
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
        self.assertIsNone(kwargs["hotwords"])
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

    def test_native_inference_failure_invalidates_poisoned_model_cache(self) -> None:
        model = Mock()
        model.model = Mock()

        def failing_segments():
            raise RuntimeError("deferred CUDA encoder failure")
            yield  # pragma: no cover - makes this a generator

        model.transcribe.return_value = (
            failing_segments(),
            SimpleNamespace(language="en", language_probability=1.0, duration=1.0),
        )
        stt_service._model = model
        stt_service._model_signature = stt_service._model_config_signature()

        with self.assertRaises(stt_service.STTTranscriptionError):
            stt_service.transcribe_samples(np.zeros(16_000, dtype=np.float32))

        self.assertIsNone(stt_service._model)
        self.assertIsNone(stt_service._model_signature)
        model.model.unload_model.assert_called_once_with()

    def test_successful_inference_keeps_warm_model_cached(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" ready ")]),
            SimpleNamespace(language="en", language_probability=1.0, duration=1.0),
        )
        stt_service._model = model
        stt_service._model_signature = stt_service._model_config_signature()

        result = stt_service.transcribe_samples(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.text, "ready")
        self.assertIs(stt_service._model, model)

    def test_runtime_profile_reports_effective_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "STT_MODEL": "small",
                "STT_DEVICE": "cuda",
                "STT_COMPUTE_TYPE": "int8_float16",
                "STT_BEAM_SIZE": "3",
                "STT_HOTWORDS": "Bunnelby   Gmail calendar",
            },
        ):
            profile = stt_service.stt_runtime_profile()
        self.assertEqual(profile.model, "small")
        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.compute_type, "int8_float16")
        self.assertEqual(profile.beam_size, 3)
        self.assertEqual(profile.hotwords, "Bunnelby Gmail calendar")

    def test_optional_hotwords_are_decoder_context_not_transcript_rewriting(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" unread email ")]),
            SimpleNamespace(language="en", language_probability=1.0, duration=1.0),
        )
        with (
            patch.dict(os.environ, {"STT_HOTWORDS": "Gmail calendar unread"}),
            patch.object(stt_service, "_load_model", return_value=model),
        ):
            result = stt_service.transcribe_samples(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.text, "unread email")
        self.assertEqual(model.transcribe.call_args.kwargs["hotwords"], "Gmail calendar unread")

    def test_per_call_context_override_is_forwarded_without_changing_global_config(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" कल का कैलेंडर ")]),
            SimpleNamespace(language="hi", language_probability=1.0, duration=1.0),
        )
        with (
            patch.dict(os.environ, {"STT_HOTWORDS": "English context"}),
            patch.object(stt_service, "_load_model", return_value=model),
        ):
            stt_service.transcribe_samples(
                np.zeros(16_000, dtype=np.float32),
                language="hi",
                hotwords_override="कल कैलेंडर",
            )

        self.assertEqual(model.transcribe.call_args.kwargs["hotwords"], "कल कैलेंडर")


if __name__ == "__main__":
    unittest.main()
