from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException

from services.api.app import main, stt_service


class STTServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        stt_service._reset_model_cache_for_tests()

    def test_defaults_lock_cpu_int8_small_model_and_beam5(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STT_MODEL", None)
            os.environ.pop("STT_DEVICE", None)
            os.environ.pop("STT_COMPUTE_TYPE", None)
            os.environ.pop("STT_BEAM_SIZE", None)
            os.environ.pop("STT_CONTEXT_BIAS_ENABLED", None)
            os.environ.pop("STT_HOTWORDS", None)
            os.environ.pop("STT_HOTWORDS_HI", None)
            self.assertEqual(stt_service.stt_model_name(), "small")
            self.assertEqual(stt_service.stt_device(), "cpu")
            self.assertEqual(stt_service.stt_compute_type(), "int8")
            self.assertEqual(stt_service.stt_beam_size(), 5)
            self.assertFalse(stt_service.stt_context_bias_enabled())
            self.assertIsNone(stt_service.stt_hotwords())
            self.assertIsNone(stt_service.stt_hindi_hotwords())

    def test_empty_audio_fails_before_model_load(self) -> None:
        with patch.object(stt_service, "_load_model") as load_model:
            with self.assertRaises(stt_service.STTAudioError):
                stt_service.transcribe_audio(b"")
        load_model.assert_not_called()

    def test_invalid_language_fails_closed(self) -> None:
        with patch.object(stt_service, "_load_model") as load_model:
            with self.assertRaises(stt_service.STTAudioError):
                stt_service.transcribe_audio(b"audio", language="fr")
        load_model.assert_not_called()

    def test_disabled_stt_fails_before_model_load(self) -> None:
        with (
            patch.dict(os.environ, {"STT_ENABLED": "false"}),
            patch.object(stt_service, "_load_model") as load_model,
        ):
            with self.assertRaises(stt_service.STTDisabledError):
                stt_service.transcribe_audio(b"audio")
        load_model.assert_not_called()

    def test_transcription_collects_segments_and_metadata(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(
                        text=" Hello ", avg_logprob=-0.20, no_speech_prob=0.05
                    ),
                    SimpleNamespace(
                        text="Bunnelby", avg_logprob=-0.40, no_speech_prob=0.10
                    ),
                ]
            ),
            SimpleNamespace(language="en", language_probability=0.98, duration=1.7),
        )
        with (
            patch.object(stt_service, "_load_model", return_value=model),
            patch("services.api.app.stt_service.tempfile.NamedTemporaryFile") as named_temp,
            patch("services.api.app.stt_service.Path.unlink"),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("STT_BEAM_SIZE", None)
            os.environ.pop("STT_CONTEXT_BIAS_ENABLED", None)
            handle = Mock()
            handle.name = "C:/Temp/bunnelby-test.webm"
            named_temp.return_value.__enter__.return_value = handle
            result = stt_service.transcribe_audio(
                b"fake-audio",
                content_type="audio/webm;codecs=opus",
                language="auto",
            )

        self.assertEqual(result.text, "Hello Bunnelby")
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.language_probability, 0.98)
        self.assertAlmostEqual(result.duration_seconds, 1.7)
        self.assertAlmostEqual(result.average_log_probability, -0.30)
        self.assertAlmostEqual(result.max_no_speech_probability, 0.10)
        handle.write.assert_called_once_with(b"fake-audio")
        kwargs = model.transcribe.call_args.kwargs
        self.assertIsNone(kwargs["language"])
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertTrue(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertIsNone(kwargs["hotwords"])

    def test_transcription_metadata_is_optional_for_backward_compatibility(self) -> None:
        result = stt_service.TranscriptionResult("ready", "en", 1.0, 0.5)
        self.assertIsNone(result.average_log_probability)
        self.assertIsNone(result.max_no_speech_probability)

    def test_low_confidence_unsupported_auto_language_is_not_trusted(self) -> None:
        result = stt_service.TranscriptionResult(
            "Nautipede Kholu",
            "ml",
            0.541,
            1.2,
            average_log_probability=-0.25,
            max_no_speech_probability=0.05,
        )
        self.assertFalse(
            stt_service.microphone_transcription_is_trustworthy(
                result, language_hint="auto"
            )
        )

    def test_hinglish_language_probability_alone_does_not_reject_primary_language(self) -> None:
        result = stt_service.TranscriptionResult(
            "Calculator kholo",
            "en",
            0.41,
            1.0,
            average_log_probability=-0.30,
            max_no_speech_probability=0.02,
        )
        self.assertTrue(
            stt_service.microphone_transcription_is_trustworthy(
                result, language_hint="auto"
            )
        )

    def test_poor_acoustic_evidence_is_not_trusted(self) -> None:
        result = stt_service.TranscriptionResult(
            "check my email",
            "en",
            0.96,
            1.0,
            average_log_probability=-1.35,
            max_no_speech_probability=0.10,
        )
        self.assertFalse(
            stt_service.microphone_transcription_is_trustworthy(
                result, language_hint="auto"
            )
        )

    def test_uncertain_ram_transcript_returns_empty_text_before_dispatch(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(
                        text=" Nautipede Kholu ",
                        avg_logprob=-0.25,
                        no_speech_prob=0.05,
                    )
                ]
            ),
            SimpleNamespace(language="ml", language_probability=0.541, duration=1.2),
        )
        with (
            patch.dict(
                os.environ,
                {"STT_CONTEXT_BIAS_ENABLED": "false", "STT_HOTWORDS": "Gmail calendar"},
            ),
            patch.object(stt_service, "_load_model", return_value=model),
        ):
            result = stt_service.transcribe_samples(
                np.zeros(16_000, dtype=np.float32), language="auto"
            )

        self.assertEqual(result.text, "")
        self.assertEqual(result.language, "ml")
        self.assertAlmostEqual(result.language_probability, 0.541)
        self.assertIsNone(model.transcribe.call_args.kwargs["hotwords"])

    def test_explicit_hindi_language_hint_is_forwarded(self) -> None:
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" नमस्ते ")]),
            SimpleNamespace(language="hi", language_probability=1.0, duration=0.8),
        )
        with (
            patch.object(stt_service, "_load_model", return_value=model),
            patch("services.api.app.stt_service.tempfile.NamedTemporaryFile") as named_temp,
            patch("services.api.app.stt_service.Path.unlink"),
        ):
            handle = Mock()
            handle.name = "C:/Temp/bunnelby-test.wav"
            named_temp.return_value.__enter__.return_value = handle
            result = stt_service.transcribe_audio(b"audio", content_type="audio/wav", language="hi")

        self.assertEqual(result.text, "नमस्ते")
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "hi")

    def test_api_maps_bad_audio_to_400(self) -> None:
        with patch.object(main, "transcribe_audio", side_effect=stt_service.STTAudioError("bad audio")):
            with self.assertRaises(HTTPException) as context:
                main.speech_to_text(b"audio", "auto", "audio/webm")
        self.assertEqual(context.exception.status_code, 400)

    def test_api_maps_unavailable_model_to_503(self) -> None:
        with patch.object(main, "transcribe_audio", side_effect=stt_service.STTUnavailableError("missing")):
            with self.assertRaises(HTTPException) as context:
                main.speech_to_text(b"audio", "auto", "audio/webm")
        self.assertEqual(context.exception.status_code, 503)

    def test_api_returns_transcription_metadata(self) -> None:
        fake = stt_service.TranscriptionResult(
            text="check my calendar",
            language="en",
            language_probability=0.96,
            duration_seconds=1.4,
        )
        with patch.object(main, "transcribe_audio", return_value=fake):
            response = main.speech_to_text(b"audio", "auto", "audio/webm")
        self.assertEqual(response.text, "check my calendar")
        self.assertEqual(response.language, "en")
        self.assertAlmostEqual(response.language_probability, 0.96)


if __name__ == "__main__":
    unittest.main()
