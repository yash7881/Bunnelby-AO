from __future__ import annotations

import inspect
import io
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.wakeword import (
    always_on_wake_listener,
    stt_profile_benchmark,
    wake_conversation_runtime,
)
from services.api.app.audio_playback import PlaybackResult, PlaybackStatus
from services.api.app.stt_service import TranscriptionResult
from services.api.app.stt_service import STTTranscriptionError


class PersistentRuntimePolicyTests(unittest.TestCase):
    def _runtime_dependencies(self):
        input_stream = Mock()
        input_context = Mock()
        input_context.__enter__ = Mock(return_value=input_stream)
        input_context.__exit__ = Mock(return_value=False)
        return input_stream, input_context

    def test_strict_wake_normalization_and_confusable_rejection(self) -> None:
        self.assertTrue(always_on_wake_listener.wake_match("HEY, BUNNELBY!"))
        self.assertTrue(always_on_wake_listener.wake_match("Hey Bonilby"))
        self.assertTrue(always_on_wake_listener.wake_match("Hello Bunnelby"))
        self.assertTrue(always_on_wake_listener.wake_match("HELLO BONILBY!"))

        for phrase in (
            "Bunnelby",
            "hello",
            "hello buddy",
            "hello everyone",
            "hey buddy",
            "hey but there'll be",
            "bundle b",
            "calendar batao",
        ):
            self.assertFalse(always_on_wake_listener.wake_match(phrase), phrase)

    def test_product_defaults_are_persistent_and_ten_second_follow_up(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOLLOW_UP_SECONDS", None)
            args = wake_conversation_runtime.parse_args([])
        self.assertEqual(args.turns, 0)
        self.assertEqual(args.follow_up_seconds, 10.0)
        self.assertTrue(args.dispatch)
        self.assertTrue(args.tts)
        self.assertTrue(args.barge_in)

    def test_follow_up_environment_and_cli_bounds_are_deterministic(self) -> None:
        with patch.dict(os.environ, {"FOLLOW_UP_SECONDS": "12.5"}):
            from_environment = wake_conversation_runtime.parse_args([])
        self.assertEqual(from_environment.follow_up_seconds, 12.5)

        clamped = wake_conversation_runtime.parse_args(
            [
                "--follow-up-seconds",
                "999",
                "--conversation-silence",
                "0",
                "--max-utterance",
                "999",
                "--barge-in-echo-margin",
                "99",
            ]
        )
        self.assertEqual(clamped.follow_up_seconds, 60.0)
        self.assertEqual(clamped.conversation_silence, 0.5)
        self.assertEqual(clamped.max_utterance, 120.0)
        self.assertEqual(clamped.barge_in_echo_margin, 10.0)

    def test_tts_url_tracks_custom_chat_base(self) -> None:
        args = wake_conversation_runtime.parse_args(
            ["--api-url", "http://127.0.0.1:9000/chat"]
        )
        self.assertEqual(args.tts_url, "http://127.0.0.1:9000/tts")

    def test_state_boundary_discards_only_the_buffer_snapshot(self) -> None:
        class BufferedInput:
            read_available = 20_000

            def __init__(self) -> None:
                self.read_sizes: list[int] = []

            def read(self, frames: int):
                self.read_sizes.append(frames)
                return np.zeros((frames, 1), dtype=np.float32), False

        stream = BufferedInput()
        discarded = wake_conversation_runtime._discard_buffered_microphone(stream)

        self.assertEqual(discarded, 20_000)
        self.assertEqual(stream.read_sizes, [16_000, 4_000])

    def test_low_confidence_indic_auto_result_uses_stronger_hindi_rescue(self) -> None:
        automatic = TranscriptionResult("Kolka calendar checker", "te", 0.36, 2.0)
        hindi = TranscriptionResult("कल का कैलेंडर चेक करो", "hi", 1.0, 2.0)
        samples = np.zeros(16_000, dtype=np.float32)

        with (
            patch.object(
                wake_conversation_runtime,
                "transcribe_samples",
                side_effect=[automatic, hindi],
            ) as transcribe,
            patch.object(
                wake_conversation_runtime,
                "stt_hindi_hotwords",
                return_value="कल कैलेंडर",
            ),
        ):
            selected = wake_conversation_runtime._transcribe_conversation(samples, "auto")

        self.assertEqual(selected, hindi)
        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(transcribe.call_args_list[1].kwargs["language"], "hi")
        self.assertEqual(
            transcribe.call_args_list[1].kwargs["hotwords_override"],
            "कल कैलेंडर",
        )

    def test_normal_english_auto_result_does_not_retry(self) -> None:
        english = TranscriptionResult(
            "Check tomorrow's calendar", "en", 0.62, 2.0
        )
        with patch.object(
            wake_conversation_runtime,
            "transcribe_samples",
            return_value=english,
        ) as transcribe:
            selected = wake_conversation_runtime._transcribe_conversation(
                np.zeros(16_000, dtype=np.float32),
                "auto",
            )

        self.assertEqual(selected, english)
        transcribe.assert_called_once()

    def test_pathological_voice_transcripts_are_detected(self) -> None:
        self.assertTrue(
            wake_conversation_runtime._is_pathological_transcript(
                "\u0930\u093e\u0939\u0941\u0932 \u0915\u094b "
                + "\u0964 " * 30
            )
        )
        self.assertTrue(
            wake_conversation_runtime._is_pathological_transcript(
                "\u094c " * 30
            )
        )
        self.assertFalse(
            wake_conversation_runtime._is_pathological_transcript(
                "Rahul ko email send karo kal 9 PM milunga"
            )
        )
        self.assertFalse(
            wake_conversation_runtime._is_pathological_transcript(
                "\u092d\u093e\u0930\u0924 \u0915\u0947 "
                "\u0930\u093e\u0937\u094d\u091f\u094d\u0930\u092a\u0924\u093f "
                "\u0915\u094c\u0928 \u0939\u0948\u0902"
            )
        )

    def test_pathological_auto_transcript_runs_bounded_rescue(self) -> None:
        automatic = TranscriptionResult(
            "????? ?? " + "? " * 25,
            "hi",
            0.95,
            3.0,
        )
        hindi_bad = TranscriptionResult(
            "? " * 25,
            "hi",
            0.90,
            3.0,
        )
        english_good = TranscriptionResult(
            "Rahul ko email send karo and say I will meet him tomorrow at 9 PM",
            "en",
            0.76,
            3.0,
        )

        with patch.object(
            wake_conversation_runtime,
            "transcribe_samples",
            side_effect=[automatic, hindi_bad, english_good],
        ) as transcribe:
            selected = wake_conversation_runtime._transcribe_conversation(
                np.zeros(48_000, dtype=np.float32),
                "auto",
            )

        self.assertEqual(selected, english_good)
        self.assertEqual(transcribe.call_count, 3)

    def test_microphone_runtime_has_no_audio_file_persistence_path(self) -> None:
        source = inspect.getsource(wake_conversation_runtime)
        self.assertNotIn("NamedTemporaryFile", source)
        self.assertNotIn("soundfile.write", source)
        self.assertNotIn("wave.open", source)

    def test_live_audio_dependency_is_not_imported_at_module_load_time(self) -> None:
        modules = (
            always_on_wake_listener,
            stt_profile_benchmark,
            wake_conversation_runtime,
        )
        for module in modules:
            self.assertNotIn("\nimport sounddevice", inspect.getsource(module))

    def test_complete_two_turn_session_uses_one_wake_and_one_follow_up(self) -> None:
        args = wake_conversation_runtime.parse_args(["--turns", "2"])
        now = time.monotonic()
        first = wake_conversation_runtime.CapturedUtterance(
            np.zeros(16_000, dtype=np.float32), now, now + 1.0
        )
        follow_up = wake_conversation_runtime.CapturedUtterance(
            np.zeros(16_000, dtype=np.float32), now + 5.0, now + 6.0
        )
        playback = PlaybackResult(
            status=PlaybackStatus.COMPLETED,
            queued_at=now - 3.0,
            started_at=now - 2.0,
            first_audio_at=now - 1.9,
            finished_at=now,
            frames_played=24_000,
        )
        handle = Mock(status=PlaybackStatus.STARTED)
        input_stream = Mock()
        input_context = Mock()
        input_context.__enter__ = Mock(return_value=input_stream)
        input_context.__exit__ = Mock(return_value=False)
        player = Mock()
        player.start.return_value = handle
        transcript = TranscriptionResult("check calendar", "en", 1.0, 1.0)
        response = {
            "reply": "You have one event.",
            "spoken_reply": "You have one event.",
            "spoken_language": "en",
        }
        output = io.StringIO()

        with (
            patch.object(wake_conversation_runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(wake_conversation_runtime, "load_wake_asr", return_value=Mock()),
            patch.object(
                wake_conversation_runtime,
                "default_microphone",
                return_value=(1, {"name": "test mic"}),
            ),
            patch.object(
                wake_conversation_runtime,
                "_sounddevice",
                return_value=SimpleNamespace(InputStream=Mock(return_value=input_context)),
            ),
            patch.object(
                wake_conversation_runtime,
                "wait_for_wake",
                return_value=("Hey Bunnelby", 0.2),
            ) as wait_for_wake,
            patch.object(
                wake_conversation_runtime,
                "capture_conversation_turn",
                side_effect=[first, follow_up],
            ) as capture,
            patch.object(wake_conversation_runtime, "transcribe_samples", return_value=transcript),
            patch.object(wake_conversation_runtime, "dispatch_to_chat", return_value=response) as chat,
            patch.object(wake_conversation_runtime, "request_tts", return_value=b"RIFF" + b"0" * 100),
            patch.object(wake_conversation_runtime, "SoundDeviceWavPlayer", return_value=player),
            patch.object(
                wake_conversation_runtime,
                "_monitor_playback_for_barge_in",
                side_effect=[(None, playback), (None, playback)],
            ),
            patch.object(
                wake_conversation_runtime,
                "stt_runtime_profile",
                return_value=SimpleNamespace(
                    model="small",
                    device="cpu",
                    compute_type="int8",
                    beam_size=5,
                    hotwords=None,
                ),
            ),
            patch("sys.stdout", output),
        ):
            exit_code = wake_conversation_runtime.run(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(wait_for_wake.call_count, 1)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(capture.call_count, 2)
        follow_up_wait = capture.call_args_list[1].kwargs["wait_seconds"]
        self.assertGreater(follow_up_wait, 9.9)
        self.assertLessEqual(follow_up_wait, 10.0)
        self.assertIn("Follow-up accepted without wake phrase.", output.getvalue())
        self.assertIn("Completed conversation turns: 2", output.getvalue())

    def test_stt_failure_returns_to_standby_and_next_wake_succeeds(self) -> None:
        args = wake_conversation_runtime.parse_args(["--turns", "1", "--no-tts"])
        now = time.monotonic()
        audio = wake_conversation_runtime.CapturedUtterance(
            np.zeros(16_000, dtype=np.float32), now, now + 1.0
        )
        transcript = TranscriptionResult("check calendar", "en", 1.0, 1.0)
        response = {"reply": "Ready.", "spoken_reply": "Ready.", "spoken_language": "en"}
        _input_stream, input_context = self._runtime_dependencies()
        output = io.StringIO()

        with (
            patch.object(wake_conversation_runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(wake_conversation_runtime, "load_wake_asr", return_value=Mock()),
            patch.object(wake_conversation_runtime, "default_microphone", return_value=(1, {"name": "test mic"})),
            patch.object(
                wake_conversation_runtime,
                "_sounddevice",
                return_value=SimpleNamespace(InputStream=Mock(return_value=input_context)),
            ),
            patch.object(wake_conversation_runtime, "wait_for_wake", side_effect=[("Hey Bunnelby", 0.2), ("Hey Bunnelby", 0.2)]) as wake,
            patch.object(wake_conversation_runtime, "capture_conversation_turn", side_effect=[audio, audio]),
            patch.object(wake_conversation_runtime, "transcribe_samples", side_effect=[STTTranscriptionError("forced"), transcript]),
            patch.object(wake_conversation_runtime, "dispatch_to_chat", return_value=response),
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5, hotwords=None)),
            patch("sys.stdout", output),
        ):
            exit_code = wake_conversation_runtime.run(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(wake.call_count, 2)
        self.assertIn("Recovered safely: conversation STT failed", output.getvalue())
        self.assertIn("Completed conversation turns: 1", output.getvalue())

    def test_chat_failure_returns_to_standby_and_next_wake_succeeds(self) -> None:
        args = wake_conversation_runtime.parse_args(["--turns", "2", "--no-tts"])
        now = time.monotonic()
        audio = wake_conversation_runtime.CapturedUtterance(
            np.zeros(16_000, dtype=np.float32), now, now + 1.0
        )
        transcript = TranscriptionResult("check calendar", "en", 1.0, 1.0)
        response = {"reply": "Ready.", "spoken_reply": "Ready.", "spoken_language": "en"}
        _input_stream, input_context = self._runtime_dependencies()
        output = io.StringIO()

        with (
            patch.object(wake_conversation_runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(wake_conversation_runtime, "load_wake_asr", return_value=Mock()),
            patch.object(wake_conversation_runtime, "default_microphone", return_value=(1, {"name": "test mic"})),
            patch.object(
                wake_conversation_runtime,
                "_sounddevice",
                return_value=SimpleNamespace(InputStream=Mock(return_value=input_context)),
            ),
            patch.object(wake_conversation_runtime, "wait_for_wake", side_effect=[("Hey Bunnelby", 0.2), ("Hey Bunnelby", 0.2)]) as wake,
            patch.object(wake_conversation_runtime, "capture_conversation_turn", side_effect=[audio, audio]),
            patch.object(wake_conversation_runtime, "transcribe_samples", return_value=transcript),
            patch.object(wake_conversation_runtime, "dispatch_to_chat", side_effect=[RuntimeError("offline"), response]),
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5, hotwords=None)),
            patch("sys.stdout", output),
        ):
            exit_code = wake_conversation_runtime.run(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(wake.call_count, 2)
        self.assertIn("Recovered safely: chat failed; transcript preserved", output.getvalue())
        self.assertIn("Chat failures: 1", output.getvalue())

    def test_input_device_failure_returns_nonzero(self) -> None:
        args = wake_conversation_runtime.parse_args(["--turns", "1"])
        output = io.StringIO()

        with (
            patch.object(wake_conversation_runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(wake_conversation_runtime, "load_wake_asr", return_value=Mock()),
            patch.object(wake_conversation_runtime, "default_microphone", return_value=(1, {"name": "test mic"})),
            patch.object(
                wake_conversation_runtime,
                "_sounddevice",
                return_value=SimpleNamespace(
                    InputStream=Mock(side_effect=OSError("device unavailable"))
                ),
            ),
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5, hotwords=None)),
            patch("sys.stdout", output),
        ):
            exit_code = wake_conversation_runtime.run(args)

        self.assertEqual(exit_code, 1)
        self.assertIn("unrecoverable device/runtime error", output.getvalue())
        self.assertIn("Fatal runtime failures: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
