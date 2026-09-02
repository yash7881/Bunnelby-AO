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
        for phrase in (
            "Bunnelby",
            "hello Bunnelby",
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
                "--barge-in-echo-threshold",
                "2",
            ]
        )
        self.assertEqual(clamped.follow_up_seconds, 60.0)
        self.assertEqual(clamped.conversation_silence, 0.5)
        self.assertEqual(clamped.max_utterance, 120.0)
        self.assertEqual(clamped.barge_in_echo_threshold, 0.95)

    def test_tts_url_tracks_custom_chat_base(self) -> None:
        args = wake_conversation_runtime.parse_args(
            ["--api-url", "http://127.0.0.1:9000/chat"]
        )
        self.assertEqual(args.tts_url, "http://127.0.0.1:9000/tts")

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
                    model="small", device="cpu", compute_type="int8", beam_size=5
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
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5)),
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
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5)),
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
            patch.object(wake_conversation_runtime, "stt_runtime_profile", return_value=SimpleNamespace(model="small", device="cpu", compute_type="int8", beam_size=5)),
            patch("sys.stdout", output),
        ):
            exit_code = wake_conversation_runtime.run(args)

        self.assertEqual(exit_code, 1)
        self.assertIn("unrecoverable device/runtime error", output.getvalue())
        self.assertIn("Fatal runtime failures: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
