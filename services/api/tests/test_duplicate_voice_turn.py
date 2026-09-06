"""Regression tests for the duplicate-answer / overlapping-TTS bug.

Observed: one spoken question produced TWO different assistant answers whose
audio overlapped. Two independent defects combined:

  1. `SoundDeviceWavPlayer.start()` launched a new thread and OutputStream
     without cancelling the one already playing, and the player held no
     reference to a "current" playback. Two turns therefore opened two output
     streams on the same device and spoke over each other.

  2. The run loop had no correlation id, so a second answer could not be
     attributed to its origin from the log.

A capture-identity idempotency guard was tried and WITHDRAWN: two real captures
never share a monotonic timestamp, so it could not fire in production, it did
not catch the duplicate that was actually reproduced (two DIFFERENT captures
from the follow-up window), and it broke a legitimate follow-up test that
reuses one capture object. These tests pin what is actually true.

They deliberately do NOT touch the wake path.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import unittest

from scripts.wakeword import wake_conversation_runtime as runtime
from services.api.app.audio_playback import (
    PlaybackResult,
    PlaybackStatus,
    SoundDeviceWavPlayer,
)
from services.api.app.stt_service import TranscriptionResult


def make_wav(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    samples = (np.sin(np.linspace(0, 200 * np.pi, int(rate * seconds))) * 0.2).astype("<f4")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


class CountingStreamFactory:
    """Counts how many OutputStreams are open at the same instant."""

    def __init__(self, write_delay: float = 0.02) -> None:
        self.lock = threading.Lock()
        self.live: list[object] = []
        self.peak = 0
        self.write_delay = write_delay

    def __call__(self, **_kwargs):
        factory = self

        class Stream:
            def start(self) -> None:
                with factory.lock:
                    factory.live.append(self)
                    factory.peak = max(factory.peak, len(factory.live))

            def write(self, _block) -> None:
                time.sleep(factory.write_delay)

            def stop(self) -> None:
                with factory.lock:
                    if self in factory.live:
                        factory.live.remove(self)

            def abort(self) -> None:
                self.stop()

            def close(self) -> None:
                self.stop()

        return Stream()


class SinglePlaybackOwnerTests(unittest.TestCase):
    """INVARIANT 3 & 4: one assistant turn, one playback owner, one stream."""

    def test_a_second_start_cancels_the_playback_already_running(self) -> None:
        factory = CountingStreamFactory()
        player = SoundDeviceWavPlayer(stream_factory=factory, block_frames=256)
        audio = make_wav(1.0)

        first = player.start(audio)
        self.assertTrue(self._wait_until(lambda: first.status is PlaybackStatus.STARTED))

        second = player.start(audio)
        try:
            self.assertEqual(
                first.status,
                PlaybackStatus.CANCELLED,
                "the superseded turn must stop, not keep speaking",
            )
            self.assertTrue(self._wait_until(lambda: second.status is PlaybackStatus.STARTED))
            self.assertEqual(
                factory.peak, 1, "two output streams must never be open at once"
            )
        finally:
            second.cancel()
            second.wait(2.0)

    def test_many_rapid_starts_never_overlap(self) -> None:
        factory = CountingStreamFactory(write_delay=0.005)
        player = SoundDeviceWavPlayer(stream_factory=factory, block_frames=256)
        audio = make_wav(0.5)

        handles = []
        for _ in range(5):
            handles.append(player.start(audio))
        try:
            self.assertEqual(factory.peak, 1, "playback ownership must hold under bursts")
        finally:
            for handle in handles:
                handle.cancel()
                handle.wait(1.0)

    def test_the_superseded_playback_is_reaped_before_the_new_one_starts(self) -> None:
        """cancel() only sets a flag; returning before the old thread released
        the device would still allow two streams to be open."""
        factory = CountingStreamFactory(write_delay=0.03)
        player = SoundDeviceWavPlayer(stream_factory=factory, block_frames=256)
        audio = make_wav(1.0)

        first = player.start(audio)
        self.assertTrue(self._wait_until(lambda: first.status is PlaybackStatus.STARTED))
        second = player.start(audio)
        try:
            with factory.lock:
                self.assertLessEqual(len(factory.live), 1)
        finally:
            second.cancel()
            second.wait(2.0)

    def test_a_malformed_payload_does_not_silence_live_playback(self) -> None:
        """Decoding happens before cancellation, so a bad TTS response cannot
        cut off an answer that is legitimately still being spoken."""
        factory = CountingStreamFactory()
        player = SoundDeviceWavPlayer(stream_factory=factory, block_frames=256)
        first = player.start(make_wav(1.0))
        self.assertTrue(self._wait_until(lambda: first.status is PlaybackStatus.STARTED))
        try:
            with self.assertRaises(Exception):
                player.start(b"not a wav")
            self.assertEqual(first.status, PlaybackStatus.STARTED)
        finally:
            first.cancel()
            first.wait(2.0)

    def test_stop_active_is_idempotent(self) -> None:
        player = SoundDeviceWavPlayer(stream_factory=CountingStreamFactory())
        player.stop_active()
        player.stop_active()

    @staticmethod
    def _wait_until(predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return False


def _playback_result(status: PlaybackStatus = PlaybackStatus.COMPLETED) -> PlaybackResult:
    now = time.monotonic()
    return PlaybackResult(
        status=status,
        queued_at=now,
        started_at=now,
        first_audio_at=now,
        finished_at=now,
        frames_played=16_000,
    )


class _FakeHandle:
    status = PlaybackStatus.STARTED

    def wait(self, _timeout=None):
        return _playback_result()

    def cancel(self) -> None:
        pass


class OneUtteranceOneDispatchTests(unittest.TestCase):
    """INVARIANT 1 & 2: one finalized utterance -> exactly one /chat call."""

    def _run(self, captures, transcripts):
        """Drive the real run loop with scripted captures and transcripts."""
        # A 1s follow-up window keeps these tests fast: capture is mocked, so
        # an unfilled follow-up window is polled rather than blocked on.
        args = runtime.parse_args(
            ["--turns", "3", "--language", "en", "--follow-up-seconds", "1"]
        )
        stream = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=stream)
        context.__exit__ = Mock(return_value=False)

        chats: list[str] = []
        playbacks: list[int] = []

        def chat(_url, text, _timeout, **_kwargs):
            chats.append(text)
            return {
                "reply": f"answer {len(chats)}",
                "spoken_reply": f"answer {len(chats)}",
                "spoken_language": "en",
            }

        capture_iter = iter(captures)
        transcript_iter = iter(transcripts)
        output = io.StringIO()

        with (
            patch.object(runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(runtime, "load_wake_asr", return_value=Mock()),
            patch.object(runtime, "default_microphone", return_value=(1, {"name": "mic"})),
            patch.object(
                runtime,
                "_sounddevice",
                return_value=SimpleNamespace(InputStream=Mock(return_value=context)),
            ),
            patch.object(
                runtime,
                "wait_for_wake",
                side_effect=[("Hey Bunnelby", 0.2), KeyboardInterrupt()],
            ),
            patch.object(
                runtime,
                "capture_conversation_turn",
                side_effect=lambda *a, **k: next(capture_iter, None),
            ),
            patch.object(
                runtime,
                "_transcribe_conversation",
                side_effect=lambda *a, **k: next(transcript_iter),
            ),
            patch.object(runtime, "dispatch_to_chat", side_effect=chat),
            patch.object(runtime, "request_tts", return_value=b"wav"),
            patch.object(
                runtime,
                "_monitor_playback_for_barge_in",
                return_value=(None, _playback_result()),
            ),
            patch.object(
                runtime,
                "stt_runtime_profile",
                return_value=SimpleNamespace(
                    model="s", device="cpu", compute_type="int8", beam_size=5, hotwords=None
                ),
            ),
            patch.object(
                runtime.SoundDeviceWavPlayer,
                "start",
                lambda _self, _wav: (playbacks.append(1), _FakeHandle())[1],
            ),
            patch("sys.stdout", output),
        ):
            runtime.run(args)

        return chats, playbacks, output.getvalue()

    @staticmethod
    def _utterance(offset: float = 0.0, size: int = 16_000):
        now = time.monotonic() + offset
        return runtime.CapturedUtterance(
            np.zeros(size, dtype=np.float32), now, now + 1.0
        )

    def test_one_utterance_produces_exactly_one_chat_dispatch(self) -> None:
        """TEST 1: the baseline invariant."""
        chats, playbacks, text = self._run(
            captures=[self._utterance()],
            transcripts=[TranscriptionResult("what is a vector database", "en", 1.0, 1.0)],
        )
        self.assertEqual(len(chats), 1, "exactly one /chat call for one utterance")
        self.assertEqual(len(playbacks), 1, "exactly one playback start")
        self.assertEqual(text.count("dispatch_begin"), 1)

    def test_a_genuinely_different_second_turn_still_works(self) -> None:
        """TEST 5: follow-up conversation must keep working normally."""
        chats, playbacks, _text = self._run(
            captures=[self._utterance(), self._utterance(size=20_000)],
            transcripts=[
                TranscriptionResult("what is a vector database", "en", 1.0, 1.0),
                TranscriptionResult("and what about a graph database", "en", 1.0, 1.0),
            ],
        )
        self.assertEqual(len(chats), 2, "a real follow-up is still a real turn")
        self.assertEqual(
            chats,
            ["what is a vector database", "and what about a graph database"],
        )
        self.assertEqual(len(playbacks), 2)

    def test_every_turn_is_correlated_end_to_end(self) -> None:
        """The log must attribute each stage to one turn id and one owner, so a
        future duplicate can be traced instead of guessed at."""
        _chats, _playbacks, text = self._run(
            captures=[self._utterance()],
            transcripts=[TranscriptionResult("what is ai", "en", 1.0, 1.0)],
        )
        for marker in (
            "turn=1 transcript_final",
            "turn=1 dispatch_begin source=persistent_runtime",
            "turn=1 chat_begin",
            "turn=1 chat_end",
            "turn=1 tts_request_begin",
            "turn=1 playback_begin owner=persistent_runtime",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_only_one_component_owns_voice_playback(self) -> None:
        """INVARIANT 3: the renderer must not also speak a voice turn. It calls
        queueSpeech only for typed input and approval decisions."""
        import pathlib

        app = pathlib.Path(__file__).resolve().parents[3] / "apps/desktop/src/App.jsx"
        source = app.read_text(encoding="utf-8")
        assistant_handler = source.split("eventType === 'assistant_response'", 1)[1][:1200]
        self.assertNotIn(
            "queueSpeech",
            assistant_handler,
            "the voice assistant_response handler must never start renderer TTS",
        )
        self.assertNotIn(
            "fetch(API_URL",
            assistant_handler,
            "the renderer must never re-dispatch a voice turn",
        )


if __name__ == "__main__":
    unittest.main()
