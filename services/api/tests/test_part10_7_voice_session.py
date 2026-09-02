from __future__ import annotations

import io
import threading
import unittest
import wave

import numpy as np

from services.api.app.audio_playback import (
    AudioPlaybackError,
    PlaybackHandle,
    PlaybackStatus,
    SoundDeviceWavPlayer,
    decode_pcm_wav,
)
from services.api.app.voice_session import (
    InvalidVoiceTransition,
    VoiceSessionController,
    VoiceState,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_wav(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    payload = np.clip(samples, -1.0, 1.0)
    pcm = (payload * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class VoiceSessionStateTests(unittest.TestCase):
    def _to_thinking(self, controller: VoiceSessionController) -> None:
        controller.wake_detected()
        controller.begin_listening()
        controller.utterance_completed()
        controller.transcription_completed()

    def test_follow_up_starts_only_when_playback_completes(self) -> None:
        clock = FakeClock()
        controller = VoiceSessionController(follow_up_seconds=10.0, clock=clock)
        self._to_thinking(controller)
        controller.speaking_started()
        clock.advance(25.0)
        self.assertIsNone(controller.follow_up_deadline)

        controller.playback_completed()
        self.assertEqual(controller.state, VoiceState.FOLLOW_UP)
        self.assertEqual(controller.follow_up_deadline, 135.0)

    def test_speech_start_at_9_point_8_seconds_is_accepted(self) -> None:
        clock = FakeClock()
        controller = VoiceSessionController(follow_up_seconds=10.0, clock=clock)
        self._to_thinking(controller)
        controller.speaking_started()
        controller.playback_completed(at=200.0)

        self.assertTrue(controller.accept_follow_up_speech(209.8))
        self.assertEqual(controller.state, VoiceState.LISTENING)
        self.assertIsNone(controller.follow_up_deadline)

    def test_speech_after_deadline_returns_to_standby(self) -> None:
        controller = VoiceSessionController(follow_up_seconds=10.0)
        self._to_thinking(controller)
        controller.speaking_started()
        controller.playback_completed(at=200.0)

        self.assertFalse(controller.accept_follow_up_speech(210.01))
        self.assertEqual(controller.state, VoiceState.STANDBY)

    def test_timer_resets_after_every_spoken_answer(self) -> None:
        controller = VoiceSessionController(follow_up_seconds=10.0)
        self._to_thinking(controller)
        controller.speaking_started()
        controller.playback_completed(at=10.0)
        controller.accept_follow_up_speech(15.0)
        controller.utterance_completed(at=16.0)
        controller.transcription_completed(at=17.0)
        controller.speaking_started(at=18.0)
        controller.playback_completed(at=30.0)

        self.assertEqual(controller.follow_up_deadline, 40.0)

    def test_tts_failure_uses_failure_time_as_documented_fallback(self) -> None:
        controller = VoiceSessionController(follow_up_seconds=10.0)
        self._to_thinking(controller)
        controller.playback_failed(at=50.0)
        self.assertEqual(controller.follow_up_deadline, 60.0)

    def test_barge_in_moves_speaking_directly_to_listening(self) -> None:
        controller = VoiceSessionController()
        self._to_thinking(controller)
        controller.speaking_started()
        controller.barge_in(speech_started_at=123.0)
        self.assertEqual(controller.state, VoiceState.LISTENING)

    def test_recovery_never_leaves_runtime_in_error_state(self) -> None:
        controller = VoiceSessionController()
        controller.wake_detected()
        controller.begin_listening()
        controller.utterance_completed()
        controller.recover("forced STT failure", at=10.0)
        self.assertEqual(controller.state, VoiceState.STANDBY)
        self.assertEqual(controller.history[-2].current, VoiceState.ERROR_RECOVERY)

    def test_invalid_transition_fails_closed(self) -> None:
        controller = VoiceSessionController()
        with self.assertRaises(InvalidVoiceTransition):
            controller.speaking_started()


class FakeOutputStream:
    def __init__(self, **_kwargs) -> None:
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.frames = 0

    def start(self) -> None:
        self.started = True

    def write(self, block: np.ndarray) -> None:
        self.frames += block.shape[0]

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class BlockingOutputStream(FakeOutputStream):
    def __init__(self, started: threading.Event, release: threading.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self.write_started = started
        self.release = release

    def write(self, block: np.ndarray) -> None:
        self.write_started.set()
        self.release.wait(2.0)
        super().write(block)


class AudioPlaybackTests(unittest.TestCase):
    def test_ffmpeg_streaming_length_header_uses_actual_received_frames(self) -> None:
        source = bytearray(make_wav(np.zeros(800, dtype=np.float32)))
        source[4:8] = b"\xff\xff\xff\xff"
        source[40:44] = b"\xff\xff\xff\xff"

        decoded = decode_pcm_wav(bytes(source))

        self.assertEqual(decoded.samples.shape, (800, 1))

    def test_real_lifecycle_reaches_completed_after_stream_stop(self) -> None:
        streams: list[FakeOutputStream] = []

        def factory(**kwargs):
            stream = FakeOutputStream(**kwargs)
            streams.append(stream)
            return stream

        samples = np.sin(np.linspace(0, np.pi * 2, 800, dtype=np.float32)) * 0.1
        handle = SoundDeviceWavPlayer(stream_factory=factory, block_frames=128).start(
            make_wav(samples)
        )
        result = handle.wait(2.0)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, PlaybackStatus.COMPLETED)
        self.assertIsNotNone(result.started_at)
        self.assertIsNotNone(result.first_audio_at)
        self.assertTrue(streams[0].stopped)
        self.assertTrue(streams[0].closed)
        self.assertEqual(result.frames_played, 800)

    def test_playback_is_cancellable(self) -> None:
        write_started = threading.Event()
        release = threading.Event()
        streams: list[BlockingOutputStream] = []

        def factory(**kwargs):
            stream = BlockingOutputStream(write_started, release, **kwargs)
            streams.append(stream)
            return stream

        handle = SoundDeviceWavPlayer(stream_factory=factory, block_frames=128).start(
            make_wav(np.zeros(1600, dtype=np.float32))
        )
        self.assertTrue(write_started.wait(1.0))
        handle.cancel()
        release.set()
        result = handle.wait(2.0)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, PlaybackStatus.CANCELLED)
        self.assertTrue(streams[0].aborted)

    def test_invalid_wav_fails_before_output_device_is_opened(self) -> None:
        with self.assertRaises(AudioPlaybackError):
            SoundDeviceWavPlayer(stream_factory=FakeOutputStream).start(b"not wav")

    def test_echo_score_recognizes_recent_output_reference(self) -> None:
        samples = np.sin(np.linspace(0, np.pi * 30, 16_000, dtype=np.float32)) * 0.2
        decoded = decode_pcm_wav(make_wav(samples))
        handle = PlaybackHandle(decoded, clock=lambda: 1.0, microphone_reference_rate=16_000)
        handle._advance(8_000)
        microphone = samples[8_000 - 1_600 - 512 : 8_000 - 1_600]
        self.assertGreater(handle.echo_score(microphone), 0.9)


if __name__ == "__main__":
    unittest.main()
