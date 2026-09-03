"""Barge-in regression suite: interrupting Bunnelby while it is speaking.

Everything here is CI-safe. No microphone, no speaker, no model download, no network:
the Silero VAD, the microphone stream, the TTS playback handle, the monotonic clock, and
the downstream utterance capture are all replaced with deterministic fakes at the same
module-attribute boundaries the existing persistent-runtime tests patch.
"""

from __future__ import annotations

import io
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.wakeword import wake_conversation_runtime as runtime
from services.api.app.audio_playback import (
    PlaybackHandle,
    PlaybackResult,
    PlaybackStatus,
    decode_pcm_wav,
)
from services.api.app.stt_service import TranscriptionResult
from services.api.app.voice_session import VoiceState


FRAME = runtime.READ_SAMPLES
FRAME_SECONDS = FRAME / runtime.SAMPLE_RATE


def voiced_frame(amplitude: float, seed: int = 0, size: int = FRAME) -> np.ndarray:
    """A deterministic broadband frame at a known RMS, standing in for speech."""
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal(size)
    raw -= raw.mean()
    raw /= np.sqrt(np.mean(np.square(raw)))
    return (raw * amplitude).astype(np.float32)


def silent_frame(size: int = FRAME) -> np.ndarray:
    return np.zeros(size, dtype=np.float32)


def make_wav(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class StepClock:
    """Monotonic clock that advances one microphone frame per microphone read."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = FRAME_SECONDS) -> None:
        self.value += seconds


class FakeVad:
    """Stand-in for the sherpa-onnx conversation VAD (no ONNX model required)."""

    def __init__(self, speech: bool = True) -> None:
        self.speech = speech
        self.accepted = 0

    def accept_waveform(self, frame) -> None:
        self.accepted += 1

    def is_speech_detected(self) -> bool:
        return self.speech

    def empty(self) -> bool:
        return True

    def pop(self) -> None:  # pragma: no cover - never reached while empty() is True
        raise AssertionError("pop() must not be called on an empty fake VAD")

    def flush(self) -> None:
        return None


class ScriptedInputStream:
    """Replays scripted microphone frames, then silence, advancing the test clock."""

    read_available = 0

    def __init__(
        self,
        frames,
        *,
        clock: StepClock,
        handle=None,
        finish_when_exhausted: bool = False,
        max_reads: int = 4_000,
    ) -> None:
        self.queue = [np.asarray(frame, dtype=np.float32).reshape(-1) for frame in frames]
        self.clock = clock
        self.handle = handle
        self.finish_when_exhausted = finish_when_exhausted
        self.max_reads = max_reads
        self.reads = 0

    def read(self, frames: int):
        self.reads += 1
        if self.reads > self.max_reads:
            # Fail a stuck monitor loop as an assertion instead of hanging the suite.
            raise AssertionError("barge-in monitor did not terminate")
        if self.queue:
            block = self.queue.pop(0)
        else:
            block = silent_frame(frames)
            if self.finish_when_exhausted and self.handle is not None:
                self.handle.done = True
        self.clock.advance(block.size / runtime.SAMPLE_RATE)
        return block.reshape(-1, 1), False


class FakePlaybackHandle:
    """Playback lifecycle stub with a controllable reference level and terminal status."""

    def __init__(
        self,
        *,
        clock: StepClock,
        reference_rms: float = 0.0,
        terminal_status: PlaybackStatus = PlaybackStatus.CANCELLED,
    ) -> None:
        self.clock = clock
        self.reference_rms = reference_rms
        self.terminal_status = terminal_status
        self.done = False
        self.cancel_calls = 0
        self.cancelled_at: float | None = None

    def reference_level(self, frames: int) -> float:
        return self.reference_rms

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.cancelled_at is None:
            self.cancelled_at = self.clock()
        self.done = True

    def cancellation_requested(self) -> bool:
        return self.cancel_calls > 0

    def wait(self, timeout: float | None = None) -> PlaybackResult:
        now = self.clock()
        return PlaybackResult(
            status=self.terminal_status,
            queued_at=now - 2.0,
            started_at=now - 2.0,
            first_audio_at=now - 1.9,
            finished_at=now,
            frames_played=32_000,
        )


class BargeInDetectorTests(unittest.TestCase):
    """Onset detection: what does and does not count as a real interruption."""

    def _detector(self, **overrides) -> runtime.BargeInDetector:
        settings = dict(
            frame_samples=FRAME,
            sample_rate=runtime.SAMPLE_RATE,
            min_speech_seconds=runtime.DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS,
            min_rms=runtime.DEFAULT_BARGE_IN_MIN_RMS,
            echo_margin=runtime.DEFAULT_BARGE_IN_ECHO_MARGIN,
        )
        settings.update(overrides)
        return runtime.BargeInDetector(**settings)

    def _feed(
        self,
        detector,
        frames,
        *,
        reference_level,
        start=0.0,
        vad_speech=True,
        stop_on_onset=True,
    ):
        """Feed frames the way the monitor does: stop at the first accepted onset."""
        onsets = []
        now = start
        for index, frame in enumerate(frames):
            onset = detector.observe(
                frame,
                reference_level=reference_level,
                vad_speech=vad_speech,
                now=now,
                armed=True,
            )
            if onset is not None:
                onsets.append((index, onset))
                if stop_on_onset:
                    break
            now += FRAME_SECONDS
        return onsets

    def test_self_echo_at_speaker_level_never_triggers_barge_in(self) -> None:
        """False-trigger guard: Bunnelby's own audio must not cancel Bunnelby."""
        detector = self._detector()
        reference = 0.20
        coupling = 0.45
        echo = [voiced_frame(reference * coupling, seed=i) for i in range(200)]

        onsets = self._feed(detector, echo, reference_level=reference)

        self.assertEqual(onsets, [])
        self.assertTrue(detector.calibrated)
        self.assertAlmostEqual(detector.coupling, coupling, delta=0.08)

    def test_sustained_user_speech_over_self_echo_is_accepted(self) -> None:
        detector = self._detector()
        reference = 0.20
        coupling = 0.30
        echo = [voiced_frame(reference * coupling, seed=i) for i in range(20)]
        # The user speaks over the reply: much louder than the measured echo level.
        user = [voiced_frame(0.22, seed=500 + i) for i in range(20)]

        onsets = self._feed(detector, echo + user, reference_level=reference)

        self.assertEqual(len(onsets), 1)
        index, onset = onsets[0]
        # Detection starts on the user's first frame, not somewhere later in the word.
        self.assertAlmostEqual(onset.detected_at, len(echo) * FRAME_SECONDS, places=6)
        # Confirmation stays inside the configured window plus one frame of quantization.
        self.assertLessEqual(
            onset.accepted_at - onset.detected_at,
            runtime.DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS + FRAME_SECONDS,
        )
        self.assertGreater(index, len(echo))

    def test_onset_audio_is_preserved_with_pre_roll_and_is_not_clipped(self) -> None:
        """Every frame from before confirmation must survive into the utterance."""
        detector = self._detector()
        reference = 0.0
        user = [voiced_frame(0.20, seed=900 + i) for i in range(12)]

        # Calibrate first so the detector is armed, using reference-driven echo frames.
        self._feed(
            detector,
            [voiced_frame(0.05, seed=i) for i in range(12)],
            reference_level=0.15,
        )
        onsets = self._feed(detector, user, reference_level=reference, start=10.0)

        self.assertEqual(len(onsets), 1)
        index, onset = onsets[0]
        confirmation_frames = index + 1
        captured_frames = onset.samples.size // FRAME

        # The captured audio must contain at least every frame observed since the onset.
        self.assertGreaterEqual(captured_frames, confirmation_frames)
        # And it must literally begin with the user's very first frame, not a later one.
        np.testing.assert_allclose(
            onset.samples[-confirmation_frames * FRAME :][:FRAME],
            user[0],
            atol=1e-6,
        )
        self.assertLess(onset.speech_started_at, 10.0 + FRAME_SECONDS)

    def test_short_interruption_survives_a_mid_word_stop_closure(self) -> None:
        """"Wait" has a silent /t/ closure; one quiet frame must not reset detection."""
        detector = self._detector()
        reference = 0.10
        self._feed(
            detector,
            [voiced_frame(0.03, seed=i) for i in range(12)],
            reference_level=reference,
        )

        word = [
            voiced_frame(0.25, seed=1),
            voiced_frame(0.25, seed=2),
            silent_frame(),  # stop closure
            voiced_frame(0.25, seed=3),
            voiced_frame(0.25, seed=4),
            voiced_frame(0.25, seed=5),
            voiced_frame(0.25, seed=6),
        ]
        onsets = self._feed(detector, word, reference_level=reference, start=5.0)

        self.assertEqual(len(onsets), 1)
        self.assertGreaterEqual(onsets[0][1].samples.size, len(word[:6]) * FRAME)

    def test_detection_is_disarmed_before_grace_and_before_calibration(self) -> None:
        detector = self._detector()
        loud = voiced_frame(0.4, seed=3)

        # Not armed: inside the post-playback-start grace window.
        for step in range(20):
            self.assertIsNone(
                detector.observe(
                    loud,
                    reference_level=0.1,
                    vad_speech=True,
                    now=step * FRAME_SECONDS,
                    armed=False,
                )
            )

        # Armed but not yet calibrated against a silent reference: still no trigger.
        fresh = self._detector()
        for step in range(runtime.BARGE_IN_MIN_CALIBRATION_FRAMES):
            self.assertIsNone(
                fresh.observe(
                    loud,
                    reference_level=0.0,
                    vad_speech=True,
                    now=step * FRAME_SECONDS,
                    armed=True,
                )
            )
        self.assertFalse(fresh.calibrated)

    def test_interrupting_voice_cannot_ratchet_the_echo_estimate_against_itself(
        self,
    ) -> None:
        """Regression guard: learning from user speech would hide the interruption."""
        detector = self._detector()
        reference = 0.20
        self._feed(
            detector,
            [voiced_frame(reference * 0.30, seed=i) for i in range(20)],
            reference_level=reference,
        )
        calibrated = detector.coupling

        # A long interruption that is only moderately louder than the echo. Frames are
        # deliberately fed past the acceptance point to exercise continued adaptation.
        self._feed(
            detector,
            [voiced_frame(0.11, seed=300 + i) for i in range(60)],
            reference_level=reference,
            start=5.0,
            stop_on_onset=False,
        )

        self.assertLessEqual(
            detector.coupling,
            calibrated * runtime.BARGE_IN_COUPLING_DRIFT_CEILING + 1e-9,
        )

    def test_vad_disagreement_blocks_non_speech_energy(self) -> None:
        detector = self._detector()
        self._feed(
            detector,
            [voiced_frame(0.02, seed=i) for i in range(12)],
            reference_level=0.10,
        )
        bang = [voiced_frame(0.5, seed=77 + i) for i in range(30)]

        onsets = self._feed(detector, bang, reference_level=0.10, vad_speech=False)

        self.assertEqual(onsets, [])


class BargeInMonitorTests(unittest.TestCase):
    """The SPEAKING-state monitor loop: cancellation, hand-off, and latency."""

    def _args(self, *extra: str):
        return runtime.parse_args(["--turns", "1", *extra])

    def _run_monitor(
        self,
        frames,
        *,
        reference_rms: float,
        terminal_status: PlaybackStatus = PlaybackStatus.CANCELLED,
        finish_when_exhausted: bool = False,
        capture_returns=None,
        args=None,
    ):
        clock = StepClock()
        handle = FakePlaybackHandle(
            clock=clock,
            reference_rms=reference_rms,
            terminal_status=terminal_status,
        )
        stream = ScriptedInputStream(
            frames,
            clock=clock,
            handle=handle,
            finish_when_exhausted=finish_when_exhausted,
        )
        accepted: list[float] = []
        captured = (
            capture_returns
            if capture_returns is not None
            else runtime.CapturedUtterance(np.zeros(8_000, dtype=np.float32), 0.0, 1.0)
        )

        with (
            patch.object(
                runtime,
                "create_conversation_vad",
                return_value=(FakeVad(), FRAME),
            ),
            patch.object(
                runtime,
                "capture_conversation_turn",
                return_value=captured,
            ) as capture,
        ):
            outcome, result = runtime._monitor_playback_for_barge_in(
                stream,
                Mock(name="vad-model-path"),
                args if args is not None else self._args(),
                handle,
                runtime.RuntimeStats(),
                on_accepted=accepted.append,
                clock=clock,
            )
        return SimpleNamespace(
            outcome=outcome,
            result=result,
            handle=handle,
            stream=stream,
            accepted=accepted,
            capture=capture,
            clock=clock,
        )

    def _echo_then_user(self, echo_frames: int = 30, user_frames: int = 20):
        reference = 0.20
        echo = [voiced_frame(reference * 0.35, seed=i) for i in range(echo_frames)]
        user = [voiced_frame(0.25, seed=700 + i) for i in range(user_frames)]
        return reference, echo + user, echo_frames

    def test_playback_cancellation_fires_on_a_simulated_real_barge_in(self) -> None:
        reference, frames, _ = self._echo_then_user()

        run = self._run_monitor(frames, reference_rms=reference)

        self.assertIsNotNone(run.outcome)
        self.assertEqual(run.handle.cancel_calls, 1)
        self.assertTrue(run.handle.cancellation_requested())
        self.assertEqual(len(run.accepted), 1)

    def test_cancellation_is_requested_within_the_three_hundred_millisecond_budget(
        self,
    ) -> None:
        reference, frames, echo_frames = self._echo_then_user()

        run = self._run_monitor(frames, reference_rms=reference)

        assert run.outcome is not None
        self.assertLess(run.outcome.detection_latency_ms, 300.0)
        # The cancel request must also land within 300 ms of the first loud user frame.
        user_started_at = 1_000.0 + echo_frames * FRAME_SECONDS
        assert run.handle.cancelled_at is not None
        self.assertLess((run.handle.cancelled_at - user_started_at) * 1000.0, 300.0)

    def test_full_interrupting_utterance_including_onset_is_handed_to_capture(
        self,
    ) -> None:
        reference, frames, echo_frames = self._echo_then_user()

        run = self._run_monitor(frames, reference_rms=reference)

        kwargs = run.capture.call_args.kwargs
        initial = kwargs["initial_samples"]
        confirmation_seconds = runtime.DEFAULT_BARGE_IN_MIN_SPEECH_SECONDS
        self.assertGreaterEqual(
            initial.size / runtime.SAMPLE_RATE,
            confirmation_seconds,
        )
        # The capture continues from exactly where detection stopped: the last frame the
        # detector consumed is the last frame handed downstream.
        last_consumed = frames[run.stream.reads - 1]
        np.testing.assert_allclose(initial[-FRAME:], last_consumed, atol=1e-6)
        self.assertEqual(
            kwargs["initial_speech_started_at"], run.outcome.speech_started_at
        )
        self.assertLessEqual(
            run.outcome.speech_started_at,
            1_000.0 + echo_frames * FRAME_SECONDS + FRAME_SECONDS,
        )

    def test_first_user_frame_is_not_clipped_from_the_captured_audio(self) -> None:
        reference, frames, echo_frames = self._echo_then_user()
        first_user_frame = frames[echo_frames]

        run = self._run_monitor(frames, reference_rms=reference)

        initial = run.capture.call_args.kwargs["initial_samples"]
        blocks = initial.reshape(-1, FRAME)
        matches = [
            index
            for index, block in enumerate(blocks)
            if np.allclose(block, first_user_frame, atol=1e-6)
        ]
        self.assertEqual(len(matches), 1, "onset frame missing from captured audio")
        self.assertGreater(
            blocks.shape[0] - matches[0],
            1,
            "captured audio must continue past the onset frame",
        )

    def test_self_echo_only_playback_completes_without_any_barge_in(self) -> None:
        reference = 0.20
        echo = [voiced_frame(reference * 0.4, seed=i) for i in range(120)]

        run = self._run_monitor(
            echo,
            reference_rms=reference,
            terminal_status=PlaybackStatus.COMPLETED,
            finish_when_exhausted=True,
        )

        self.assertIsNone(run.outcome)
        self.assertEqual(run.handle.cancel_calls, 0)
        self.assertEqual(run.accepted, [])
        self.assertEqual(run.result.status, PlaybackStatus.COMPLETED)

    def test_monitor_drops_the_microphone_backlog_before_it_starts_listening(
        self,
    ) -> None:
        """Frames buffered during STT/chat/TTS must not be scored against live playback."""
        reference, frames, _ = self._echo_then_user()
        with patch.object(
            runtime, "_discard_buffered_microphone", return_value=0
        ) as discard:
            run = self._run_monitor(frames, reference_rms=reference)

        discard.assert_called_once()
        self.assertIsNotNone(run.outcome)

    def test_disabled_barge_in_monitor_never_reads_the_microphone(self) -> None:
        clock = StepClock()
        handle = FakePlaybackHandle(
            clock=clock, terminal_status=PlaybackStatus.COMPLETED
        )
        stream = ScriptedInputStream([], clock=clock, handle=handle)

        outcome, result = runtime._monitor_playback_for_barge_in(
            stream,
            Mock(),
            self._args("--no-barge-in"),
            handle,
            runtime.RuntimeStats(),
            clock=clock,
        )

        self.assertIsNone(outcome)
        self.assertEqual(stream.reads, 0)
        self.assertEqual(result.status, PlaybackStatus.COMPLETED)


class PlaybackCancellationRaceTests(unittest.TestCase):
    """A terminal result that lands after cancellation must not look like success."""

    def test_completion_racing_a_cancellation_is_published_as_cancelled(self) -> None:
        decoded = decode_pcm_wav(make_wav(np.zeros(1_600, dtype=np.float32)))
        clock = StepClock()
        playback = PlaybackHandle(
            decoded, clock=clock, microphone_reference_rate=16_000
        )
        playback._mark_started()
        playback.cancel()
        # The writer drained its last block before it noticed the cancellation flag.
        playback._finish(PlaybackStatus.COMPLETED)

        result = playback.wait(1.0)
        assert result is not None
        self.assertEqual(result.status, PlaybackStatus.CANCELLED)

    def test_first_terminal_result_wins_and_later_ones_are_ignored(self) -> None:
        decoded = decode_pcm_wav(make_wav(np.zeros(1_600, dtype=np.float32)))
        playback = PlaybackHandle(
            decoded, clock=StepClock(), microphone_reference_rate=16_000
        )
        playback._finish(PlaybackStatus.COMPLETED)
        playback._finish(PlaybackStatus.FAILED, "late device error")

        result = playback.wait(1.0)
        assert result is not None
        self.assertEqual(result.status, PlaybackStatus.COMPLETED)
        self.assertIsNone(result.error)

    def test_reference_level_reports_recent_output_loudness(self) -> None:
        samples = np.full(16_000, 0.5, dtype=np.float32)
        decoded = decode_pcm_wav(make_wav(samples))
        playback = PlaybackHandle(
            decoded, clock=lambda: 0.0, microphone_reference_rate=16_000
        )
        self.assertEqual(playback.reference_level(4_800), 0.0)
        playback._advance(8_000)
        self.assertAlmostEqual(playback.reference_level(4_800), 0.5, places=3)

    def test_reference_level_reports_the_peak_window_not_the_average(self) -> None:
        """A quiet tail must not hide a loud syllable the microphone is still hearing."""
        samples = np.zeros(16_000, dtype=np.float32)
        samples[4_000:5_000] = 0.8  # one loud burst, then silence
        decoded = decode_pcm_wav(make_wav(samples))
        playback = PlaybackHandle(
            decoded, clock=lambda: 0.0, microphone_reference_rate=16_000
        )
        playback._advance(8_000)

        peak = playback.reference_level(4_800, window=512)
        mean = float(np.sqrt(np.mean(np.square(samples[3_200:8_000], dtype=np.float64))))

        self.assertGreater(peak, 0.7)
        self.assertGreater(peak, mean * 2.0)


class BargeInRuntimeStateTests(unittest.TestCase):
    """End-to-end state-machine behaviour of run() around a barge-in."""

    def _session(self, *, monitor, turns: int = 2):
        args = runtime.parse_args(["--turns", str(turns)])
        now = time.monotonic()
        first = runtime.CapturedUtterance(
            np.zeros(16_000, dtype=np.float32), now, now + 1.0
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
            patch.object(runtime, "ensure_silero_vad_model", return_value=Mock()),
            patch.object(runtime, "load_wake_asr", return_value=Mock()),
            patch.object(
                runtime, "default_microphone", return_value=(1, {"name": "test mic"})
            ),
            patch.object(
                runtime,
                "_sounddevice",
                return_value=SimpleNamespace(InputStream=Mock(return_value=input_context)),
            ),
            patch.object(
                runtime, "wait_for_wake", return_value=("Hey Bunnelby", 0.2)
            ) as wake,
            patch.object(
                runtime, "capture_conversation_turn", return_value=first
            ) as capture,
            patch.object(runtime, "transcribe_samples", return_value=transcript),
            patch.object(runtime, "dispatch_to_chat", return_value=response) as chat,
            patch.object(runtime, "request_tts", return_value=b"RIFF" + b"0" * 100),
            patch.object(runtime, "SoundDeviceWavPlayer", return_value=player),
            patch.object(
                runtime, "_monitor_playback_for_barge_in", side_effect=monitor
            ),
            patch.object(
                runtime,
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
            exit_code = runtime.run(args)

        return SimpleNamespace(
            exit_code=exit_code,
            output=output.getvalue(),
            wake=wake,
            chat=chat,
            capture=capture,
        )

    @staticmethod
    def _barge_in_monitor(status: PlaybackStatus):
        """Monitor stub that behaves like a real accepted barge-in on the first turn."""
        calls = {"count": 0}

        def monitor(*_args, on_accepted=None, **_kwargs):
            calls["count"] += 1
            now = time.monotonic()
            completed = PlaybackResult(
                status=status,
                queued_at=now - 3.0,
                started_at=now - 2.0,
                first_audio_at=now - 1.9,
                finished_at=now,
                frames_played=24_000,
            )
            if calls["count"] == 1:
                if on_accepted is not None:
                    on_accepted(now - 0.5)
                interruption = runtime.CapturedUtterance(
                    np.zeros(24_000, dtype=np.float32), now - 0.5, now
                )
                return (
                    runtime.BargeInOutcome(
                        utterance=interruption,
                        speech_started_at=now - 0.5,
                        cancellation_latency_ms=42.0,
                        detection_latency_ms=210.0,
                        coupling=0.4,
                    ),
                    completed,
                )
            return (
                None,
                PlaybackResult(
                    status=PlaybackStatus.COMPLETED,
                    queued_at=now - 3.0,
                    started_at=now - 2.0,
                    first_audio_at=now - 1.9,
                    finished_at=now,
                    frames_played=24_000,
                ),
            )

        return monitor

    def test_speaking_transitions_to_listening_and_no_wake_word_is_required(
        self,
    ) -> None:
        session = self._session(
            monitor=self._barge_in_monitor(PlaybackStatus.CANCELLED)
        )

        self.assertEqual(session.exit_code, 0)
        self.assertIn("State: LISTENING - barge-in speech began", session.output)
        self.assertIn("Barge-in accepted; TTS playback cancelled.", session.output)
        self.assertIn("Barge-ins: 1", session.output)
        self.assertIn("Completed conversation turns: 2", session.output)
        # One wake phrase for two turns: the interruption needed no wake word.
        self.assertEqual(session.wake.call_count, 1)
        self.assertEqual(session.chat.call_count, 2)
        # The interrupting turn went through the normal Brain pipeline: transcribe, then
        # /chat, without ever passing through the follow-up window.
        after_barge_in = session.output.split("Barge-in accepted")[1]
        self.assertIn("State: TRANSCRIBING", after_barge_in)
        self.assertNotIn("Follow-up listening", after_barge_in)

    def test_late_playback_completion_does_not_overwrite_the_new_listening_state(
        self,
    ) -> None:
        """Race guard: playback reported COMPLETED after the runtime already cancelled."""
        session = self._session(
            monitor=self._barge_in_monitor(PlaybackStatus.COMPLETED)
        )

        self.assertEqual(session.exit_code, 0)
        self.assertIn("State: LISTENING - barge-in speech began", session.output)
        self.assertIn("Barge-ins: 1", session.output)
        self.assertEqual(session.wake.call_count, 1)
        self.assertEqual(session.chat.call_count, 2)
        barge_in_turn = session.output.split("Barge-in accepted")[1]
        self.assertNotIn(
            "TTS playback completed; fresh follow-up window started.",
            session.output.split("Barge-in accepted")[0],
        )
        self.assertIn("State: TRANSCRIBING", barge_in_turn)

    def test_normal_follow_up_window_is_unaffected_when_nobody_interrupts(self) -> None:
        def monitor(*_args, on_accepted=None, **_kwargs):
            now = time.monotonic()
            return (
                None,
                PlaybackResult(
                    status=PlaybackStatus.COMPLETED,
                    queued_at=now - 3.0,
                    started_at=now - 2.0,
                    first_audio_at=now - 1.9,
                    finished_at=now,
                    frames_played=24_000,
                ),
            )

        session = self._session(monitor=monitor)

        self.assertEqual(session.exit_code, 0)
        self.assertIn(
            "TTS playback completed; fresh follow-up window started.", session.output
        )
        self.assertIn("Follow-up accepted without wake phrase.", session.output)
        self.assertIn("Barge-ins: 0", session.output)
        self.assertEqual(session.wake.call_count, 1)
        follow_up_wait = session.capture.call_args_list[1].kwargs["wait_seconds"]
        self.assertGreater(follow_up_wait, 9.9)
        self.assertLessEqual(follow_up_wait, 10.0)


class BargeInStateMachineTests(unittest.TestCase):
    def test_speaking_to_listening_is_the_only_barge_in_transition(self) -> None:
        from services.api.app.voice_session import VoiceSessionController

        controller = VoiceSessionController()
        controller.wake_detected()
        controller.begin_listening()
        controller.utterance_completed()
        controller.transcription_completed()
        controller.speaking_started()

        controller.barge_in(speech_started_at=42.0)

        self.assertEqual(controller.state, VoiceState.LISTENING)
        self.assertIsNone(controller.follow_up_deadline)
        self.assertEqual(controller.history[-1].reason, "barge-in speech began")


if __name__ == "__main__":
    unittest.main()
