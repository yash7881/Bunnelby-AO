from __future__ import annotations

import io
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class AudioPlaybackError(RuntimeError):
    """Raised when synthesized WAV audio cannot be decoded or played safely."""


class PlaybackStatus(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class DecodedWav:
    samples: np.ndarray
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class PlaybackResult:
    status: PlaybackStatus
    queued_at: float
    started_at: float | None
    first_audio_at: float | None
    finished_at: float
    frames_played: int
    error: str | None = None

    @property
    def playback_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (self.finished_at - self.started_at) * 1000.0)


def decode_pcm_wav(wav_bytes: bytes) -> DecodedWav:
    """Decode an uncompressed PCM WAV into float32 without writing a temp file."""
    if not isinstance(wav_bytes, (bytes, bytearray)) or len(wav_bytes) <= 44:
        raise AudioPlaybackError("TTS returned an empty or invalid WAV payload.")

    try:
        with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as wav_file:
            channels = int(wav_file.getnchannels())
            sample_rate = int(wav_file.getframerate())
            sample_width = int(wav_file.getsampwidth())
            declared_frame_count = int(wav_file.getnframes())
            compression = wav_file.getcomptype()
            payload = wav_file.readframes(declared_frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioPlaybackError("TTS returned malformed WAV audio.") from exc

    if compression != "NONE":
        raise AudioPlaybackError("Only uncompressed PCM WAV playback is supported.")
    if channels < 1 or channels > 2 or sample_rate < 8_000 or sample_rate > 96_000:
        raise AudioPlaybackError("TTS WAV channel count or sample rate is unsupported.")
    bytes_per_frame = sample_width * channels
    if not payload or bytes_per_frame <= 0 or len(payload) % bytes_per_frame:
        raise AudioPlaybackError("TTS WAV contains no audio frames.")
    # FFmpeg writes 0xFFFFFFFF for RIFF/data sizes when WAV is streamed to stdout because
    # it cannot seek back and finalize the header. Python's wave module then reports a huge
    # declared frame count. The received byte payload is authoritative and already bounded
    # by the local TTS endpoint/request timeout.
    frame_count = len(payload) // bytes_per_frame

    if sample_width == 1:
        samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioPlaybackError("TTS WAV sample width is unsupported.")

    if samples.size != frame_count * channels:
        raise AudioPlaybackError("TTS WAV frame data is incomplete.")
    shaped = np.ascontiguousarray(samples.reshape(frame_count, channels), dtype=np.float32)
    return DecodedWav(shaped, sample_rate, channels)


def _mono_resampled(decoded: DecodedWav, target_rate: int) -> np.ndarray:
    mono = decoded.samples.mean(axis=1, dtype=np.float32)
    if decoded.sample_rate == target_rate:
        return np.ascontiguousarray(mono, dtype=np.float32)
    target_size = max(1, int(round(mono.size * target_rate / decoded.sample_rate)))
    source_positions = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


class PlaybackHandle:
    """Thread-safe lifecycle and cancellation handle for one WAV playback."""

    def __init__(
        self,
        decoded: DecodedWav,
        *,
        clock: Callable[[], float],
        microphone_reference_rate: int,
    ) -> None:
        self.decoded = decoded
        self._clock = clock
        self._reference_rate = int(microphone_reference_rate)
        self._reference = _mono_resampled(decoded, self._reference_rate)
        self._queued_at = clock()
        self._started_at: float | None = None
        self._first_audio_at: float | None = None
        self._finished_at: float | None = None
        self._frames_played = 0
        self._status = PlaybackStatus.QUEUED
        self._error: str | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> PlaybackStatus:
        with self._lock:
            return self._status

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def _launch(self, target: Callable[[PlaybackHandle], None]) -> None:
        self._thread = threading.Thread(
            target=target,
            args=(self,),
            name="bunnelby-tts-playback",
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def cancellation_requested(self) -> bool:
        return self._cancel.is_set()

    def _mark_started(self) -> None:
        with self._lock:
            self._started_at = self._clock()
            self._status = PlaybackStatus.STARTED

    def _mark_first_audio(self) -> None:
        with self._lock:
            if self._first_audio_at is None:
                self._first_audio_at = self._clock()

    def _advance(self, output_frames: int) -> None:
        with self._lock:
            self._frames_played += int(output_frames)

    def _finish(self, status: PlaybackStatus, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error = error
            self._finished_at = self._clock()
        self._done.set()

    def wait(self, timeout: float | None = None) -> PlaybackResult | None:
        if not self._done.wait(timeout):
            return None
        with self._lock:
            return PlaybackResult(
                status=self._status,
                queued_at=self._queued_at,
                started_at=self._started_at,
                first_audio_at=self._first_audio_at,
                finished_at=self._finished_at or self._clock(),
                frames_played=self._frames_played,
                error=self._error,
            )

    def echo_score(self, microphone_samples: np.ndarray) -> float:
        """Return the best normalized match to recently played output.

        This is a conservative self-echo gate, not acoustic echo cancellation. It prevents
        obvious speaker feedback from being treated as barge-in while retaining an explicit
        live-device acceptance gate for overlapping user speech.
        """
        microphone = np.asarray(microphone_samples, dtype=np.float32).reshape(-1)
        if microphone.size < 32:
            return 1.0
        microphone = microphone - float(microphone.mean())
        mic_norm = float(np.linalg.norm(microphone))
        if mic_norm < 1e-4:
            return 1.0

        with self._lock:
            reference_cursor = int(
                round(self._frames_played * self._reference_rate / self.decoded.sample_rate)
            )
        if reference_cursor <= microphone.size:
            return 1.0

        best = 0.0
        min_lag = int(0.02 * self._reference_rate)
        max_lag = int(0.50 * self._reference_rate)
        step = max(32, microphone.size // 8)
        for lag in range(min_lag, max_lag + 1, step):
            end = reference_cursor - lag
            start = end - microphone.size
            if start < 0 or end > self._reference.size:
                continue
            reference = self._reference[start:end]
            reference = reference - float(reference.mean())
            ref_norm = float(np.linalg.norm(reference))
            if ref_norm < 1e-4:
                continue
            score = abs(float(np.dot(microphone, reference) / (mic_norm * ref_norm)))
            best = max(best, score)
        return min(1.0, best)


class SoundDeviceWavPlayer:
    """Cancellable real-time WAV playback through PortAudio/sounddevice."""

    def __init__(
        self,
        *,
        output_device: int | None = None,
        block_frames: int = 1024,
        clock: Callable[[], float] = time.monotonic,
        stream_factory: Callable[..., Any] | None = None,
        microphone_reference_rate: int = 16_000,
    ) -> None:
        self.output_device = output_device
        self.block_frames = max(128, min(int(block_frames), 4096))
        self.clock = clock
        self.stream_factory = stream_factory
        self.microphone_reference_rate = microphone_reference_rate

    def start(self, wav_bytes: bytes) -> PlaybackHandle:
        decoded = decode_pcm_wav(wav_bytes)
        handle = PlaybackHandle(
            decoded,
            clock=self.clock,
            microphone_reference_rate=self.microphone_reference_rate,
        )
        handle._launch(self._play)
        return handle

    def _play(self, handle: PlaybackHandle) -> None:
        stream: Any | None = None
        try:
            factory = self.stream_factory
            if factory is None:
                import sounddevice as sd

                factory = sd.OutputStream
            stream_kwargs = {
                "device": self.output_device,
                "samplerate": handle.decoded.sample_rate,
                "channels": handle.decoded.channels,
                "dtype": "float32",
                "blocksize": self.block_frames,
            }
            try:
                stream = factory(**stream_kwargs, latency="low")
            except Exception:
                # Some Windows audio endpoints reject an explicit low-latency request but
                # work with PortAudio's device default. The second failure is reported by
                # the outer lifecycle handler.
                stream = factory(**stream_kwargs)
            stream.start()
            handle._mark_started()

            samples = handle.decoded.samples
            for offset in range(0, samples.shape[0], self.block_frames):
                if handle.cancellation_requested():
                    stream.abort()
                    handle._finish(PlaybackStatus.CANCELLED)
                    return
                block = samples[offset : offset + self.block_frames]
                handle._mark_first_audio()
                stream.write(block)
                handle._advance(block.shape[0])

            stream.stop()
            handle._finish(PlaybackStatus.COMPLETED)
        except Exception as exc:
            if stream is not None:
                try:
                    stream.abort()
                except Exception:
                    pass
            handle._finish(PlaybackStatus.FAILED, str(exc))
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
