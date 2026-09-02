from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_runtime import configure_windows_cuda_runtime

logger = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "small"
DEFAULT_STT_DEVICE = "cpu"
DEFAULT_STT_COMPUTE_TYPE = "int8"
DEFAULT_STT_CPU_THREADS = 4
DEFAULT_STT_BEAM_SIZE = 5
DEFAULT_STT_MAX_AUDIO_BYTES = 12 * 1024 * 1024
DEFAULT_STT_MAX_SAMPLE_SECONDS = 120.0
STT_SAMPLE_RATE = 16_000
SUPPORTED_LANGUAGE_HINTS = {"auto", "en", "hi"}

_CONTENT_TYPE_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "application/octet-stream": ".bin",
}


class STTServiceError(RuntimeError):
    """Base exception for local speech-to-text failures."""


class STTDisabledError(STTServiceError):
    """Raised when local STT is disabled by configuration."""


class STTUnavailableError(STTServiceError):
    """Raised when faster-whisper or its model cannot be loaded."""


class STTAudioError(STTServiceError):
    """Raised when the supplied audio payload is invalid or unsupported."""


class STTTranscriptionError(STTServiceError):
    """Raised when inference fails after the model has loaded."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_seconds: float


@dataclass(frozen=True)
class STTRuntimeProfile:
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int


_model_lock = threading.Lock()
_model: Any | None = None
_model_signature: tuple[str, str, str, int, str] | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off"}


def stt_enabled() -> bool:
    return _env_bool("STT_ENABLED", True)


def stt_model_name() -> str:
    return os.getenv("STT_MODEL", DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL


def stt_device() -> str:
    return os.getenv("STT_DEVICE", DEFAULT_STT_DEVICE).strip() or DEFAULT_STT_DEVICE


def stt_compute_type() -> str:
    return os.getenv("STT_COMPUTE_TYPE", DEFAULT_STT_COMPUTE_TYPE).strip() or DEFAULT_STT_COMPUTE_TYPE


def stt_cpu_threads() -> int:
    raw = os.getenv("STT_CPU_THREADS", str(DEFAULT_STT_CPU_THREADS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STT_CPU_THREADS
    return max(1, min(value, 16))


def stt_beam_size() -> int:
    raw = os.getenv("STT_BEAM_SIZE", str(DEFAULT_STT_BEAM_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STT_BEAM_SIZE
    return max(1, min(value, 10))


def stt_max_audio_bytes() -> int:
    raw = os.getenv("STT_MAX_AUDIO_BYTES", str(DEFAULT_STT_MAX_AUDIO_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STT_MAX_AUDIO_BYTES
    return max(64 * 1024, min(value, 100 * 1024 * 1024))


def stt_max_sample_seconds() -> float:
    raw = os.getenv("STT_MAX_SAMPLE_SECONDS", str(DEFAULT_STT_MAX_SAMPLE_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_STT_MAX_SAMPLE_SECONDS
    return max(5.0, min(value, 300.0))


def stt_model_root() -> Path:
    configured = os.getenv("STT_MODEL_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "Bunnelby" / "models" / "stt"
    return Path.home() / ".bunnelby" / "models" / "stt"


def _model_config_signature() -> tuple[str, str, str, int, str]:
    return (
        stt_model_name(),
        stt_device(),
        stt_compute_type(),
        stt_cpu_threads(),
        str(stt_model_root()),
    )


def stt_runtime_profile() -> STTRuntimeProfile:
    """Return the effective configuration used for the next model load/inference."""
    return STTRuntimeProfile(
        model=stt_model_name(),
        device=stt_device(),
        compute_type=stt_compute_type(),
        cpu_threads=stt_cpu_threads(),
        beam_size=stt_beam_size(),
    )


def _invalidate_cached_model(failed_model: Any, reason: str) -> bool:
    """Remove a model that suffered a native inference failure.

    CTranslate2 failures can leave the Python object alive while its CUDA/native state is
    unusable. Identity checking prevents one failed call from clearing a newer model that
    another thread has already installed.
    """
    global _model, _model_signature

    invalidated = False
    with _model_lock:
        if _model is failed_model:
            _model = None
            _model_signature = None
            invalidated = True

    if invalidated:
        native_model = getattr(failed_model, "model", None)
        unload = getattr(native_model, "unload_model", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                logger.debug("Could not eagerly unload failed STT native model", exc_info=True)
        logger.warning("Invalidated failed Bunnelby STT model cache: %s", reason)
    return invalidated


def _load_model() -> Any:
    global _model, _model_signature

    signature = _model_config_signature()
    if _model is not None and _model_signature == signature:
        return _model

    with _model_lock:
        if _model is not None and _model_signature == signature:
            return _model

        if stt_device().casefold() == "cuda":
            cuda_configuration = configure_windows_cuda_runtime()
            if cuda_configuration.dll_directories:
                logger.info(
                    "Registered Bunnelby-local CUDA DLL directories: %s",
                    cuda_configuration.dll_directories,
                )
            if cuda_configuration.path_directories:
                logger.info(
                    "Prepended process-local CUDA PATH directories: %s",
                    cuda_configuration.path_directories,
                )

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise STTUnavailableError(
                "faster-whisper is not available in the Bunnelby backend environment."
            ) from exc

        model_root = stt_model_root()
        try:
            model_root.mkdir(parents=True, exist_ok=True)
            model = WhisperModel(
                stt_model_name(),
                device=stt_device(),
                compute_type=stt_compute_type(),
                cpu_threads=stt_cpu_threads(),
                num_workers=1,
                download_root=str(model_root),
            )
        except Exception as exc:
            logger.warning("Bunnelby STT model load failed: %s", exc)
            raise STTUnavailableError(
                "The local speech recognition model could not be loaded."
            ) from exc

        _model = model
        _model_signature = signature
        logger.info(
            "Bunnelby STT ready: model=%s device=%s compute=%s threads=%s",
            stt_model_name(),
            stt_device(),
            stt_compute_type(),
            stt_cpu_threads(),
        )
        return model


def _suffix_for_content_type(content_type: str | None) -> str:
    normalized = (content_type or "application/octet-stream").split(";", 1)[0].strip().casefold()
    return _CONTENT_TYPE_SUFFIXES.get(normalized, ".bin")


def _validate_language_hint(language: str | None) -> str:
    normalized = (language or "auto").strip().casefold()
    if normalized not in SUPPORTED_LANGUAGE_HINTS:
        raise STTAudioError("STT language must be 'auto', 'en', or 'hi'.")
    return normalized


def _normalize_text(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _result_from_transcription(segments: Any, info: Any, language_hint: str) -> TranscriptionResult:
    text = _normalize_text([str(segment.text) for segment in segments])
    detected_language = str(getattr(info, "language", "") or language_hint or "auto")
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return TranscriptionResult(
        text=text,
        language=detected_language,
        language_probability=max(0.0, min(probability, 1.0)),
        duration_seconds=max(0.0, duration),
    )


def _transcribe_source(
    source: Any,
    *,
    language_hint: str,
    vad_filter: bool,
) -> TranscriptionResult:
    model = _load_model()
    try:
        kwargs: dict[str, Any] = {
            "language": None if language_hint == "auto" else language_hint,
            "task": "transcribe",
            "beam_size": stt_beam_size(),
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "vad_filter": vad_filter,
            "word_timestamps": False,
        }
        if vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 350}
        segments, info = model.transcribe(source, **kwargs)
        return _result_from_transcription(segments, info, language_hint)
    except STTServiceError:
        raise
    except Exception as exc:
        logger.warning("Bunnelby STT inference failed: %s", exc)
        _invalidate_cached_model(model, str(exc))
        raise STTTranscriptionError("Local speech recognition failed for this audio.") from exc


def transcribe_samples(
    samples: np.ndarray,
    *,
    sample_rate: int = STT_SAMPLE_RATE,
    language: str | None = "auto",
) -> TranscriptionResult:
    """Transcribe a microphone utterance directly from RAM.

    This path is intended for Bunnelby's post-wake conversation runtime. The waveform is
    passed to faster-whisper as a numpy array, so no temporary audio file is created.
    External conversation VAD should already have isolated the user's utterance; a second
    Whisper VAD pass is therefore disabled to avoid trimming words at the boundaries.
    """
    if not stt_enabled():
        raise STTDisabledError("Local speech recognition is disabled.")
    if int(sample_rate) != STT_SAMPLE_RATE:
        raise STTAudioError(f"RAM STT expects {STT_SAMPLE_RATE} Hz mono audio.")

    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise STTAudioError("Audio samples are empty.")
    duration = waveform.size / float(STT_SAMPLE_RATE)
    if duration > stt_max_sample_seconds():
        raise STTAudioError("Audio samples are too long for a single Bunnelby utterance.")
    if not np.isfinite(waveform).all():
        raise STTAudioError("Audio samples contain invalid values.")

    language_hint = _validate_language_hint(language)
    return _transcribe_source(
        np.ascontiguousarray(waveform),
        language_hint=language_hint,
        vad_filter=False,
    )


def transcribe_audio(
    audio_bytes: bytes,
    *,
    content_type: str | None = None,
    language: str | None = "auto",
) -> TranscriptionResult:
    """Transcribe uploaded local audio and delete the temporary file immediately.

    Browser-originated formats such as WebM still require decode-from-file compatibility.
    The microphone runtime uses transcribe_samples() instead and stays RAM-only.
    """
    if not stt_enabled():
        raise STTDisabledError("Local speech recognition is disabled.")
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        raise STTAudioError("Audio payload is empty.")
    if len(audio_bytes) > stt_max_audio_bytes():
        raise STTAudioError("Audio payload is too large for a single Bunnelby utterance.")

    language_hint = _validate_language_hint(language)
    temp_path: str | None = None

    try:
        suffix = _suffix_for_content_type(content_type)
        with tempfile.NamedTemporaryFile(prefix="bunnelby-stt-", suffix=suffix, delete=False) as handle:
            handle.write(bytes(audio_bytes))
            temp_path = handle.name
        return _transcribe_source(
            temp_path,
            language_hint=language_hint,
            vad_filter=True,
        )
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove temporary STT audio file: %s", temp_path)


def _reset_model_cache_for_tests() -> None:
    unload_stt_model()


def unload_stt_model() -> None:
    """Release the current warm STT model for an explicit profile switch/shutdown."""
    global _model, _model_signature
    with _model_lock:
        model = _model
        _model = None
        _model_signature = None
    native_model = getattr(model, "model", None)
    unload = getattr(native_model, "unload_model", None)
    if callable(unload):
        try:
            unload()
        except Exception:
            logger.debug("Could not eagerly unload STT model", exc_info=True)
