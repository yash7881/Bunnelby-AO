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
DEFAULT_STT_CONTEXT_BIAS_ENABLED = False
DEFAULT_STT_HOTWORDS = ""
DEFAULT_STT_HINDI_HOTWORDS = ""
DEFAULT_STT_MAX_AUDIO_BYTES = 12 * 1024 * 1024
DEFAULT_STT_MAX_SAMPLE_SECONDS = 120.0
STT_SAMPLE_RATE = 16_000
SUPPORTED_LANGUAGE_HINTS = {"auto", "en", "hi"}

# Voice transcripts are allowed to be imperfect, especially for Hinglish, but a
# weak short-language guess must never become authority to execute a real tool.
# These thresholds are deliberately conservative: language_probability is only
# a language-ID signal, so English/Hindi are never rejected on that value alone.
VOICE_PRIMARY_LANGUAGES = frozenset({"en", "hi"})
MIN_UNSUPPORTED_LANGUAGE_CONFIDENCE = 0.70
MIN_ACCEPTABLE_AVG_LOGPROB = -1.20
HIGH_NO_SPEECH_PROBABILITY = 0.60
NO_SPEECH_AVG_LOGPROB = -0.80

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
    # Decoder evidence is optional for backwards compatibility with callers and
    # tests that construct TranscriptionResult directly. The live Whisper path
    # populates these whenever the model supplies segment metadata.
    average_log_probability: float | None = None
    max_no_speech_probability: float | None = None


@dataclass(frozen=True)
class STTRuntimeProfile:
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    hotwords: str | None


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


def stt_context_bias_enabled() -> bool:
    """Whether optional decoder hotwords may influence recognition.

    Bunnelby is a general assistant, so domain vocabulary is opt-in rather than
    a global default. This also makes stale Gmail/Calendar hotword entries in an
    old local .env inert unless the user deliberately enables decoder bias.
    """
    return _env_bool("STT_CONTEXT_BIAS_ENABLED", DEFAULT_STT_CONTEXT_BIAS_ENABLED)


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


def stt_hotwords() -> str | None:
    """Return bounded optional decoder context only when explicitly enabled."""
    if not stt_context_bias_enabled():
        return None
    normalized = " ".join(os.getenv("STT_HOTWORDS", DEFAULT_STT_HOTWORDS).split())
    return normalized[:300] or None


def stt_hindi_hotwords() -> str | None:
    """Optional Hindi rescue context; neutral/empty by default."""
    if not stt_context_bias_enabled():
        return None
    normalized = " ".join(
        os.getenv("STT_HOTWORDS_HI", DEFAULT_STT_HINDI_HOTWORDS).split()
    )
    return normalized[:300] or None


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
        hotwords=stt_hotwords(),
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


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _result_from_transcription(segments: Any, info: Any, language_hint: str) -> TranscriptionResult:
    segment_list = list(segments)
    text = _normalize_text([str(segment.text) for segment in segment_list])
    detected_language = str(getattr(info, "language", "") or language_hint or "auto")
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    duration = float(getattr(info, "duration", 0.0) or 0.0)

    log_probabilities = [
        value
        for segment in segment_list
        if (value := _finite_float(getattr(segment, "avg_logprob", None))) is not None
    ]
    no_speech_probabilities = [
        value
        for segment in segment_list
        if (value := _finite_float(getattr(segment, "no_speech_prob", None))) is not None
    ]

    average_log_probability = (
        sum(log_probabilities) / len(log_probabilities)
        if log_probabilities
        else None
    )
    max_no_speech_probability = (
        max(0.0, min(max(no_speech_probabilities), 1.0))
        if no_speech_probabilities
        else None
    )

    return TranscriptionResult(
        text=text,
        language=detected_language,
        language_probability=max(0.0, min(probability, 1.0)),
        duration_seconds=max(0.0, duration),
        average_log_probability=average_log_probability,
        max_no_speech_probability=max_no_speech_probability,
    )


def microphone_transcription_is_trustworthy(
    result: TranscriptionResult,
    *,
    language_hint: str = "auto",
) -> bool:
    """Fail closed on weak microphone evidence before text can reach /chat.

    The policy intentionally does NOT use language_probability as a generic
    transcript-confidence score: Hinglish often has ambiguous language ID. A
    low-confidence *unsupported* language guess is different -- e.g. a short
    English/Hinglish command misdetected as Malayalam -- and triggers the
    runtime's existing bounded English/Hindi rescue instead of being executed.
    """
    if not str(result.text or "").strip():
        return False

    average_log_probability = result.average_log_probability
    max_no_speech_probability = result.max_no_speech_probability

    if (
        average_log_probability is not None
        and average_log_probability < MIN_ACCEPTABLE_AVG_LOGPROB
    ):
        return False

    if (
        average_log_probability is not None
        and max_no_speech_probability is not None
        and average_log_probability < NO_SPEECH_AVG_LOGPROB
        and max_no_speech_probability > HIGH_NO_SPEECH_PROBABILITY
    ):
        return False

    normalized_hint = _validate_language_hint(language_hint)
    if normalized_hint == "auto":
        detected = str(result.language or "").strip().casefold()
        if (
            detected
            and detected not in VOICE_PRIMARY_LANGUAGES
            and detected != "auto"
            and result.language_probability < MIN_UNSUPPORTED_LANGUAGE_CONFIDENCE
        ):
            return False

    return True


def _with_empty_text(result: TranscriptionResult) -> TranscriptionResult:
    return TranscriptionResult(
        text="",
        language=result.language,
        language_probability=result.language_probability,
        duration_seconds=result.duration_seconds,
        average_log_probability=result.average_log_probability,
        max_no_speech_probability=result.max_no_speech_probability,
    )


def _transcribe_source(
    source: Any,
    *,
    language_hint: str,
    vad_filter: bool,
    hotwords_override: str | None = None,
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
            "hotwords": hotwords_override if hotwords_override is not None else stt_hotwords(),
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
    hotwords_override: str | None = None,
) -> TranscriptionResult:
    """Transcribe one microphone utterance directly from RAM.

    This path is intended for Bunnelby's post-wake conversation runtime. The waveform is
    passed to faster-whisper as a numpy array, so no temporary audio file is created.
    External conversation VAD should already have isolated the user's utterance; a second
    Whisper VAD pass is therefore disabled to avoid trimming words at the boundaries.

    A weak automatic-language/acoustic result is returned with empty text. The persistent
    runtime already treats empty text as non-authoritative, performs its bounded rescue
    passes, and refuses to dispatch anything to /chat if no trustworthy candidate exists.
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
    result = _transcribe_source(
        np.ascontiguousarray(waveform),
        language_hint=language_hint,
        vad_filter=False,
        hotwords_override=hotwords_override,
    )
    if microphone_transcription_is_trustworthy(result, language_hint=language_hint):
        return result

    logger.warning(
        "Rejected uncertain microphone transcript before dispatch: "
        "language=%s language_probability=%.3f avg_logprob=%s no_speech=%s",
        result.language,
        result.language_probability,
        result.average_log_probability,
        result.max_no_speech_probability,
    )
    return _with_empty_text(result)


def transcribe_audio(
    audio_bytes: bytes,
    *,
    content_type: str | None = None,
    language: str | None = "auto",
) -> TranscriptionResult:
    """Transcribe uploaded local audio and delete the temporary file immediately.

    Browser-originated formats such as WebM still require decode-from-file compatibility.
    The microphone runtime uses transcribe_samples() instead and stays RAM-only. Uploaded
    STT remains a transcription API; the stricter execution trust gate applies only to the
    persistent microphone-command path.
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
