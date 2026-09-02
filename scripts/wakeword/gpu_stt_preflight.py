from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.cuda_runtime import (
    candidate_windows_cuda_dll_directories,
    configure_windows_cuda_runtime,
)

REQUIRED_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
)
DEFAULT_COMPUTE_TYPE = "int8_float16"
DEFAULT_MODEL = "small"
SMOKE_SAMPLE_RATE = 16_000


def _find_dll(name: str, directories: list[Path]) -> Path | None:
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _load_dll(path: Path | None, name: str) -> tuple[bool, str]:
    if platform.system() != "Windows":
        return False, "Windows-only DLL check"
    target = str(path) if path else name
    try:
        ctypes.WinDLL(target)
    except OSError as exc:
        return False, str(exc)
    return True, "loadable"


def run(*, run_inference: bool, compute_type: str, model_name: str) -> int:
    # Configure before importing CTranslate2/faster-whisper. Some CUDA dependencies are
    # loaded lazily at the first encoder call and consult this process's PATH.
    configuration = configure_windows_cuda_runtime()

    import ctranslate2

    print("=" * 78)
    print("BUNNELBY RTX STT GPU PREFLIGHT")
    print("=" * 78)
    print(f"Python: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"CTranslate2: {ctranslate2.__version__}")
    print(f"Selected model: {model_name}")
    print("Requested device: cuda")
    print("Effective device: cuda")
    print(f"Requested compute type: {compute_type}")
    print(f"Effective compute type: {compute_type}")

    cuda_devices = ctranslate2.get_cuda_device_count()
    print(f"CUDA devices visible: {cuda_devices}")
    if cuda_devices <= 0:
        print("RESULT: FAIL - CTranslate2 cannot see a CUDA device.")
        return 2

    try:
        supported = sorted(ctranslate2.get_supported_compute_types("cuda", 0))
    except Exception as exc:
        print(f"CUDA compute-type query failed: {exc}")
        return 3

    print(f"CUDA compute types: {', '.join(supported)}")
    if compute_type not in supported:
        print(f"RESULT: FAIL - requested compute type {compute_type!r} is not supported.")
        return 4

    if configuration.dll_directories:
        print("Registered process-local NVIDIA DLL directories:")
        for path in configuration.dll_directories:
            print(f"  {path}")
    if configuration.path_directories:
        print("Prepended process-local PATH directories:")
        for path in configuration.path_directories:
            print(f"  {path}")

    directories = candidate_windows_cuda_dll_directories()
    print()
    print("Required CUDA/cuDNN DLLs:")
    all_loadable = True
    for name in REQUIRED_DLLS:
        path = _find_dll(name, directories)
        loadable, detail = _load_dll(path, name)
        all_loadable = all_loadable and loadable
        location = str(path) if path else "NOT FOUND in venv NVIDIA bins / CUDA 12 bins"
        status = "PASS" if loadable else "FAIL"
        print(f"  {name}: {status}")
        print(f"    location: {location}")
        print(f"    loader: {detail}")

    if not all_loadable:
        print()
        print(
            "RESULT: FAIL - CUDA device is visible, but required runtime DLLs "
            "are missing or not loadable."
        )
        return 5

    if not run_inference:
        print()
        print("REAL inference: NOT RUN (--dll-only diagnostic mode)")
        print("RESULT: PASS (DLL checks only) - CUDA STT is NOT production-verified.")
        return 0

    print()
    print(f"GPU Whisper model load: {model_name}")
    from faster_whisper import WhisperModel

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        model_root = Path(local_app_data) / "Bunnelby" / "models" / "stt"
    else:
        model_root = Path.home() / ".bunnelby" / "models" / "stt"
    model_root.mkdir(parents=True, exist_ok=True)

    try:
        load_started = time.perf_counter()
        model = WhisperModel(
            model_name,
            device="cuda",
            compute_type=compute_type,
            num_workers=1,
            download_root=str(model_root),
        )
    except Exception as exc:
        print(f"RESULT: FAIL - GPU Whisper model could not load: {exc}")
        return 6
    print(f"Model load: PASS ({time.perf_counter() - load_started:.2f}s)")

    # Transcription is lazy. list(segments) is required to force the real CUDA encoder
    # and prove deferred cuBLAS/cuDNN resolution instead of only constructing an object.
    waveform = np.zeros(SMOKE_SAMPLE_RATE, dtype=np.float32)
    try:
        inference_started = time.perf_counter()
        segments, info = model.transcribe(
            waveform,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        realized_segments = list(segments)
        inference_seconds = time.perf_counter() - inference_started
    except Exception as exc:
        print(f"REAL inference: FAIL - {exc}")
        print("RESULT: FAIL - model construction passed but CUDA inference failed.")
        return 7
    finally:
        del model

    detected_language = str(getattr(info, "language", "") or "unknown")
    print(
        "REAL inference: PASS "
        f"({inference_seconds:.2f}s, segments={len(realized_segments)}, "
        f"language={detected_language})"
    )
    print("Inference waveform persisted: NO")
    print()
    print("RESULT: PASS - CUDA runtime and real faster-whisper inference are ready.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTX/CUDA real-inference preflight for Bunnelby conversation STT."
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="CUDA compute type to validate (default: int8_float16).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="faster-whisper model used for the real inference gate (default: small).",
    )
    parser.add_argument(
        "--dll-only",
        action="store_true",
        help="Run discovery/DLL checks only; this does not verify production inference.",
    )
    # Backward compatibility for the previously documented command. Real inference is now
    # the default and is strictly stronger than the old construction-only smoke test.
    parser.add_argument("--smoke-model", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run(
            run_inference=not args.dll_only,
            compute_type=args.compute_type,
            model_name=args.model,
        )
    )
