from __future__ import annotations

import argparse
import ctypes
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.cuda_runtime import (
    candidate_windows_cuda_dll_directories,
    configure_windows_cuda_dll_directories,
)

REQUIRED_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
)
DEFAULT_COMPUTE_TYPE = "int8_float16"


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


def run(smoke_model: bool, compute_type: str) -> int:
    import ctranslate2

    print("=" * 78)
    print("BUNNELBY RTX STT GPU PREFLIGHT")
    print("=" * 78)
    print(f"Python: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"CTranslate2: {ctranslate2.__version__}")

    cuda_devices = ctranslate2.get_cuda_device_count()
    print(f"CUDA devices visible: {cuda_devices}")
    if cuda_devices <= 0:
        print("RESULT: FAIL — CTranslate2 cannot see a CUDA device.")
        return 2

    try:
        supported = sorted(ctranslate2.get_supported_compute_types("cuda", 0))
    except Exception as exc:
        print(f"CUDA compute-type query failed: {exc}")
        return 3

    print(f"CUDA compute types: {', '.join(supported)}")
    if compute_type not in supported:
        print(f"RESULT: FAIL — requested compute type {compute_type!r} is not supported.")
        return 4

    registered = configure_windows_cuda_dll_directories()
    if registered:
        print("Registered process-local NVIDIA DLL directories:")
        for path in registered:
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
            "RESULT: FAIL — CUDA device is visible, but required runtime DLLs "
            "are missing or not loadable."
        )
        print("Do not run the full Bunnelby GPU STT benchmark until this preflight passes.")
        return 5

    if smoke_model:
        print()
        print("GPU Whisper smoke test: loading faster-whisper small...")
        from faster_whisper import WhisperModel

        import os

        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            model_root = Path(local_app_data) / "Bunnelby" / "models" / "stt"
        else:
            model_root = Path.home() / ".bunnelby" / "models" / "stt"
        model_root.mkdir(parents=True, exist_ok=True)

        try:
            model = WhisperModel(
                "small",
                device="cuda",
                compute_type=compute_type,
                num_workers=1,
                download_root=str(model_root),
            )
        except Exception as exc:
            print(f"RESULT: FAIL — GPU Whisper model could not load: {exc}")
            return 6
        else:
            del model
            print("GPU Whisper model load: PASS")

    print()
    print("RESULT: PASS — CUDA runtime is ready for the controlled Bunnelby STT benchmark.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only RTX/CUDA preflight for Bunnelby conversation STT."
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="CUDA compute type to validate (default: int8_float16).",
    )
    parser.add_argument(
        "--smoke-model",
        action="store_true",
        help="After DLL checks pass, also construct the faster-whisper small CUDA model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.smoke_model, args.compute_type))
