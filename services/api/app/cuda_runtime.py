from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_DLL_DIRECTORY_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()


def candidate_windows_cuda_dll_directories() -> list[Path]:
    """Return local CUDA/cuDNN DLL directories without mutating system state."""
    if os.name != "nt":
        return []

    candidates: list[Path] = []

    # NVIDIA's Windows pip wheels place native DLLs below the active Python
    # environment, e.g. .venv/Lib/site-packages/nvidia/cublas/bin.
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        candidates.extend(nvidia_root.glob("*/bin"))
        candidates.extend(nvidia_root.glob("*/lib"))

    # Also support an existing system CUDA 12 installation when present.
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        candidates.extend(sorted(cuda_root.glob("v12*/bin"), reverse=True))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_dir():
            continue
        key = str(path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def configure_windows_cuda_dll_directories() -> tuple[str, ...]:
    """Register CUDA/cuDNN DLL folders for this Python process only.

    Python 3.8+ on Windows uses an explicit DLL search-path API for imported
    extension dependencies. Keeping the returned handles alive is required for
    the directories to remain registered. Nothing is written to the user's
    global PATH or Windows configuration.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return ()

    added: list[str] = []
    for path in candidate_windows_cuda_dll_directories():
        resolved = str(path.resolve())
        key = resolved.casefold()
        if key in _REGISTERED_DLL_DIRECTORIES:
            continue
        try:
            handle = os.add_dll_directory(resolved)
        except OSError:
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        _REGISTERED_DLL_DIRECTORIES.add(key)
        added.append(resolved)
    return tuple(added)
