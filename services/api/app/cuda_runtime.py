from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DLL_DIRECTORY_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()
_PREPENDED_PATH_DIRECTORIES: set[str] = set()


@dataclass(frozen=True)
class WindowsCudaRuntimeConfiguration:
    dll_directories: tuple[str, ...]
    path_directories: tuple[str, ...]


def _is_windows() -> bool:
    # sys.platform is intentionally used instead of a patched os.name. Constructing
    # pathlib.Path while os.name is mocked to "nt" on Ubuntu creates WindowsPath and
    # raises NotImplementedError before a test can reach the production logic.
    return sys.platform == "win32"


def candidate_windows_cuda_dll_directories() -> list[Path]:
    """Return local CUDA/cuDNN DLL directories without mutating system state."""
    if not _is_windows():
        return []

    candidates: list[Path] = []

    # NVIDIA's Windows pip wheels place native DLLs below the active Python
    # environment, e.g. .venv/Lib/site-packages/nvidia/cublas/bin.
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        candidates.extend(sorted(nvidia_root.glob("*/bin")))
        candidates.extend(sorted(nvidia_root.glob("*/lib")))

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
    if not _is_windows() or not hasattr(os, "add_dll_directory"):
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


def prepend_windows_cuda_process_path() -> tuple[str, ...]:
    """Prepend CUDA DLL folders to this process only, idempotently.

    CTranslate2 resolves some deferred CUDA dependencies with the Windows native loader
    during encoder inference. Python's os.add_dll_directory() is sufficient for extension
    imports and ctypes, but it does not cover that deferred LoadLibrary path on every
    Windows build. Updating os.environ affects this process and child processes only; it
    does not modify the user's or machine's persistent PATH.
    """
    if not _is_windows():
        return ()

    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    known = {entry.casefold() for entry in current_entries}
    added: list[str] = []
    for path in candidate_windows_cuda_dll_directories():
        resolved = str(path.resolve())
        key = resolved.casefold()
        if key in known or key in _PREPENDED_PATH_DIRECTORIES:
            continue
        known.add(key)
        _PREPENDED_PATH_DIRECTORIES.add(key)
        added.append(resolved)

    if added:
        os.environ["PATH"] = os.pathsep.join([*added, *current_entries])
    return tuple(added)


def configure_windows_cuda_runtime() -> WindowsCudaRuntimeConfiguration:
    """Configure both Python and deferred native CUDA loading for this process."""
    dll_directories = configure_windows_cuda_dll_directories()
    path_directories = prepend_windows_cuda_process_path()
    return WindowsCudaRuntimeConfiguration(dll_directories, path_directories)
