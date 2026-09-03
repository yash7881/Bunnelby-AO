from __future__ import annotations

import ctypes
import fnmatch
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping


_ALIAS_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

EXCLUDED_DIRECTORY_NAMES: Final[frozenset[str]] = frozenset(
    name.casefold()
    for name in (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".next",
        ".ao-backups",
        "dist",
        "build",
        "coverage",
        "site-packages",
        "appdata",
        "programdata",
        "program files",
        "program files (x86)",
        "windows",
    )
)

SECRET_FILE_PATTERNS: Final[tuple[str, ...]] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "client_secret*.json",
    "credentials*.json",
    "oauth*.json",
    "token*.json",
    "secrets.*",
    "*.kdbx",
    "*.db",
    "*.db-wal",
    "*.db-shm",
    "*.db-journal",
)


class RootResolutionError(RuntimeError):
    """Raised when Windows cannot supply a required known folder safely."""


def _canonical(path: Path, *, strict: bool) -> Path:
    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"path cannot be canonicalized: {path}") from exc


def _windows_key(path: Path) -> str:
    # normcase is the platform-supported Windows case/drive normalizer.  The
    # explicit casefold also keeps injected test roots deterministic.
    return os.path.normcase(str(path)).casefold()


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//")


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & flag)


def is_secret_file(path: Path) -> bool:
    """Name-only decision suitable for use before opening a candidate."""
    name = path.name.casefold()
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in SECRET_FILE_PATTERNS)


@dataclass(frozen=True)
class ApprovedRoot:
    alias: str
    path: Path

    def __post_init__(self) -> None:
        alias = self.alias.strip().casefold()
        if not _ALIAS_RE.fullmatch(alias):
            raise ValueError("root alias must be a short machine identifier")
        canonical = _canonical(Path(self.path), strict=True)
        if _is_unc(canonical):
            raise ValueError("UNC/network roots are not approved by default")
        if not canonical.is_dir():
            raise ValueError("approved root must be an existing directory")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "path", canonical)


@dataclass(frozen=True)
class PathDecision:
    allowed: bool
    reason: str
    canonical_path: Path | None = None
    root_alias: str | None = None
    relative_path: str | None = None


class PathPolicy:
    """Fail-closed policy used by discovery, indexing, search verification, and watchers."""

    def __init__(self, roots: Iterable[ApprovedRoot]) -> None:
        by_alias: dict[str, ApprovedRoot] = {}
        for root in roots:
            if root.alias in by_alias:
                raise ValueError(f"duplicate approved root alias: {root.alias}")
            by_alias[root.alias] = root
        if not by_alias:
            raise ValueError("at least one approved root is required")
        self._roots = by_alias

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(self._roots)

    def root(self, alias: str) -> ApprovedRoot | None:
        return self._roots.get(str(alias).strip().casefold())

    def check_file(self, candidate: Path | str) -> PathDecision:
        raw = Path(candidate)
        if _is_unc(raw):
            return PathDecision(False, "network_path")
        if is_secret_file(raw):
            return PathDecision(False, "secret_file")
        try:
            canonical = _canonical(raw, strict=True)
        except ValueError:
            return PathDecision(False, "unresolvable_path")
        if _is_unc(canonical):
            return PathDecision(False, "network_path")

        for root in self._roots.values():
            try:
                relative = canonical.relative_to(root.path)
            except ValueError:
                # pathlib's comparison follows host semantics; commonpath below
                # additionally pins Windows-style case-insensitive containment.
                try:
                    common = os.path.commonpath((_windows_key(canonical), _windows_key(root.path)))
                except ValueError:
                    continue
                if common != _windows_key(root.path):
                    continue
                try:
                    relative = Path(os.path.relpath(canonical, root.path))
                except ValueError:
                    continue

            parts = relative.parts
            if any(part.casefold() in EXCLUDED_DIRECTORY_NAMES for part in parts[:-1]):
                return PathDecision(False, "excluded_directory")
            if is_secret_file(canonical):
                return PathDecision(False, "secret_file")

            # Reject links/junctions at every traversed component.  Checking the
            # canonical target alone would miss that the original spelling
            # passed through a junction before resolve() escaped it.
            try:
                unresolved_relative = raw.absolute().relative_to(root.path)
            except ValueError:
                unresolved_relative = relative
            cursor = root.path
            for part in unresolved_relative.parts:
                cursor = cursor / part
                if _is_reparse_point(cursor):
                    return PathDecision(False, "reparse_point")

            if not canonical.is_file():
                return PathDecision(False, "not_regular_file")
            return PathDecision(
                True,
                "approved",
                canonical_path=canonical,
                root_alias=root.alias,
                relative_path=relative.as_posix(),
            )
        return PathDecision(False, "outside_approved_roots")


# KNOWNFOLDERID values from Microsoft's Known Folder ID documentation.
_KNOWN_FOLDER_GUIDS: Final[Mapping[str, str]] = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
}


class _GUID(ctypes.Structure):
    _fields_ = (("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8))

    @classmethod
    def parse(cls, value: str) -> "_GUID":
        import uuid

        raw = uuid.UUID(value).bytes_le
        return cls.from_buffer_copy(raw)


def resolve_windows_known_folder(alias: str) -> Path:
    """Resolve Desktop/Documents/Downloads through SHGetKnownFolderPath."""
    normalized = alias.strip().casefold()
    if normalized not in _KNOWN_FOLDER_GUIDS:
        raise RootResolutionError(f"unknown Windows known-folder alias: {alias}")
    if sys.platform != "win32":
        raise RootResolutionError("Windows Known Folder APIs are unavailable")
    folder_id = _GUID.parse(_KNOWN_FOLDER_GUIDS[normalized])
    output = ctypes.c_wchar_p()
    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    shell32.SHGetKnownFolderPath.argtypes = (
        ctypes.POINTER(_GUID),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)
    ole32.CoTaskMemFree.restype = None
    status = shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(output))
    if status != 0 or not output.value:
        raise RootResolutionError(f"SHGetKnownFolderPath failed for {normalized}: 0x{status & 0xFFFFFFFF:08x}")
    try:
        return _canonical(Path(output.value), strict=True)
    finally:
        ole32.CoTaskMemFree(output)


def default_windows_roots() -> tuple[ApprovedRoot, ...]:
    """Return only OS-resolved roots; never guesses a user-specific path."""
    return tuple(
        ApprovedRoot(alias, resolve_windows_known_folder(alias))
        for alias in ("desktop", "documents", "downloads")
    )


__all__ = [
    "ApprovedRoot",
    "EXCLUDED_DIRECTORY_NAMES",
    "PathDecision",
    "PathPolicy",
    "RootResolutionError",
    "SECRET_FILE_PATTERNS",
    "default_windows_roots",
    "is_secret_file",
    "resolve_windows_known_folder",
]
