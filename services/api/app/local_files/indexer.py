from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Iterable

from .extractors import ExtractionLimits, ExtractedUnit, extract_file, unit_hash
from .models import ChunkInput, IndexedFileInput
from .path_policy import EXCLUDED_DIRECTORY_NAMES, PathPolicy, _is_reparse_point, is_secret_file
from .redaction import redact_secrets
from .storage import FileIndexStore


MAX_CHUNKS_PER_FILE: Final[int] = 512
MAX_CHUNK_CHARS: Final[int] = 2400
CHUNK_OVERLAP_CHARS: Final[int] = 120


@dataclass(frozen=True)
class ReconcileReport:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    metadata_refreshed: int = 0
    deleted: int = 0
    errors: int = 0
    unsafe_skipped: int = 0
    incomplete_roots: tuple[str, ...] = ()


def _content_hash(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(min(1024 * 1024, limit + 1 - read))
            if not block:
                break
            digest.update(block)
            read += len(block)
            if read > limit:
                return ""
    return digest.hexdigest()


def _split_unit(unit: ExtractedUnit) -> Iterable[ExtractedUnit]:
    if len(unit.text) <= MAX_CHUNK_CHARS:
        yield unit
        return
    start = 0
    while start < len(unit.text):
        end = min(len(unit.text), start + MAX_CHUNK_CHARS)
        if end < len(unit.text):
            boundary = max(unit.text.rfind("\n", start, end), unit.text.rfind(" ", start, end))
            if boundary > start + MAX_CHUNK_CHARS // 2:
                end = boundary
        yield replace(unit, text=unit.text[start:end], truncated=True)
        if end >= len(unit.text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_CHARS)


def _chunks(units: Iterable[ExtractedUnit]) -> tuple[tuple[ChunkInput, ...], int, bool]:
    output: list[ChunkInput] = []
    redactions = 0
    truncated = False
    for source_unit in units:
        redacted, count = redact_secrets(source_unit.text)
        redactions += count
        for unit in _split_unit(replace(source_unit, text=redacted)):
            if len(output) >= MAX_CHUNKS_PER_FILE:
                truncated = True
                break
            output.append(
                ChunkInput(
                    ordinal=len(output),
                    text=unit.text,
                    text_hash=unit_hash(unit),
                    page_number=unit.page_number,
                    slide_number=unit.slide_number,
                    sheet_name=unit.sheet_name,
                    row_start=unit.row_start,
                    row_end=unit.row_end,
                    line_start=unit.line_start,
                    line_end=unit.line_end,
                    section=unit.section,
                    extraction_method=unit.extraction_method,
                    confidence=unit.confidence,
                    truncated=unit.truncated,
                )
            )
        if truncated:
            break
    return tuple(output), redactions, truncated


def _walk_root(root: Path) -> tuple[list[Path], bool]:
    files: list[Path] = []
    complete = True
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    name = entry.name.casefold()
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if name in EXCLUDED_DIRECTORY_NAMES or _is_reparse_point(path):
                                continue
                            stack.append(path)
                        elif entry.is_file(follow_symlinks=False):
                            if not is_secret_file(path):
                                files.append(path)
                    except OSError:
                        complete = False
        except OSError:
            complete = False
    return files, complete


class FileIndexer:
    def __init__(self, store: FileIndexStore, policy: PathPolicy, *, limits: ExtractionLimits | None = None) -> None:
        self.store = store
        self.policy = policy
        self.limits = limits or ExtractionLimits()

    def _index_once(self, path: Path, alias: str, relative_path: str) -> str:
        before = path.stat()
        canonical = str(path.resolve(strict=True))
        existing = self.store.file_metadata(canonical)
        if existing and int(existing["size_bytes"]) == before.st_size and int(existing["mtime_ns"]) == before.st_mtime_ns:
            self.store.touch_seen(canonical)
            return "unchanged"

        digest = _content_hash(path, self.limits.max_file_size)
        if existing and digest and digest == str(existing["content_hash"]):
            after_hash = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after_hash.st_size, after_hash.st_mtime_ns):
                return "race"
            self.store.refresh_unchanged_content(canonical, size_bytes=after_hash.st_size, mtime_ns=after_hash.st_mtime_ns)
            return "metadata_refreshed"

        extraction = extract_file(path, self.limits)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return "race"
        chunks, redaction_count, chunk_truncated = _chunks(extraction.units)
        item = IndexedFileInput(
            canonical_path=canonical,
            root_alias=alias,
            relative_path=relative_path,
            filename=path.name,
            extension=path.suffix.casefold(),
            size_bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,
            content_hash=digest,
            file_type=path.suffix.casefold().lstrip(".") or "unknown",
            parser_name=extraction.parser_name,
            parser_version=extraction.parser_version,
            extraction_status=extraction.status,
            needs_ocr=extraction.needs_ocr,
            redaction_count=redaction_count,
            chunks=chunks,
            error_code=extraction.error_code or ("chunk_limit" if chunk_truncated else None),
        )
        self.store.replace_file(item)
        return "error" if extraction.status == "error" else "indexed"

    def _index_decision(self, decision: object) -> str:
        if not decision.allowed or not decision.canonical_path or not decision.root_alias or decision.relative_path is None:
            return "unsafe_skipped"
        for attempt in range(2):
            try:
                outcome = self._index_once(decision.canonical_path, decision.root_alias, decision.relative_path)
            except OSError as exc:
                self.store.mark_stale(str(decision.canonical_path), type(exc).__name__)
                outcome = "error"
            if outcome != "race":
                return outcome
        self.store.mark_stale(str(decision.canonical_path), "changed_during_parse")
        return "error"

    def index_file(self, path: Path | str) -> str:
        return self._index_decision(self.policy.check_file(path))

    def reconcile(self, root_aliases: Iterable[str] | None = None) -> ReconcileReport:
        selected = tuple(root_aliases or self.policy.aliases)
        counts = {"discovered": 0, "indexed": 0, "unchanged": 0, "metadata_refreshed": 0, "deleted": 0, "errors": 0, "unsafe_skipped": 0}
        incomplete: list[str] = []
        with self.store.batch():
            for alias in selected:
                approved = self.policy.root(alias)
                if approved is None:
                    counts["unsafe_skipped"] += 1
                    continue
                paths, complete = _walk_root(approved.path)
                if not complete:
                    incomplete.append(alias)
                seen: set[str] = set()
                for path in paths:
                    counts["discovered"] += 1
                    decision = self.policy.check_file(path)
                    if decision.allowed and decision.canonical_path:
                        seen.add(str(decision.canonical_path))
                    outcome = self._index_decision(decision)
                    key = "errors" if outcome == "error" else outcome
                    if key in counts:
                        counts[key] += 1
                if complete:
                    for row in self.store.all_file_rows((alias,)):
                        canonical = str(row["canonical_path"])
                        if canonical not in seen and self.store.delete_file(canonical):
                            counts["deleted"] += 1
        return ReconcileReport(**counts, incomplete_roots=tuple(incomplete))


__all__ = ["FileIndexer", "ReconcileReport"]
