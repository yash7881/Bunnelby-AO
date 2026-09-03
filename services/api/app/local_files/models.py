from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping


SearchMode = Literal["filename", "content", "hybrid"]


@dataclass(frozen=True)
class ChunkInput:
    ordinal: int
    text: str
    text_hash: str
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    section: str | None = None
    extraction_method: str = "native"
    confidence: float = 1.0
    truncated: bool = False


@dataclass(frozen=True)
class IndexedFileInput:
    canonical_path: str
    root_alias: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    mtime_ns: int
    content_hash: str
    file_type: str
    parser_name: str
    parser_version: str
    extraction_status: str
    needs_ocr: bool
    redaction_count: int
    chunks: tuple[ChunkInput, ...] = ()
    error_code: str | None = None


@dataclass(frozen=True)
class FileSearchResult:
    file_id: int
    root_alias: str
    relative_path: str
    filename: str
    extension: str
    score: float
    match_type: Literal["filename", "content", "hybrid"]
    snippet: str
    provenance: Mapping[str, object] = field(default_factory=dict)
    modified_at: datetime | None = None
    size: int = 0
    extraction_status: str = "metadata_only"
    needs_ocr: bool = False
    chunk_id: int | None = None


__all__ = ["ChunkInput", "FileSearchResult", "IndexedFileInput", "SearchMode"]
