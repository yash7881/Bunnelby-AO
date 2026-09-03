from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Sequence

from .models import FileSearchResult, IndexedFileInput, SearchMode


SCHEMA_VERSION: Final[int] = 1
MAX_SEARCH_LIMIT: Final[int] = 50
MAX_SNIPPET_CHARS: Final[int] = 500


class FTS5UnavailableError(RuntimeError):
    """The index must not silently degrade to an unindexed content scan."""


def _literal_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    current: list[str] = []
    for character in str(value):
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"} or character == "_":
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return tuple(term for term in terms if term)


def compile_fts_query(value: str, *, trigram: bool = False) -> str | None:
    """Compile user text into literal FTS phrases, never raw FTS grammar."""
    terms = _literal_terms(value)
    if trigram:
        terms = tuple(term for term in terms if len(term) >= 3)
    if not terms:
        return None
    escaped = [term.replace('"', '""') for term in terms]
    return " AND ".join(f'"{term}"' for term in escaped)


class FileIndexStore:
    """Owned SQLite connection for the disposable, non-authoritative file index."""

    def __init__(self, database_path: Path | str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._batch_depth = 0
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        # This is the disposable file index, not ao.db. WAL + NORMAL avoids one
        # full-disk sync per indexed file while retaining atomic commits; a
        # missed tail is always recoverable from the authoritative filesystem.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextlib.contextmanager
    def _write_scope(self):
        with self._lock:
            if self._batch_depth:
                yield
            else:
                with self._connection:
                    yield

    @contextlib.contextmanager
    def batch(self):
        """Amortize commits during a reconcile while preserving per-file replacement atomicity."""
        with self._lock:
            outermost = self._batch_depth == 0
            if outermost:
                self._connection.execute("BEGIN")
            self._batch_depth += 1
            try:
                yield
            except Exception:
                self._batch_depth -= 1
                if outermost:
                    self._connection.rollback()
                raise
            else:
                self._batch_depth -= 1
                if outermost:
                    self._connection.commit()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM index_metadata WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0

    def table_names(self) -> set[str]:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def file_count(self) -> int:
        return int(self._connection.execute("SELECT count(*) FROM files").fetchone()[0])

    def _initialize(self) -> None:
        try:
            self._connection.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(value)")
            self._connection.execute("DROP TABLE temp.__fts5_probe")
            self._connection.execute(
                "CREATE VIRTUAL TABLE temp.__trigram_probe USING fts5(value, tokenize='trigram')"
            )
            self._connection.execute("DROP TABLE temp.__trigram_probe")
        except sqlite3.OperationalError as exc:
            raise FTS5UnavailableError(
                "SQLite FTS5 with unicode61 and trigram tokenizers is required"
            ) from exc

        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    canonical_path TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    root_alias TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    extraction_status TEXT NOT NULL,
                    needs_ocr INTEGER NOT NULL DEFAULT 0 CHECK(needs_ocr IN (0,1)),
                    indexed_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    error_code TEXT,
                    redaction_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    page_number INTEGER,
                    slide_number INTEGER,
                    sheet_name TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    section TEXT,
                    extraction_method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    truncated INTEGER NOT NULL DEFAULT 0 CHECK(truncated IN (0,1)),
                    UNIQUE(file_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS ix_files_root_extension ON files(root_alias, extension);
                CREATE INDEX IF NOT EXISTS ix_files_seen ON files(last_seen_at);
                CREATE INDEX IF NOT EXISTS ix_chunks_file ON chunks(file_id, ordinal);
                CREATE VIRTUAL TABLE IF NOT EXISTS filename_fts USING fts5(
                    file_id UNINDEXED,
                    filename,
                    relative_path,
                    tokenize='trigram'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    chunk_id UNINDEXED,
                    file_id UNINDEXED,
                    text,
                    tokenize='unicode61 remove_diacritics 0'
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM index_metadata WHERE key='schema_version'"
            ).fetchone()
            if row and int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported file index schema {row[0]}; rebuild is required"
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO index_metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    def replace_file(self, item: IndexedFileInput) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._write_scope():
            existing = self._connection.execute(
                "SELECT id FROM files WHERE canonical_path=?", (item.canonical_path,)
            ).fetchone()
            if existing:
                file_id = int(existing[0])
                self._connection.execute(
                    """UPDATE files SET root_alias=?,relative_path=?,filename=?,extension=?,
                    size_bytes=?,mtime_ns=?,content_hash=?,file_type=?,parser_name=?,parser_version=?,
                    extraction_status=?,needs_ocr=?,indexed_at=?,last_seen_at=?,error_code=?,redaction_count=?
                    WHERE id=?""",
                    (
                        item.root_alias, item.relative_path, item.filename, item.extension,
                        item.size_bytes, item.mtime_ns, item.content_hash, item.file_type,
                        item.parser_name, item.parser_version, item.extraction_status,
                        int(item.needs_ocr), now, now, item.error_code, item.redaction_count,
                        file_id,
                    ),
                )
                chunk_ids = [row[0] for row in self._connection.execute("SELECT id FROM chunks WHERE file_id=?", (file_id,))]
                if chunk_ids:
                    self._connection.executemany("DELETE FROM content_fts WHERE chunk_id=?", ((value,) for value in chunk_ids))
                self._connection.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
                self._connection.execute("DELETE FROM filename_fts WHERE file_id=?", (file_id,))
            else:
                cursor = self._connection.execute(
                    """INSERT INTO files(canonical_path,root_alias,relative_path,filename,extension,
                    size_bytes,mtime_ns,content_hash,file_type,parser_name,parser_version,
                    extraction_status,needs_ocr,indexed_at,last_seen_at,error_code,redaction_count)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item.canonical_path, item.root_alias, item.relative_path, item.filename,
                        item.extension, item.size_bytes, item.mtime_ns, item.content_hash,
                        item.file_type, item.parser_name, item.parser_version,
                        item.extraction_status, int(item.needs_ocr), now, now,
                        item.error_code, item.redaction_count,
                    ),
                )
                file_id = int(cursor.lastrowid)
            self._connection.execute(
                "INSERT INTO filename_fts(file_id,filename,relative_path) VALUES(?,?,?)",
                (file_id, item.filename, item.relative_path),
            )
            for chunk in item.chunks:
                cursor = self._connection.execute(
                    """INSERT INTO chunks(file_id,ordinal,text,text_hash,page_number,slide_number,
                    sheet_name,row_start,row_end,line_start,line_end,section,extraction_method,confidence,truncated)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        file_id, chunk.ordinal, chunk.text, chunk.text_hash, chunk.page_number,
                        chunk.slide_number, chunk.sheet_name, chunk.row_start, chunk.row_end,
                        chunk.line_start, chunk.line_end, chunk.section,
                        chunk.extraction_method, chunk.confidence, int(chunk.truncated),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO content_fts(chunk_id,file_id,text) VALUES(?,?,?)",
                    (int(cursor.lastrowid), file_id, chunk.text),
                )
            return file_id

    def delete_file(self, canonical_path: str) -> bool:
        with self._write_scope():
            row = self._connection.execute("SELECT id FROM files WHERE canonical_path=?", (canonical_path,)).fetchone()
            if not row:
                return False
            file_id = int(row[0])
            chunk_ids = [item[0] for item in self._connection.execute("SELECT id FROM chunks WHERE file_id=?", (file_id,))]
            self._connection.executemany("DELETE FROM content_fts WHERE chunk_id=?", ((value,) for value in chunk_ids))
            self._connection.execute("DELETE FROM filename_fts WHERE file_id=?", (file_id,))
            self._connection.execute("DELETE FROM files WHERE id=?", (file_id,))
            return True

    def file_metadata(self, canonical_path: str) -> sqlite3.Row | None:
        return self._connection.execute("SELECT * FROM files WHERE canonical_path=?", (canonical_path,)).fetchone()

    def touch_seen(self, canonical_path: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._write_scope():
            self._connection.execute(
                "UPDATE files SET last_seen_at=? WHERE canonical_path=?",
                (now, canonical_path),
            )

    def refresh_unchanged_content(
        self,
        canonical_path: str,
        *,
        size_bytes: int,
        mtime_ns: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._write_scope():
            self._connection.execute(
                """UPDATE files SET size_bytes=?,mtime_ns=?,last_seen_at=?,indexed_at=?,
                extraction_status=CASE WHEN extraction_status='stale' THEN 'indexed' ELSE extraction_status END,
                error_code=NULL WHERE canonical_path=?""",
                (size_bytes, mtime_ns, now, now, canonical_path),
            )

    def mark_stale(self, canonical_path: str, error_code: str = "unreadable") -> None:
        """Make prior text unsearchable immediately when freshness is uncertain."""
        with self._write_scope():
            row = self._connection.execute(
                "SELECT id FROM files WHERE canonical_path=?", (canonical_path,)
            ).fetchone()
            if not row:
                return
            file_id = int(row[0])
            chunk_ids = [item[0] for item in self._connection.execute("SELECT id FROM chunks WHERE file_id=?", (file_id,))]
            self._connection.executemany("DELETE FROM content_fts WHERE chunk_id=?", ((value,) for value in chunk_ids))
            self._connection.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            self._connection.execute("DELETE FROM filename_fts WHERE file_id=?", (file_id,))
            self._connection.execute(
                "UPDATE files SET extraction_status='stale',error_code=? WHERE id=?",
                (error_code, file_id),
            )

    def all_file_rows(self, root_aliases: Sequence[str] | None = None) -> list[sqlite3.Row]:
        if not root_aliases:
            return list(self._connection.execute("SELECT * FROM files"))
        placeholders = ",".join("?" for _ in root_aliases)
        return list(self._connection.execute(f"SELECT * FROM files WHERE root_alias IN ({placeholders})", tuple(root_aliases)))

    def chunk_row(self, chunk_id: int) -> sqlite3.Row | None:
        return self._connection.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()

    def file_row(self, file_id: int) -> sqlite3.Row | None:
        return self._connection.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()

    def _rows_by_ids(self, table: str, values: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not values:
            return {}
        unique = tuple(dict.fromkeys(int(value) for value in values))
        placeholders = ",".join("?" for _ in unique)
        rows = self._connection.execute(
            f"SELECT * FROM {table} WHERE id IN ({placeholders})", unique
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        limit: int = 10,
        root_aliases: Sequence[str] = (),
        extensions: Sequence[str] = (),
        modified_after_ns: int | None = None,
        modified_before_ns: int | None = None,
        file_ids: Sequence[int] = (),
    ) -> list[FileSearchResult]:
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        filename_query = compile_fts_query(query, trigram=True)
        content_query = compile_fts_query(query)
        candidates: dict[int, dict[str, object]] = {}

        if mode in {"filename", "hybrid"} and filename_query:
            rows = self._connection.execute(
                """SELECT CAST(file_id AS INTEGER) file_id,
                bm25(filename_fts,0.0,8.0,3.0) AS rank
                FROM filename_fts WHERE filename_fts MATCH ? ORDER BY rank LIMIT ?""",
                (filename_query, min(2000, limit * 100)),
            )
            for row in rows:
                candidates[int(row["file_id"])] = {
                    "filename_score": 2.0 + 1.0 / (1.0 + abs(float(row["rank"]))),
                    "content_score": 0.0,
                    "snippet": "",
                    "chunk_id": None,
                }
        elif mode in {"filename", "hybrid"} and len(query.strip()) < 3:
            # Trigram cannot represent one/two-character terms. This bounded
            # metadata-only fallback is intentional and is never used as a
            # substitute for missing FTS5 content search.
            rows = self._connection.execute(
                "SELECT id FROM files WHERE instr(lower(filename),lower(?))>0 LIMIT ?",
                (query.strip(), limit * 20),
            )
            for row in rows:
                candidates[int(row[0])] = {"filename_score": 2.0, "content_score": 0.0, "snippet": "", "chunk_id": None}

        if mode in {"content", "hybrid"} and content_query:
            rows = self._connection.execute(
                """SELECT CAST(file_id AS INTEGER) file_id, CAST(chunk_id AS INTEGER) chunk_id,
                bm25(content_fts,0.0,0.0,1.0) AS rank,
                snippet(content_fts,2,'[',']',' … ',24) AS snippet
                FROM content_fts WHERE content_fts MATCH ? ORDER BY rank LIMIT ?""",
                (content_query, min(2000, limit * 100)),
            )
            for row in rows:
                file_id = int(row["file_id"])
                score = 1.0 / (1.0 + abs(float(row["rank"])))
                state = candidates.setdefault(file_id, {"filename_score": 0.0, "content_score": 0.0, "snippet": "", "chunk_id": None})
                if score > float(state["content_score"]):
                    state.update(content_score=score, snippet=str(row["snippet"] or ""), chunk_id=int(row["chunk_id"]))

        allowed_roots = {item.casefold() for item in root_aliases}
        allowed_exts = {item.casefold() if item.startswith(".") else "." + item.casefold() for item in extensions}
        allowed_ids = {int(item) for item in file_ids}
        file_rows = self._rows_by_ids("files", tuple(candidates))
        chunk_rows = self._rows_by_ids(
            "chunks",
            tuple(int(state["chunk_id"]) for state in candidates.values() if state["chunk_id"]),
        )
        results: list[FileSearchResult] = []
        for file_id, state in candidates.items():
            row = file_rows.get(file_id)
            if not row or row["extraction_status"] in {"deleted", "stale"}:
                continue
            if allowed_roots and str(row["root_alias"]).casefold() not in allowed_roots:
                continue
            if allowed_exts and str(row["extension"]).casefold() not in allowed_exts:
                continue
            if allowed_ids and file_id not in allowed_ids:
                continue
            if modified_after_ns is not None and int(row["mtime_ns"]) < modified_after_ns:
                continue
            if modified_before_ns is not None and int(row["mtime_ns"]) > modified_before_ns:
                continue
            chunk = chunk_rows.get(int(state["chunk_id"])) if state["chunk_id"] else None
            filename_score = float(state["filename_score"])
            content_score = float(state["content_score"])
            match_type = "hybrid" if filename_score and content_score else ("filename" if filename_score else "content")
            # Part 11.1: extraction_method and confidence are surfaced so a
            # caller can tell OCR-derived text from native text in a result, and
            # so FileSearchVerifier can compare a hit against the indexed chunk.
            provenance = {
                key: chunk[key]
                for key in (
                    "page_number", "slide_number", "sheet_name", "row_start",
                    "row_end", "line_start", "line_end", "section",
                    "extraction_method", "confidence",
                )
                if chunk is not None and chunk[key] is not None
            }
            snippet = str(state["snippet"] or row["relative_path"])
            results.append(
                FileSearchResult(
                    file_id=file_id,
                    root_alias=str(row["root_alias"]),
                    relative_path=str(row["relative_path"]),
                    filename=str(row["filename"]),
                    extension=str(row["extension"]),
                    score=filename_score + content_score,
                    match_type=match_type,  # type: ignore[arg-type]
                    snippet=snippet[:MAX_SNIPPET_CHARS],
                    provenance=provenance,
                    modified_at=datetime.fromtimestamp(int(row["mtime_ns"]) / 1_000_000_000, tz=timezone.utc),
                    size=int(row["size_bytes"]),
                    extraction_status=str(row["extraction_status"]),
                    needs_ocr=bool(row["needs_ocr"]),
                    chunk_id=int(state["chunk_id"]) if state["chunk_id"] else None,
                )
            )
        results.sort(key=lambda item: (-item.score, item.relative_path.casefold(), item.file_id))
        return results[:limit]


__all__ = [
    "FTS5UnavailableError",
    "FileIndexStore",
    "MAX_SEARCH_LIMIT",
    "SCHEMA_VERSION",
    "compile_fts_query",
]
