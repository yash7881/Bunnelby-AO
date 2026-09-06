from __future__ import annotations

import contextlib
import contextvars
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from ..session_service import new_result_set_id
from ..untrusted_content import UntrustedContent, wrap
from .indexer import FileIndexer, ReconcileReport
from .models import FileSearchResult
from .path_policy import PathPolicy, default_windows_roots, is_secret_file
from .storage import FileIndexStore


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "database" / "file_index.db"
_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("file_search_session", default="anonymous")


@contextlib.contextmanager
def execution_session(session_id: str | None) -> Iterator[None]:
    token = _SESSION_ID.set(session_id or "anonymous")
    try:
        yield
    finally:
        _SESSION_ID.reset(token)


class ResultSetRegistry:
    """Bounded process-local handles; stores IDs only, never document text."""

    def __init__(self, max_sets: int = 256) -> None:
        self.max_sets = max_sets
        self._lock = threading.Lock()
        self._sets: dict[tuple[str, str], tuple[int, ...]] = {}
        self._order: list[tuple[str, str]] = []

    def record(self, session_id: str, result_set_id: str, file_ids: Sequence[int]) -> None:
        key = (session_id, result_set_id)
        with self._lock:
            self._sets[key] = tuple(dict.fromkeys(int(value) for value in file_ids))
            self._order.append(key)
            while len(self._order) > self.max_sets:
                oldest = self._order.pop(0)
                self._sets.pop(oldest, None)

    def resolve(self, session_id: str, result_set_id: str) -> tuple[int, ...] | None:
        with self._lock:
            return self._sets.get((session_id, result_set_id))


def _claimed_extraction_method(result: object, index: int) -> str | None:
    """Extraction method the result asserts for its index-th hit, if any."""
    metadata = dict(getattr(result, "spoken_metadata", {}) or {})
    methods = metadata.get("extraction_methods")
    if isinstance(methods, (list, tuple)) and index < len(methods):
        value = methods[index]
        return str(value) if value is not None else None
    return None


@dataclass(frozen=True)
class SearchEnvelope:
    result_set_id: str
    results: tuple[FileSearchResult, ...]
    untrusted_snippets: tuple[UntrustedContent, ...]
    refinement_missing: bool = False


class LocalFileSearchService:
    def __init__(
        self,
        store: FileIndexStore,
        policy: PathPolicy,
        *,
        result_sets: ResultSetRegistry | None = None,
        limits: object | None = None,
    ) -> None:
        self.store = store
        self.policy = policy
        # `limits` carries Part 11.1 OcrSettings when a caller wants explicit
        # control; None keeps the production default of environment config.
        self.indexer = (
            FileIndexer(store, policy, limits=limits)  # type: ignore[arg-type]
            if limits is not None
            else FileIndexer(store, policy)
        )
        self.result_sets = result_sets or ResultSetRegistry()

    def reconcile(self, root_aliases: Sequence[str] = ()) -> ReconcileReport:
        return self.indexer.reconcile(root_aliases or None)

    def ocr_status(self) -> dict[str, object]:
        """Report OCR readiness (Part 11.1) without running any OCR.

        Surfaced so the absence of Tesseract is visible and actionable rather
        than silently degrading scanned PDFs to metadata-only.
        """
        from . import ocr as ocr_module

        settings = getattr(self.indexer, "limits", None)
        configured = getattr(settings, "ocr", None) or ocr_module.settings_from_env()
        status = ocr_module.ocr_status(configured)
        payload = status.as_dict()
        payload["remediation"] = list(status.remediation)
        return payload

    def search_request(self, request: object) -> SearchEnvelope:
        session_id = _SESSION_ID.get()
        within = getattr(request, "within_result_set_id", None)
        file_ids: tuple[int, ...] = ()
        if within:
            resolved = self.result_sets.resolve(session_id, str(within))
            if resolved is None:
                result_set_id = new_result_set_id("file-search")
                self.result_sets.record(session_id, result_set_id, ())
                return SearchEnvelope(result_set_id, (), (), refinement_missing=True)
            file_ids = resolved
            if not file_ids:
                return SearchEnvelope(new_result_set_id("file-search"), (), ())
        after = getattr(request, "modified_after", None)
        before = getattr(request, "modified_before", None)
        results = tuple(
            self.store.search(
                getattr(request, "query"),
                mode=getattr(request, "search_mode"),
                limit=getattr(request, "limit"),
                root_aliases=getattr(request, "root_scope"),
                extensions=getattr(request, "extensions"),
                modified_after_ns=int(after.timestamp() * 1_000_000_000) if isinstance(after, datetime) else None,
                modified_before_ns=int(before.timestamp() * 1_000_000_000) if isinstance(before, datetime) else None,
                file_ids=file_ids,
            )
        )
        result_set_id = new_result_set_id("file-search")
        self.result_sets.record(session_id, result_set_id, [item.file_id for item in results])
        snippets = tuple(
            wrap(
                "file",
                item.snippet,
                provenance=f"{item.root_alias}:{item.relative_path}:{dict(item.provenance)}",
                source_id=f"{result_set_id}:{item.file_id}",
                limit=500,
            )
            for item in results
        )
        return SearchEnvelope(result_set_id, results, snippets)

    def verify_envelope(self, request: object, result: object) -> tuple[bool, str, dict[str, object]]:
        metadata = dict(getattr(result, "spoken_metadata", {}) or {})
        ids = tuple(metadata.get("file_ids", ()))
        chunk_ids = tuple(metadata.get("chunk_ids", ()))
        if getattr(result, "action_type", None) != "file_search" or getattr(result, "approval", None) is not None:
            return False, "file search produced a non-read action or approval", metadata
        if len(ids) != metadata.get("result_count") or len(ids) > getattr(request, "limit"):
            return False, "result count was inconsistent or exceeded the request limit", metadata
        for index, file_id in enumerate(ids):
            row = self.store.file_row(int(file_id))
            if row is None:
                return False, "a referenced file id no longer exists", metadata
            decision = self.policy.check_file(str(row["canonical_path"]))
            if not decision.allowed or decision.root_alias != row["root_alias"] or is_secret_file(Path(str(row["canonical_path"]))):
                return False, "a result failed approved-root or exclusion policy", metadata
            if index < len(chunk_ids) and chunk_ids[index] is not None:
                chunk = self.store.chunk_row(int(chunk_ids[index]))
                if chunk is None or int(chunk["file_id"]) != int(file_id):
                    return False, "chunk provenance did not match its file", metadata
                # Part 11.1: a result may not claim an extraction method the
                # indexed chunk does not actually have. This stops OCR
                # provenance being asserted for text that was never OCR'd
                # (and vice versa).
                claimed = _claimed_extraction_method(result, index)
                if claimed is not None and claimed != str(chunk["extraction_method"]):
                    return (
                        False,
                        "a result claimed an extraction method the indexed chunk does not have",
                        metadata,
                    )
        return True, f"verified {len(ids)} local, read-only result(s)", metadata


_DEFAULT_SERVICE: LocalFileSearchService | None = None
_DEFAULT_LOCK = threading.Lock()


def default_service() -> LocalFileSearchService:
    """Lazy construction opens the index but never initiates a personal-folder scan."""
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        if _DEFAULT_SERVICE is None:
            roots = default_windows_roots()
            _DEFAULT_SERVICE = LocalFileSearchService(
                FileIndexStore(DEFAULT_INDEX_PATH), PathPolicy(roots)
            )
        return _DEFAULT_SERVICE


__all__ = [
    "DEFAULT_INDEX_PATH",
    "LocalFileSearchService",
    "ResultSetRegistry",
    "SearchEnvelope",
    "default_service",
    "execution_session",
]
