from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.api.app.local_files.indexer import FileIndexer
from services.api.app.local_files.path_policy import ApprovedRoot, PathPolicy
from services.api.app.local_files.storage import FileIndexStore


class Part11IndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "Documents"
        self.root.mkdir()
        self.store = FileIndexStore(Path(self.temp.name) / "file_index.db")
        self.indexer = FileIndexer(self.store, PathPolicy((ApprovedRoot("documents", self.root),)))

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_new_unchanged_modified_and_deleted_reconciliation(self) -> None:
        path = self.root / "notes.txt"
        path.write_text("old searchable phrase", encoding="utf-8")
        first = self.indexer.reconcile()
        self.assertEqual(first.indexed, 1)
        self.assertEqual(self.indexer.reconcile().unchanged, 1)
        path.write_text("new searchable phrase with more bytes", encoding="utf-8")
        changed = self.indexer.reconcile()
        self.assertEqual(changed.indexed, 1)
        self.assertEqual(self.store.search("old searchable"), [])
        self.assertEqual(self.store.search("new searchable")[0].filename, "notes.txt")
        path.unlink()
        self.assertEqual(self.indexer.reconcile().deleted, 1)
        self.assertEqual(self.store.file_count(), 0)

    def test_secret_files_are_excluded_before_extractor(self) -> None:
        (self.root / ".env").write_text("API_KEY=never-read", encoding="utf-8")
        (self.root / "notes.txt").write_text("safe", encoding="utf-8")
        with patch("services.api.app.local_files.indexer.extract_file", wraps=__import__("services.api.app.local_files.extractors", fromlist=["extract_file"]).extract_file) as extractor:
            report = self.indexer.reconcile()
        self.assertEqual(report.discovered, 1)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(extractor.call_args.args[0].name, "notes.txt")

    def test_embedded_secret_is_redacted_before_fts_persistence(self) -> None:
        path = self.root / "config-notes.txt"
        path.write_text("password=super-secret-value\narchitecture notes", encoding="utf-8")
        self.indexer.reconcile()
        self.assertEqual(self.store.search("super secret value"), [])
        self.assertEqual(self.store.search("architecture")[0].filename, path.name)
        row = self.store.file_metadata(str(path.resolve()))
        self.assertEqual(row["redaction_count"], 1)

    def test_parse_race_retries_and_never_exposes_first_parse(self) -> None:
        path = self.root / "race.txt"
        path.write_text("first value", encoding="utf-8")
        from services.api.app.local_files import indexer as module

        real_extract = module.extract_file
        calls = 0

        def changing_extract(target: Path, limits: object):
            nonlocal calls
            calls += 1
            result = real_extract(target, limits)
            if calls == 1:
                target.write_text("second stable value that changed size", encoding="utf-8")
            return result

        with patch.object(module, "extract_file", side_effect=changing_extract):
            self.assertEqual(self.indexer.index_file(path), "indexed")
        self.assertEqual(calls, 2)
        self.assertEqual(self.store.search("first value"), [])
        self.assertEqual(self.store.search("second stable")[0].filename, "race.txt")

    def test_original_files_remain_byte_identical(self) -> None:
        path = self.root / "immutable.md"
        path.write_bytes(b"# immutable\nlocal search")
        before = hashlib.sha256(path.read_bytes()).digest()
        self.indexer.reconcile()
        after = hashlib.sha256(path.read_bytes()).digest()
        self.assertEqual(before, after)

    def test_same_content_metadata_change_avoids_reparse(self) -> None:
        path = self.root / "touch.txt"
        path.write_text("same bytes", encoding="utf-8")
        self.indexer.reconcile()
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        with patch("services.api.app.local_files.indexer.extract_file") as extractor:
            report = self.indexer.reconcile()
        self.assertEqual(report.metadata_refreshed, 1)
        extractor.assert_not_called()
        self.assertEqual(self.store.search("same bytes")[0].filename, "touch.txt")

    def test_unreadable_or_unstable_file_never_leaves_stale_text_searchable(self) -> None:
        path = self.root / "stale.txt"
        path.write_text("stale secret phrase", encoding="utf-8")
        self.indexer.reconcile()
        with patch("services.api.app.local_files.indexer.extract_file", side_effect=PermissionError("denied")):
            path.write_text("changed content with more bytes", encoding="utf-8")
            outcome = self.indexer.index_file(path)
        self.assertEqual(outcome, "error")
        self.assertEqual(self.store.search("stale secret phrase"), [])
        row = self.store.file_metadata(str(path.resolve()))
        self.assertEqual(row["extraction_status"], "stale")


if __name__ == "__main__":
    unittest.main()
