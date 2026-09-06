from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.api.app.local_files.models import ChunkInput, IndexedFileInput
from services.api.app.local_files.storage import FileIndexStore, FTS5UnavailableError


def fixture(path: str, text: str, *, root: str = "documents") -> IndexedFileInput:
    return IndexedFileInput(
        canonical_path=f"C:/synthetic/{path}",
        root_alias=root,
        relative_path=path,
        filename=Path(path).name,
        extension=Path(path).suffix.casefold(),
        size_bytes=len(text.encode()),
        mtime_ns=123,
        content_hash="abc",
        file_type="text",
        parser_name="test",
        parser_version="1",
        extraction_status="indexed",
        needs_ocr=False,
        redaction_count=0,
        chunks=(ChunkInput(0, text, text_hash="hash", line_start=1, line_end=2),),
    )


class Part11StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "file_index.db"
        self.store = FileIndexStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_schema_uses_fts5_and_is_separate_persistent_database(self) -> None:
        self.assertEqual(self.store.schema_version, 1)
        tables = self.store.table_names()
        self.assertIn("files", tables)
        self.assertIn("chunks", tables)
        self.assertIn("filename_fts", tables)
        self.assertIn("content_fts", tables)
        self.store.replace_file(fixture("Resume_Parth.pdf", "vector databases"))
        self.store.close()
        self.store = FileIndexStore(self.db)
        self.assertEqual(self.store.file_count(), 1)

    def test_english_hindi_gujarati_and_filename_substring_search(self) -> None:
        self.store.replace_file(fixture("Resume_Parth.pdf", "Built a vector database retrieval system."))
        self.store.replace_file(fixture("hindi_notes.txt", "यह वेक्टर डेटाबेस के बारे में नोट है।"))
        self.store.replace_file(fixture("gujarati_notes.txt", "આ વેક્ટર ડેટાબેઝ વિશે નોંધ છે."))
        self.assertEqual(self.store.search("Parth", mode="filename")[0].filename, "Resume_Parth.pdf")
        self.assertEqual(self.store.search("vector database", mode="content")[0].filename, "Resume_Parth.pdf")
        self.assertEqual(self.store.search("वेक्टर डेटाबेस", mode="content")[0].filename, "hindi_notes.txt")
        self.assertEqual(self.store.search("વેક્ટર ડેટાબેઝ", mode="content")[0].filename, "gujarati_notes.txt")

    def test_replace_is_atomic_and_delete_removes_fts_rows(self) -> None:
        first = self.store.replace_file(fixture("notes.txt", "old phrase"))
        self.store.replace_file(fixture("notes.txt", "new phrase"))
        self.assertEqual(self.store.search("old phrase"), [])
        self.assertEqual(self.store.search("new phrase")[0].file_id, first)
        self.store.delete_file("C:/synthetic/notes.txt")
        self.assertEqual(self.store.search("new phrase"), [])

    def test_filters_and_bounded_limit(self) -> None:
        self.store.replace_file(fixture("a.txt", "shared needle", root="documents"))
        self.store.replace_file(fixture("b.md", "shared needle", root="desktop"))
        results = self.store.search("needle", root_aliases=("documents",), extensions=(".txt",), limit=1)
        self.assertEqual([item.filename for item in results], ["a.txt"])

    def test_fts_syntax_is_compiled_as_literal_terms(self) -> None:
        self.store.replace_file(fixture("query-notes.txt", "quotes operators parentheses SQL select drop table"))
        for query in ('" OR *', "a-b", "(select OR drop)", "NEAR(one two)", "'; DROP TABLE files; --"):
            with self.subTest(query=query):
                self.store.search(query)
        self.assertEqual(self.store.file_count(), 1)


if __name__ == "__main__":
    unittest.main()
