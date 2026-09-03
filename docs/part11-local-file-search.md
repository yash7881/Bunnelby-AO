# Part 11 local file search

Part 11 is a local-only, deterministic retrieval subsystem. The original filesystem remains authoritative; `database/file_index.db` is disposable derived data and has its own schema initializer (no Alembic migration and no `ao.db` changes).

## Safety and lifecycle

- Trusted roots come from Windows Known Folder APIs and are represented only by `desktop`, `documents`, and `downloads` aliases. Tests inject synthetic aliases/roots.
- Discovery skips secret names before opening them, refuses UNC paths and link/junction/reparse traversal, and canonicalizes every candidate before indexing.
- Extraction is bounded by file, text, page, row, cell, Office-package, chunk, result, and snippet limits. Original files are only opened for reading.
- Native PDF text is indexed with page provenance. A low/no-text PDF is marked `needs_ocr`; Part 11 does not run OCR.
- Extracted text is locally redacted before transactional storage. Search snippets are untrusted external data and file-derived memory is rewrapped on re-entry.
- A deterministic full reconciliation handles create/change/rename-as-delete-plus-create/delete and is the recovery authority after missed events.

No watcher ships in Part 11. Windows notification buffers can overflow, so a watcher cannot replace reconciliation; adding a background lifecycle, debounce, shutdown, and overflow recovery would increase failure surface without improving the acceptance path. A later watcher can call `FileIndexer.index_file()` for every event and `FileIndexer.reconcile()` after overflow/startup, preserving the same path policy.

## Retrieval

`filename_fts` uses the FTS5 trigram tokenizer for substring matching. `content_fts` uses `unicode61` without English stemming for English, Hindi, and Gujarati. User text is compiled into quoted literal terms before `MATCH`; raw FTS grammar is never accepted. Results aggregate at file level with a filename/path boost plus content BM25, deterministic tie-breaking, bounded snippets, and chunk provenance.

`LocalFileSearchService.default_service()` opens the index lazily but never scans. The first real personal-folder reconciliation must be initiated later with explicit human authorization. All automated fixtures and the benchmark use temporary synthetic roots.

## OCR integration point

Part 11.1 can consume indexed files where `files.needs_ocr=1`, run an explicitly configured local OCR extractor per page, then feed returned `ExtractedUnit` values through the existing redaction, chunking, race check, and atomic `replace_file()` path. It must not bypass root validation or write OCR output back to the source PDF.
