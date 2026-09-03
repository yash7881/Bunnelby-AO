from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.api.app.local_files.indexer import FileIndexer
from services.api.app.local_files.path_policy import ApprovedRoot, PathPolicy
from services.api.app.local_files.storage import FileIndexStore


def elapsed_ms(callable_: object) -> tuple[object, float]:
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000


def warm_latency(search: object, repetitions: int = 50) -> float:
    values: list[float] = []
    for _ in range(repetitions):
        _, duration = elapsed_ms(search)
        values.append(duration)
    return statistics.median(values)


def run(file_count: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="bunnelby-part11-benchmark-") as temp:
        base = Path(temp)
        root = base / "Documents"
        root.mkdir()
        extensions = (".txt", ".md", ".py")
        for index in range(file_count):
            language = (
                "local vector database retrieval"
                if index % 3 == 0
                else "स्थानीय इंटर्नशिप खोज"
                if index % 3 == 1
                else "સ્થાનિક ઇન્ટર્નશિપ શોધ"
            )
            (root / f"Synthetic_Project_{index:05d}{extensions[index % 3]}").write_text(
                f"document {index}\n{language}\nBunnelby deterministic FTS5 benchmark token-{index}",
                encoding="utf-8",
            )

        db_path = base / "file_index.db"
        store = FileIndexStore(db_path)
        indexer = FileIndexer(store, PathPolicy((ApprovedRoot("documents", root),)))
        tracemalloc.start()
        initial, initial_ms = elapsed_ms(indexer.reconcile)
        _, unchanged_ms = elapsed_ms(indexer.reconcile)
        changed = root / f"Synthetic_Project_{file_count // 2:05d}{extensions[(file_count // 2) % 3]}"
        changed.write_text("single changed file unique-updated-token vector database", encoding="utf-8")
        _, changed_ms = elapsed_ms(indexer.reconcile)

        store.search("Project 000", mode="filename", limit=10)
        store.search("vector database", mode="content", limit=10)
        store.search("internship", mode="hybrid", limit=10)
        filename_ms = warm_latency(lambda: store.search("Project 000", mode="filename", limit=10))
        content_ms = warm_latency(lambda: store.search("vector database", mode="content", limit=10))
        hybrid_ms = warm_latency(lambda: store.search("Bunnelby benchmark", mode="hybrid", limit=10))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        indexed = store.file_count()
        store.close()
        return {
            "synthetic_files": file_count,
            "indexed_files": indexed,
            "initial_index_ms": round(initial_ms, 3),
            "unchanged_reconcile_ms": round(unchanged_ms, 3),
            "single_changed_reconcile_ms": round(changed_ms, 3),
            "warm_filename_median_ms_50": round(filename_ms, 3),
            "warm_content_median_ms_50": round(content_ms, 3),
            "warm_hybrid_median_ms_50": round(hybrid_ms, 3),
            "index_db_bytes": db_path.stat().st_size,
            "python_tracemalloc_peak_bytes": peak,
            "initial_report": initial.__dict__,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-only Part 11 benchmark")
    parser.add_argument("--files", type=int, default=1500)
    arguments = parser.parse_args()
    if arguments.files < 100:
        parser.error("--files must be at least 100")
    print(json.dumps(run(arguments.files), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
