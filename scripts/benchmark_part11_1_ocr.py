"""Part 11.1 OCR benchmark.

Measures the OCR path on synthetic scanned PDFs generated at runtime. No real
user document is ever read.

When Tesseract is absent the script still reports every measurable component --
page rasterization (the PyMuPDF half of OCR cost, which dominates at higher
DPI), the native-first decision overhead, and end-to-end pipeline throughput
with a stub recognizer -- and states plainly that per-page Tesseract
recognition time is UNMEASURED. It never invents numbers.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_part11_1_ocr.py
"""
from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymupdf  # noqa: E402

from services.api.app.local_files import ocr as ocr_module  # noqa: E402
from services.api.app.local_files.extractors import (  # noqa: E402
    ExtractionLimits,
    extract_file,
)

SAMPLE_TEXT = "Invoice 4200 vector database migration quarterly summary"
REPEATS = 5


def make_scanned_pdf(path: Path, pages: int, dpi: int) -> Path:
    raster = pymupdf.open()
    page = raster.new_page(width=595, height=842)
    for line in range(12):
        page.insert_text((40, 60 + line * 24), f"{SAMPLE_TEXT} line {line}", fontsize=13)
    pixmap = page.get_pixmap(dpi=dpi)
    raster.close()

    document = pymupdf.open()
    for _ in range(pages):
        target = document.new_page(width=pixmap.width, height=pixmap.height)
        target.insert_image(pymupdf.Rect(0, 0, pixmap.width, pixmap.height), pixmap=pixmap)
    document.save(path, deflate=True)
    document.close()
    return path


def make_native_pdf(path: Path, pages: int) -> Path:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        for line in range(12):
            page.insert_text((40, 60 + line * 24), f"{SAMPLE_TEXT} p{index} l{line}", fontsize=12)
    document.save(path)
    document.close()
    return path


def timed(function, repeats: int = REPEATS) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), min(samples)


def report(label: str, median_ms: float, best_ms: float, note: str = "") -> None:
    suffix = f"   {note}" if note else ""
    print(f"  {label:<46} median {median_ms:8.1f} ms   best {best_ms:8.1f} ms{suffix}")


def main() -> int:
    status = ocr_module.ocr_status()
    print("=" * 78)
    print("PART 11.1 OCR BENCHMARK (synthetic fixtures only)")
    print("=" * 78)
    print(f"PyMuPDF            : {pymupdf.__version__}")
    print(f"OCR availability   : {status.availability}")
    print(f"tessdata           : {status.tessdata_dir}")
    print(f"installed languages: {', '.join(status.installed_languages) or '(none)'}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("1. Native-first decision overhead (no OCR should occur)")
        native = make_native_pdf(root / "native_10p.pdf", 10)
        limits_off = ExtractionLimits(ocr=ocr_module.OcrSettings(enabled=False))
        median, best = timed(lambda: extract_file(native, limits_off))
        result = extract_file(native, limits_off)
        report("native 10-page PDF, OCR disabled", median, best)
        median, best = timed(
            lambda: extract_file(
                native, ExtractionLimits(ocr=ocr_module.OcrSettings(enabled=True))
            )
        )
        report("native 10-page PDF, OCR enabled", median, best,
               f"used_ocr={result.used_ocr}")
        print(f"     pages OCR'd: {result.ocr_pages}  needs_ocr: {result.needs_ocr}")
        print()

        print("2. Page rasterization cost by DPI (the PyMuPDF half of OCR)")
        scanned = make_scanned_pdf(root / "scanned_1p.pdf", 1, dpi=150)
        for dpi in (72, 150, 200, 300):
            def render(dpi=dpi):
                document = pymupdf.open(scanned)
                try:
                    document[0].get_pixmap(dpi=dpi)
                finally:
                    document.close()

            median, best = timed(render)
            marker = "  <- Bunnelby default" if dpi == ocr_module.DEFAULT_DPI else ""
            report(f"rasterize 1 page @ {dpi} dpi", median, best, marker)
        print()

        print("3. End-to-end pipeline with a stub recognizer (isolates plumbing)")
        stub_status = ocr_module.OcrStatus(
            availability="available", detail="benchmark stub",
            tessdata_dir="stub", installed_languages=("eng",),
            requested_languages=("eng",), usable_languages=("eng",),
        )
        for pages in (1, 5, 10):
            path = make_scanned_pdf(root / f"scanned_{pages}p.pdf", pages, dpi=150)
            limits = ExtractionLimits(
                ocr=ocr_module.OcrSettings(enabled=True, max_ocr_pages=pages)
            )

            def run(path=path, limits=limits):
                with patch.object(
                    ocr_module, "ocr_status", return_value=stub_status
                ), patch.object(ocr_module, "ocr_page_text", return_value=SAMPLE_TEXT):
                    return extract_file(path, limits)

            median, best = timed(run, repeats=3)
            outcome = run()
            report(f"{pages}-page scan, stub OCR", median, best,
                   f"ocr_pages={len(outcome.ocr_pages)}")
            print(f"     file size: {path.stat().st_size / 1024:.0f} KiB")
        print()

        print("4. Real Tesseract recognition")
        if not status.available:
            print("     UNMEASURED - Tesseract is not installed on this machine.")
            print("     Per-page recognition time cannot be reported honestly until")
            print("     the human install step completes. Re-run this script after.")
        else:
            for dpi in (150, 200, 300):
                path = make_scanned_pdf(root / f"live_{dpi}.pdf", 1, dpi=dpi)
                limits = ExtractionLimits(
                    ocr=ocr_module.OcrSettings(enabled=True, dpi=dpi, languages=("eng",))
                )
                median, best = timed(lambda: extract_file(path, limits), repeats=3)
                outcome = extract_file(path, limits)
                report(f"live OCR 1 page @ {dpi} dpi", median, best,
                       f"recovered={len(outcome.text)} chars")
        print()

    print("=" * 78)
    print("Concurrency: OCR runs sequentially, one page at a time, inside the")
    print("existing single indexing path. No parallel OCR pool is introduced;")
    print(f"a document is bounded to {ocr_module.DEFAULT_MAX_OCR_PAGES} OCR pages and")
    print(f"{ocr_module.DEFAULT_DOCUMENT_BUDGET_SECONDS:.0f}s of wall clock.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
