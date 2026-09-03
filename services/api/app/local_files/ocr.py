from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

logger = logging.getLogger(__name__)

# Part 11.1: local OCR fallback for scanned / image-based PDF pages.
#
# This module is deliberately the ONLY new moving part. It plugs into the
# OCR-ready seam Part 11 already left behind:
#
#   extractors._extract_pdf  ->  per-page native text
#                            ->  page_needs_ocr()          (this module)
#                            ->  ocr_page_text()           (this module)
#                            ->  ExtractedUnit(extraction_method="pymupdf_ocr")
#                            ->  redaction -> chunking -> file_index -> verifier
#                            ->  UNTRUSTED_EXTERNAL_DATA
#
# Nothing downstream changes: OCR text is just another ExtractedUnit, so it
# inherits Part 11's redaction, provenance, bounding, verification and untrusted
# handling for free. There is no parallel OCR store and no second search engine.
#
# Two hard rules:
#   1. Native text ALWAYS wins. OCR runs only for pages that genuinely lack it.
#   2. If OCR cannot run, the page stays honestly marked needs_ocr. We never
#      fabricate text and never claim a page was indexed when it was not.

OcrAvailability = Literal[
    "available",
    "disabled",
    "tesseract_missing",
    "languages_missing",
    "unsupported",
]

DEFAULT_LANGUAGES: Final[tuple[str, ...]] = ("eng", "hin", "guj")

# tessdata_fast is Tesseract's own documented recommendation ("Most users will
# want tessdata_fast and that is what will be shipped as part of Linux
# distributions"). On a Ryzen 5 / 16 GB laptop with no OCR GPU path, the speed
# difference matters more than tessdata_best's marginal accuracy gain.
RECOMMENDED_TESSDATA_VARIANT: Final[str] = "tessdata_fast"

# 200 DPI is a deliberate middle ground: 72 (PyMuPDF's default) loses accuracy
# on small scanned type, while 300+ roughly doubles pixel count and OCR time for
# little gain on ordinary documents.
DEFAULT_DPI: Final[int] = 200
MIN_DPI: Final[int] = 72
MAX_DPI: Final[int] = 400

# A page with fewer than this many alphanumeric characters of NATIVE text is
# treated as image-only / insufficiently extracted.
DEFAULT_MIN_PAGE_CHARS: Final[int] = 12

DEFAULT_MAX_OCR_PAGES: Final[int] = 20
DEFAULT_DOCUMENT_BUDGET_SECONDS: Final[float] = 60.0
DEFAULT_PAGE_TIMEOUT_SECONDS: Final[float] = 25.0

# OCR text is machine-read, not authored. Chunks carry a lower confidence so
# ranking and any future consumer can tell OCR text from native text.
OCR_CONFIDENCE: Final[float] = 0.6
OCR_EXTRACTION_METHOD: Final[str] = "pymupdf_ocr"

_LANG_LIST_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'List of available languages in "(.+)"'
)

# Standard install locations for the community UB Mannheim Windows build. These
# are probed read-only; nothing here installs, elevates or edits PATH.
_WINDOWS_CANDIDATES: Final[tuple[str, ...]] = (
    r"C:\Program Files\Tesseract-OCR\tessdata",
    r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
)

INSTALL_STEPS: Final[tuple[str, ...]] = (
    "Install the community Windows build of Tesseract 5.x from "
    "https://github.com/UB-Mannheim/tesseract/wiki (the official project does "
    "not publish a Windows binary itself).",
    "During setup, open 'Additional language data' and select Hindi and "
    "Gujarati, or add them afterwards by copying the traineddata files.",
    "If adding languages manually, download hin.traineddata and "
    "guj.traineddata from https://github.com/tesseract-ocr/tessdata_fast and "
    r"place them in C:\Program Files\Tesseract-OCR\tessdata",
    r"Verify: & 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs "
    "(expect eng, guj and hin in the output).",
    "Bunnelby finds tessdata automatically. If your install is elsewhere, set "
    "BUNNELBY_TESSDATA_DIR to that tessdata folder; no PATH change is needed.",
)


class OcrError(RuntimeError):
    """Base class for OCR problems that must never abort a whole index run."""


class OcrUnavailableError(OcrError):
    """OCR cannot run at all (disabled, Tesseract absent, languages absent)."""


class OcrFailedError(OcrError):
    """OCR was attempted for a page and did not produce a usable result."""


@dataclass(frozen=True)
class OcrSettings:
    """Bounded OCR configuration. Every limit is deliberately conservative."""

    enabled: bool = True
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    dpi: int = DEFAULT_DPI
    tessdata_dir: str | None = None
    min_page_chars: int = DEFAULT_MIN_PAGE_CHARS
    max_ocr_pages: int = DEFAULT_MAX_OCR_PAGES
    document_budget_seconds: float = DEFAULT_DOCUMENT_BUDGET_SECONDS
    page_timeout_seconds: float = DEFAULT_PAGE_TIMEOUT_SECONDS

    @property
    def language_spec(self) -> str:
        """Tesseract multi-language spec, e.g. 'eng+hin+guj'."""
        return "+".join(self.languages)

    def bounded(self) -> "OcrSettings":
        from dataclasses import replace

        return replace(
            self,
            dpi=max(MIN_DPI, min(MAX_DPI, int(self.dpi))),
            max_ocr_pages=max(0, int(self.max_ocr_pages)),
            min_page_chars=max(0, int(self.min_page_chars)),
        )


@dataclass(frozen=True)
class OcrStatus:
    """Structured, reportable OCR readiness."""

    availability: OcrAvailability
    detail: str
    tessdata_dir: str | None = None
    installed_languages: tuple[str, ...] = ()
    requested_languages: tuple[str, ...] = ()
    missing_languages: tuple[str, ...] = ()
    usable_languages: tuple[str, ...] = ()
    remediation: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.availability == "available"

    def language_spec(self) -> str:
        return "+".join(self.usable_languages)

    def as_dict(self) -> dict[str, object]:
        """Audit/report-safe summary. Contains no document content."""
        return {
            "availability": self.availability,
            "detail": self.detail,
            "tessdata_dir": self.tessdata_dir,
            "installed_languages": list(self.installed_languages),
            "requested_languages": list(self.requested_languages),
            "missing_languages": list(self.missing_languages),
            "usable_languages": list(self.usable_languages),
        }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def settings_from_env() -> OcrSettings:
    """Build settings from optional environment configuration.

    Every variable is optional; the defaults are production-safe. Nothing here
    writes to the environment.
    """
    raw_languages = os.getenv("BUNNELBY_OCR_LANGUAGES", "").strip()
    languages = (
        tuple(part for part in re.split(r"[+,\s]+", raw_languages) if part)
        if raw_languages
        else DEFAULT_LANGUAGES
    )
    tessdata = os.getenv("BUNNELBY_TESSDATA_DIR", "").strip() or None
    return OcrSettings(
        enabled=_env_flag("BUNNELBY_OCR_ENABLED", True),
        languages=languages,
        dpi=_env_int("BUNNELBY_OCR_DPI", DEFAULT_DPI),
        tessdata_dir=tessdata,
        min_page_chars=_env_int("BUNNELBY_OCR_MIN_PAGE_CHARS", DEFAULT_MIN_PAGE_CHARS),
        max_ocr_pages=_env_int("BUNNELBY_OCR_MAX_PAGES", DEFAULT_MAX_OCR_PAGES),
        document_budget_seconds=_env_float(
            "BUNNELBY_OCR_DOCUMENT_BUDGET_SECONDS", DEFAULT_DOCUMENT_BUDGET_SECONDS
        ),
        page_timeout_seconds=_env_float(
            "BUNNELBY_OCR_PAGE_TIMEOUT_SECONDS", DEFAULT_PAGE_TIMEOUT_SECONDS
        ),
    ).bounded()


# --------------------------------------------------------------------------- #
# Tesseract discovery -- read-only, cached, no global mutation
# --------------------------------------------------------------------------- #

_discovery_lock = threading.Lock()
_discovery_cache: dict[str, str | None] = {}


def reset_discovery_cache() -> None:
    """Clear cached discovery. Used by tests and after an install."""
    with _discovery_lock:
        _discovery_cache.clear()


def _probe_tesseract_langs() -> str | None:
    """Ask an installed tesseract where its tessdata is, without shell=True."""
    executable = shutil.which("tesseract")
    if not executable:
        return None
    try:
        import subprocess

        completed = subprocess.run(  # nosec B603 - fixed argv, resolved via which()
            [executable, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        logger.debug("tesseract --list-langs probe failed: %s", exc)
        return None
    match = _LANG_LIST_PATTERN.search(f"{completed.stdout}\n{completed.stderr}")
    if match:
        candidate = match.group(1).strip().rstrip("/\\")
        if Path(candidate).is_dir():
            return candidate
    # Fall back to the conventional layout beside the executable.
    sibling = Path(executable).resolve().parent / "tessdata"
    return str(sibling) if sibling.is_dir() else None


def discover_tessdata(explicit: str | None = None) -> str | None:
    """Locate Tesseract's tessdata folder, or return None.

    Resolution order, most explicit first:

    1. the value passed in (from OcrSettings / BUNNELBY_TESSDATA_DIR)
    2. TESSDATA_PREFIX, if the user already set it
    3. `tesseract --list-langs`, which reports its own tessdata path
    4. the standard Windows install locations
    5. PyMuPDF's own get_tessdata(), as a last resort

    Deliberately does NOT set TESSDATA_PREFIX or touch PATH: the resolved
    directory is passed to PyMuPDF per call via its `tessdata=` parameter.
    """
    if explicit:
        return explicit if Path(explicit).is_dir() else None

    cache_key = "default"
    with _discovery_lock:
        if cache_key in _discovery_cache:
            return _discovery_cache[cache_key]

    resolved: str | None = None

    prefix = os.getenv("TESSDATA_PREFIX", "").strip()
    if prefix and Path(prefix).is_dir():
        resolved = prefix

    if resolved is None:
        resolved = _probe_tesseract_langs()

    if resolved is None:
        for candidate in _WINDOWS_CANDIDATES:
            if Path(candidate).is_dir():
                resolved = candidate
                break

    if resolved is None:
        try:
            import pymupdf

            candidate = pymupdf.get_tessdata()
            if candidate and Path(str(candidate)).is_dir():
                resolved = str(candidate)
        except Exception as exc:
            logger.debug("PyMuPDF get_tessdata() found nothing: %s", exc)

    with _discovery_lock:
        _discovery_cache[cache_key] = resolved
    return resolved


def installed_languages(tessdata_dir: str | None) -> tuple[str, ...]:
    """Language codes present as <code>.traineddata in tessdata_dir."""
    if not tessdata_dir:
        return ()
    directory = Path(tessdata_dir)
    if not directory.is_dir():
        return ()
    try:
        return tuple(
            sorted(item.stem for item in directory.glob("*.traineddata") if item.is_file())
        )
    except OSError as exc:
        logger.warning("Could not list tessdata in %s: %s", tessdata_dir, exc)
        return ()


def ocr_status(settings: OcrSettings | None = None) -> OcrStatus:
    """Report whether OCR can run, and exactly what is missing if not."""
    active = (settings or settings_from_env()).bounded()
    requested = tuple(active.languages)

    if not active.enabled:
        return OcrStatus(
            availability="disabled",
            detail="OCR is disabled by configuration (BUNNELBY_OCR_ENABLED=0).",
            requested_languages=requested,
        )

    tessdata = discover_tessdata(active.tessdata_dir)
    if not tessdata:
        return OcrStatus(
            availability="tesseract_missing",
            detail=(
                "Tesseract is not installed, so scanned pages stay marked "
                "needs_ocr instead of being read."
            ),
            requested_languages=requested,
            remediation=INSTALL_STEPS,
        )

    present = installed_languages(tessdata)
    usable = tuple(code for code in requested if code in present)
    missing = tuple(code for code in requested if code not in present)

    if not usable:
        return OcrStatus(
            availability="languages_missing",
            detail=(
                f"Tesseract was found at {tessdata} but none of the requested "
                f"languages ({', '.join(requested)}) are installed."
            ),
            tessdata_dir=tessdata,
            installed_languages=present,
            requested_languages=requested,
            missing_languages=missing,
            remediation=INSTALL_STEPS,
        )

    detail = f"OCR ready with {'+'.join(usable)} from {tessdata}."
    if missing:
        # Partial availability is still usable: OCR proceeds with what exists
        # rather than refusing the whole page.
        detail += f" Missing language data: {', '.join(missing)}."
    return OcrStatus(
        availability="available",
        detail=detail,
        tessdata_dir=tessdata,
        installed_languages=present,
        requested_languages=requested,
        missing_languages=missing,
        usable_languages=usable,
        remediation=INSTALL_STEPS if missing else (),
    )


# --------------------------------------------------------------------------- #
# Per-page OCR decision and execution
# --------------------------------------------------------------------------- #


def meaningful_characters(text: str) -> int:
    """Count alphanumeric characters, Unicode-aware (Devanagari/Gujarati too)."""
    return sum(1 for character in text or "" if character.isalnum())


def page_needs_ocr(native_text: str, settings: OcrSettings | None = None) -> bool:
    """True when a single page's NATIVE text is too sparse to trust.

    Per-page by design. Part 11 decided needs_ocr for the whole document, which
    would have forced an all-or-nothing choice on a mixed PDF; scoring each page
    lets good native pages be kept verbatim while only image pages are OCR'd.
    """
    active = (settings or settings_from_env()).bounded()
    return meaningful_characters(native_text) < active.min_page_chars


class OcrBudget:
    """Wall-clock budget for one document's OCR work.

    Bounds how long a single pathological PDF can occupy the indexer. Checked
    between pages: pages that do not fit stay honestly marked needs_ocr.
    """

    def __init__(self, settings: OcrSettings) -> None:
        self._settings = settings
        self._started = time.monotonic()
        self.pages_attempted = 0
        self.pages_succeeded = 0
        self.pages_failed = 0
        self.seconds_spent = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def allows_another_page(self) -> bool:
        if self.pages_attempted >= self._settings.max_ocr_pages:
            return False
        return self.elapsed < self._settings.document_budget_seconds

    def record(self, *, succeeded: bool, seconds: float) -> None:
        self.pages_attempted += 1
        self.seconds_spent += seconds
        if succeeded:
            self.pages_succeeded += 1
        else:
            self.pages_failed += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "pages_attempted": self.pages_attempted,
            "pages_succeeded": self.pages_succeeded,
            "pages_failed": self.pages_failed,
            "ocr_seconds": round(self.seconds_spent, 3),
        }


def ocr_page_text(
    page: object,
    settings: OcrSettings,
    status: OcrStatus,
) -> str:
    """OCR one already-open PyMuPDF page and return its text.

    Uses PyMuPDF's native Tesseract integration, passing `tessdata` explicitly
    so no environment variable or PATH entry has to be set.

    `full=True` is correct here: this function is only ever called for a page
    whose native text was already judged insufficient, so there is no native
    text worth preserving in the text page and OCRing everything is what
    actually recovers the content.

    Raises OcrUnavailableError / OcrFailedError; never returns fabricated text.
    """
    if not status.available:
        raise OcrUnavailableError(status.detail)

    language = status.language_spec() or "eng"
    try:
        textpage = page.get_textpage_ocr(
            flags=0,
            language=language,
            dpi=settings.dpi,
            full=True,
            tessdata=status.tessdata_dir,
        )
    except Exception as exc:
        raise OcrFailedError(f"OCR call failed: {type(exc).__name__}") from exc

    try:
        text = str(page.get_text("text", textpage=textpage) or "")
    except Exception as exc:
        raise OcrFailedError(f"OCR text read failed: {type(exc).__name__}") from exc
    finally:
        # PyMuPDF TextPage holds native memory; release it deterministically.
        closer = getattr(textpage, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.debug("OCR textpage close failed", exc_info=True)

    if not text.strip():
        raise OcrFailedError("OCR produced no text for this page")
    return text


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_LANGUAGES",
    "INSTALL_STEPS",
    "OCR_CONFIDENCE",
    "OCR_EXTRACTION_METHOD",
    "RECOMMENDED_TESSDATA_VARIANT",
    "OcrAvailability",
    "OcrBudget",
    "OcrError",
    "OcrFailedError",
    "OcrSettings",
    "OcrStatus",
    "OcrUnavailableError",
    "discover_tessdata",
    "installed_languages",
    "meaningful_characters",
    "ocr_page_text",
    "ocr_status",
    "page_needs_ocr",
    "reset_discovery_cache",
    "settings_from_env",
]
