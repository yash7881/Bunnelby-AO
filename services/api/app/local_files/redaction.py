from __future__ import annotations

import re
from typing import Final


REDACTION_MARKER: Final[str] = "[REDACTED_SECRET]"

_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL | re.IGNORECASE),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(
        r"(?im)^(\s*(?:password|passwd|pwd|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|github[_-]?token)\s*[:=]\s*)"
        r"(?:['\"])?([^\s'\"]{8,})(?:['\"])?\s*$"
    ),
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Conservatively remove credential-shaped values before persistence."""
    result = text
    total = 0
    for pattern in _PATTERNS:
        if pattern.groups:
            result, count = pattern.subn(lambda match: match.group(1) + REDACTION_MARKER, result)
        else:
            result, count = pattern.subn(REDACTION_MARKER, result)
        total += count
    return result, total


__all__ = ["REDACTION_MARKER", "redact_secrets"]
