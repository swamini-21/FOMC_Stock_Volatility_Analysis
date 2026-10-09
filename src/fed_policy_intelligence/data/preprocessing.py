"""Conservative, auditable text cleaning for Federal Reserve documents."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned text and diagnostics needed for auditability."""

    cleaned_text: str
    original_length: int
    cleaned_length: int
    characters_removed: int
    removal_ratio: float
    matched_rules: tuple[str, ...]


_CONTROL_CHARACTERS = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")

# Rules are anchored to document starts or complete lines. This intentionally
# avoids the notebook's unanchored `Conference Call.*?on <date>` expression.
_BOILERPLATE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "transcript_disclosure_preamble",
        re.compile(
            r"\A\s*TRANSCRIPT\s+FEDERAL OPEN MARKET COMMITTEE MEETING"
            r"[\s\S]*?All information deleted in this manner is exempt from disclosure "
            r"under applicable provisions of the Freedom of Information Act\.\s*",
            re.IGNORECASE,
        ),
    ),
    (
        "staff_statements_heading",
        re.compile(r"(?im)^\s*Staff Statements Appended to the Transcript\s*$"),
    ),
    (
        "content_modified_line",
        re.compile(r"(?im)^\s*Content\s+last\s+modified\s+\d{1,2}/\d{1,2}/\d{2,4}\.?\s*$"),
    ),
    (
        "conference_call_heading",
        re.compile(
            r"(?im)^\s*Conference Call of (?:the\s+)?Federal Open Market Committee "
            r"on [A-Za-z]+\s+\d{1,2},\s+\d{4}"
            r"(?:,?\s+at\s+\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.))?\s*$"
        ),
    ),
    (
        "meeting_heading",
        re.compile(
            r"(?im)^\s*Meeting of (?:the\s+)?Federal Open Market Committee "
            r"[A-Za-z]+\s+\d{1,2}(?:[-–]\d{1,2})?,\s+\d{4}\s*$"
        ),
    ),
)


def clean_document_text(value: object) -> CleaningResult:
    """Clean text conservatively and return per-rule diagnostics."""

    original = value if isinstance(value, str) else ""
    cleaned = unicodedata.normalize("NFKC", html.unescape(original))
    matched: list[str] = []

    for name, pattern in _BOILERPLATE_RULES:
        cleaned, count = pattern.subn("", cleaned)
        if count:
            matched.append(name)

    cleaned = cleaned.replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "")
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _CONTROL_CHARACTERS.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    original_length = len(original)
    cleaned_length = len(cleaned)
    removed = max(original_length - cleaned_length, 0)
    ratio = removed / original_length if original_length else 0.0
    return CleaningResult(
        cleaned_text=cleaned,
        original_length=original_length,
        cleaned_length=cleaned_length,
        characters_removed=removed,
        removal_ratio=ratio,
        matched_rules=tuple(matched),
    )

