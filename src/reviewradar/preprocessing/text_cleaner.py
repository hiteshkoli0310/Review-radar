"""Reusable text cleaning helpers for ReviewRadar comments."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any


URL_PATTERN = re.compile(r"https?://\S+|(?<!\w)www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.;,])\1+")
MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ðŸ", "ï¸")


def normalize_unicode(text: str) -> str:
    """Normalize unicode forms and decode common HTML entities."""
    return unicodedata.normalize("NFKC", html.unescape(text))


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was decoded as Latin-1 or Windows-1252."""
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    candidates = [text]
    for encoding in ("latin-1", "cp1252"):
        try:
            candidates.append(text.encode(encoding, errors="ignore").decode("utf-8"))
        except UnicodeDecodeError:
            continue

    return min(candidates, key=_mojibake_score)


def remove_unreadable_characters(text: str) -> str:
    """Remove replacement and control characters that are not useful for labeling."""
    readable_characters = []
    for character in text:
        category = unicodedata.category(character)
        if character == "\ufffd":
            continue
        if category.startswith("C") and not character.isspace():
            continue
        readable_characters.append(character)
    return "".join(readable_characters)


def remove_urls(text: str) -> str:
    """Remove HTTP, HTTPS, and www-style URLs from text."""
    return URL_PATTERN.sub(" ", text)


def remove_html(text: str) -> str:
    """Remove simple HTML tags from text."""
    return HTML_TAG_PATTERN.sub(" ", text)


def normalize_repeated_punctuation(text: str) -> str:
    """Collapse repeated punctuation runs to a single punctuation mark."""
    return REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace and trim surrounding whitespace."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: Any) -> str:
    """Clean comment text while preserving sentiment-bearing words."""
    if text is None:
        return ""

    value = str(text)
    value = repair_mojibake(value)
    value = normalize_unicode(value)
    value = remove_unreadable_characters(value)
    value = remove_html(value)
    value = remove_urls(value)
    value = value.lower()
    value = normalize_repeated_punctuation(value)
    value = normalize_whitespace(value)
    return value


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS) + text.count("\ufffd")
