"""Reusable text cleaning helpers for ReviewRadar comments."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.;,])\1+")


def normalize_unicode(text: str) -> str:
    """Normalize unicode forms and decode common HTML entities."""
    return unicodedata.normalize("NFKC", html.unescape(text))


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
    value = normalize_unicode(value)
    value = remove_html(value)
    value = remove_urls(value)
    value = value.lower()
    value = normalize_repeated_punctuation(value)
    value = normalize_whitespace(value)
    return value
