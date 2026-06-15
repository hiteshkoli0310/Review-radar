"""Heuristic spam and low-quality comment flags."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from reviewradar.preprocessing.text_cleaner import URL_PATTERN


DELETED_MARKERS = {
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
    "comment deleted",
    "comment removed",
}

PROMOTIONAL_MARKERS = {
    "check out",
    "documenting the start",
    "http",
    "subscribe",
    "subscribed",
    "youtube.com/shorts",
    "like share",
    "check my channel",
    "visit my channel",
    "follow me",
    "giveaway",
    "promo code",
    "whatsapp",
    "telegram",
}

SOLICITATION_MARKERS = {
    "plz",
    "kalka dikao",
    "phone kodi",
    "please give",
    "please send",
    "please dedo",
    "please share",
    "dedo",
    "kudunga",
    "madi bro",
}

REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{5,}", flags=re.IGNORECASE)


def is_empty_comment(text: Any) -> bool:
    """Return True when a comment is missing or has no non-whitespace text."""
    if text is None or pd.isna(text):
        return True
    return not str(text).strip()


def is_deleted_comment(text: Any) -> bool:
    """Return True when a comment looks deleted or removed by YouTube/user."""
    if text is None or pd.isna(text):
        return False
    normalized = str(text).strip().lower()
    return normalized in DELETED_MARKERS


def is_short_comment(text: Any, minimum_length: int = 3) -> bool:
    """Return True when trimmed comment text is shorter than the minimum length."""
    if text is None or pd.isna(text):
        return True
    return len(str(text).strip()) < minimum_length


def is_emoji_only(text: str) -> bool:
    """Return True when text contains only emoji/symbols and no alphanumeric chars."""
    return _is_emoji_only(text)


def is_single_word(text: Any) -> bool:
    """Return True when trimmed text contains exactly one word."""
    if text is None or pd.isna(text):
        return False
    words = str(text).strip().split()
    return len(words) == 1


def _is_solicitation(text: str) -> bool:
    return any(marker in text for marker in SOLICITATION_MARKERS)


def is_spam_comment(text: Any) -> bool:
    """Return True when a comment matches simple spam heuristics."""
    if text is None or pd.isna(text):
        return False

    value = str(text).strip()
    if not value:
        return False

    normalized = value.lower()
    return any(
        [
            _has_many_urls(value),
            _is_link_only_or_link_promo(value),
            _has_promotional_text(normalized),
            _is_emoji_only(value),
            _has_excessive_repeated_characters(value),
            _is_solicitation(normalized),
        ]
    )


def _has_many_urls(text: str, minimum_url_count: int = 2) -> bool:
    return len(URL_PATTERN.findall(text)) >= minimum_url_count


def _has_promotional_text(text: str) -> bool:
    return any(marker in text for marker in PROMOTIONAL_MARKERS)


def _is_link_only_or_link_promo(text: str) -> bool:
    if not URL_PATTERN.search(text):
        return False

    text_without_urls = URL_PATTERN.sub(" ", text).strip()
    if not re.search(r"\w", text_without_urls):
        return True

    word_count = len(re.findall(r"\w+", text_without_urls))
    return word_count <= 10 and _has_promotional_text(text.lower())


def _is_emoji_only(text: str) -> bool:
    visible_characters = [character for character in text.strip() if not character.isspace()]
    if not visible_characters:
        return False

    has_text = any(character.isalnum() for character in visible_characters)
    has_symbol_or_mark = any(
        unicodedata.category(character).startswith(("S", "M", "P"))
        for character in visible_characters
    )
    return not has_text and has_symbol_or_mark


def _has_excessive_repeated_characters(text: str) -> bool:
    return bool(REPEATED_CHARACTER_PATTERN.search(text))
