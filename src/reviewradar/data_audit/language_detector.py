"""Lightweight language detection for multilingual YouTube comments."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)

LANGUAGE_COLUMN = "detected_language"

SCRIPT_RANGES = {
    "hindi": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "malayalam": ("\u0d00", "\u0d7f"),
    "kannada": ("\u0c80", "\u0cff"),
    "gujarati": ("\u0a80", "\u0aff"),
    "gurmukhi": ("\u0a00", "\u0a7f"),
}

HINGLISH_MARKERS = {
    "acha",
    "accha",
    "bhai",
    "hai",
    "hain",
    "kya",
    "ka",
    "ki",
    "ke",
    "nahi",
    "nahin",
    "mat",
    "mast",
    "bahut",
    "bhot",
    "sahi",
    "paisa",
    "bekar",
    "jugaad",
    "yaar",
}

MALAYALAM_ROMAN_MARKERS = {
    "aanu",
    "alle",
    "chetta",
    "kollam",
    "nalla",
    "polichu",
    "sheri",
}


def add_detected_language_column(
    frame: pd.DataFrame,
    text_column: str = "comment_text",
) -> pd.DataFrame:
    """Return a copy of a comment dataset with a detected language column added."""
    logger.info("Detecting languages for %s comments", len(frame))
    audited = frame.copy()
    if text_column not in audited.columns:
        audited[LANGUAGE_COLUMN] = "unknown"
        return audited

    audited[LANGUAGE_COLUMN] = audited[text_column].apply(detect_comment_language)
    return audited


def detect_comment_language(text: Any) -> str:
    """Detect a broad language label for a single comment."""
    if text is None or pd.isna(text):
        return "unknown"

    value = str(text)
    if not value.strip():
        return "unknown"

    script_counts = _count_indic_scripts(value)
    if len(script_counts) > 1:
        return _mixed_label(script_counts, has_latin=_has_latin_letters(value))

    if len(script_counts) == 1:
        language = next(iter(script_counts))
        if _has_latin_letters(value):
            return f"mixed_{language}_english"
        return language

    roman_label = _detect_romanized_language(value)
    if roman_label is not None:
        return roman_label

    if _has_latin_letters(value):
        return "english"

    return "unknown"


def _count_indic_scripts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for character in text:
        for language, (start, end) in SCRIPT_RANGES.items():
            if start <= character <= end:
                counts[language] += 1
                break
    return Counter({language: count for language, count in counts.items() if count > 0})


def _mixed_label(script_counts: Counter[str], has_latin: bool) -> str:
    languages = sorted(script_counts, key=lambda language: (-script_counts[language], language))
    if has_latin:
        languages.append("english")
    return "mixed_" + "_".join(languages[:3])


def _detect_romanized_language(text: str) -> str | None:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]+", text)}
    if not tokens:
        return None

    hinglish_hits = tokens & HINGLISH_MARKERS
    malayalam_hits = tokens & MALAYALAM_ROMAN_MARKERS

    if hinglish_hits and _has_english_context(tokens):
        return "hinglish"
    if hinglish_hits:
        return "hinglish"
    if malayalam_hits:
        return "romanized_malayalam"
    return None


def _has_english_context(tokens: set[str]) -> bool:
    common_english = {"good", "bad", "phone", "camera", "price", "review", "best", "worst"}
    return bool(tokens & common_english)


def _has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))
