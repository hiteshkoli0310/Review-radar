"""Text cleaning and normalization."""

from __future__ import annotations

from reviewradar.preprocessing.text_cleaner import clean_text


def clean_comment_text(text: str) -> str:
    return clean_text(text)
