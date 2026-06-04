"""Text cleaning and normalization."""

from __future__ import annotations

import re


def clean_comment_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized