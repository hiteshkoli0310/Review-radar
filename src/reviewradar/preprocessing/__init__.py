"""Preprocessing utilities for ReviewRadar."""

from reviewradar.preprocessing.preprocessing_pipeline import (
    build_preprocessing_report,
    preprocess_comments,
    save_preprocessing_report,
)
from reviewradar.preprocessing.spam_detector import (
    is_emoji_only,
    is_single_word,
    is_spam_comment,
)
from reviewradar.preprocessing.text_cleaner import clean_text

__all__ = [
    "build_preprocessing_report",
    "clean_text",
    "is_emoji_only",
    "is_single_word",
    "is_spam_comment",
    "preprocess_comments",
    "save_preprocessing_report",
]
