"""Comment preprocessing pipeline helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.preprocessing.spam_detector import (
    is_deleted_comment,
    is_empty_comment,
    is_short_comment,
    is_spam_comment,
)
from reviewradar.preprocessing.text_cleaner import clean_text


logger = logging.getLogger(__name__)

CLEANED_TEXT_COLUMN = "cleaned_comment_text"
IS_SPAM_COLUMN = "is_spam"
IS_EMPTY_COLUMN = "is_empty"
IS_DELETED_COLUMN = "is_deleted"
IS_SHORT_COMMENT_COLUMN = "is_short_comment"


def preprocess_comments(
    frame: pd.DataFrame,
    text_column: str = "comment_text",
) -> pd.DataFrame:
    """Return a copy of comments with cleaned text and quality flags added."""
    logger.info("Preprocessing %s comments", len(frame))
    processed = frame.copy()

    if text_column not in processed.columns:
        processed[CLEANED_TEXT_COLUMN] = ""
        processed[IS_EMPTY_COLUMN] = True
        processed[IS_DELETED_COLUMN] = False
        processed[IS_SHORT_COMMENT_COLUMN] = True
        processed[IS_SPAM_COLUMN] = False
        return processed

    text_values = processed[text_column]
    processed[CLEANED_TEXT_COLUMN] = text_values.apply(clean_text)
    processed[IS_EMPTY_COLUMN] = text_values.apply(is_empty_comment)
    processed[IS_DELETED_COLUMN] = text_values.apply(is_deleted_comment)
    processed[IS_SHORT_COMMENT_COLUMN] = text_values.apply(is_short_comment)
    processed[IS_SPAM_COLUMN] = text_values.apply(is_spam_comment)
    return processed


def build_preprocessing_report(processed_frame: pd.DataFrame, original_rows: int) -> dict[str, int]:
    """Build summary statistics for a processed comment dataset."""
    return {
        "original_rows": int(original_rows),
        "processed_rows": int(len(processed_frame)),
        "spam_count": _sum_boolean_column(processed_frame, IS_SPAM_COLUMN),
        "empty_count": _sum_boolean_column(processed_frame, IS_EMPTY_COLUMN),
        "deleted_count": _sum_boolean_column(processed_frame, IS_DELETED_COLUMN),
        "short_comment_count": _sum_boolean_column(processed_frame, IS_SHORT_COMMENT_COLUMN),
    }


def save_preprocessing_report(report: dict[str, Any], output_path: Path) -> Path:
    """Save a preprocessing report as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved preprocessing report to %s", output_path)
    return output_path


def _sum_boolean_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())
