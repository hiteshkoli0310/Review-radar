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
    is_single_word,
    is_spam_comment,
)
from reviewradar.preprocessing.text_cleaner import clean_text


logger = logging.getLogger(__name__)

CLEANED_TEXT_COLUMN = "cleaned_comment_text"
IS_SPAM_COLUMN = "is_spam"
IS_EMPTY_COLUMN = "is_empty"
IS_DELETED_COLUMN = "is_deleted"
IS_SHORT_COMMENT_COLUMN = "is_short_comment"
IS_SINGLE_WORD_COLUMN = "is_single_word"
IS_REMOVED_BY_CLEANING_COLUMN = "is_removed_by_cleaning"


def preprocess_comments(
    frame: pd.DataFrame,
    text_column: str = "comment_text",
) -> pd.DataFrame:
    """Return comments with cleaned text, quality flags, and low-value rows removed."""
    logger.info("Preprocessing %s comments", len(frame))
    processed = frame.copy()

    if text_column not in processed.columns:
        processed[CLEANED_TEXT_COLUMN] = ""
        processed[IS_EMPTY_COLUMN] = True
        processed[IS_DELETED_COLUMN] = False
        processed[IS_SHORT_COMMENT_COLUMN] = True
        processed[IS_SINGLE_WORD_COLUMN] = True
        processed[IS_SPAM_COLUMN] = False
        processed[IS_REMOVED_BY_CLEANING_COLUMN] = True
        empty_processed = processed.iloc[0:0].copy()
        empty_processed.attrs["preprocessing_summary"] = _build_flag_summary(processed)
        return empty_processed

    text_values = processed[text_column]
    processed[CLEANED_TEXT_COLUMN] = text_values.apply(clean_text)
    processed[IS_EMPTY_COLUMN] = processed[CLEANED_TEXT_COLUMN].apply(is_empty_comment)
    processed[IS_DELETED_COLUMN] = text_values.apply(is_deleted_comment)
    processed[IS_SHORT_COMMENT_COLUMN] = processed[CLEANED_TEXT_COLUMN].apply(is_short_comment)
    processed[IS_SINGLE_WORD_COLUMN] = processed[CLEANED_TEXT_COLUMN].apply(is_single_word)
    processed[IS_SPAM_COLUMN] = text_values.apply(is_spam_comment)
    processed[IS_REMOVED_BY_CLEANING_COLUMN] = (
        processed[IS_EMPTY_COLUMN]
        | processed[IS_DELETED_COLUMN]
        | processed[IS_SHORT_COMMENT_COLUMN]
        | processed[IS_SINGLE_WORD_COLUMN]
        | processed[IS_SPAM_COLUMN]
    )
    summary = _build_flag_summary(processed)
    filtered = processed.loc[~processed[IS_REMOVED_BY_CLEANING_COLUMN]].reset_index(drop=True)
    filtered.attrs["preprocessing_summary"] = summary
    return filtered


def build_preprocessing_report(processed_frame: pd.DataFrame, original_rows: int) -> dict[str, int]:
    """Build summary statistics for a processed comment dataset."""
    summary = processed_frame.attrs.get("preprocessing_summary", {})
    return {
        "original_rows": int(original_rows),
        "processed_rows": int(len(processed_frame)),
        "spam_count": int(summary.get("spam_count", _sum_boolean_column(processed_frame, IS_SPAM_COLUMN))),
        "empty_count": int(summary.get("empty_count", _sum_boolean_column(processed_frame, IS_EMPTY_COLUMN))),
        "deleted_count": int(
            summary.get("deleted_count", _sum_boolean_column(processed_frame, IS_DELETED_COLUMN))
        ),
        "short_comment_count": int(
            summary.get(
                "short_comment_count",
                _sum_boolean_column(processed_frame, IS_SHORT_COMMENT_COLUMN),
            )
        ),
        "single_word_count": int(
            summary.get("single_word_count", _sum_boolean_column(processed_frame, IS_SINGLE_WORD_COLUMN))
        ),
        "rows_removed_by_cleaning": int(original_rows) - int(len(processed_frame)),
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


def _build_flag_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "spam_count": _sum_boolean_column(frame, IS_SPAM_COLUMN),
        "empty_count": _sum_boolean_column(frame, IS_EMPTY_COLUMN),
        "deleted_count": _sum_boolean_column(frame, IS_DELETED_COLUMN),
        "short_comment_count": _sum_boolean_column(frame, IS_SHORT_COMMENT_COLUMN),
        "single_word_count": _sum_boolean_column(frame, IS_SINGLE_WORD_COLUMN),
    }
