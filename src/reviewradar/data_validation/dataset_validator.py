"""Validation helpers for collected ReviewRadar datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def save_report_json(report: dict[str, Any], output_path: Path) -> Path:
    """Save a validation or profiling report as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved report to %s", output_path)
    return output_path


def validate_video_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate video dataset shape, identity, URL, and timestamp quality."""
    logger.info("Validating video dataset with %s rows", len(frame))
    duplicate_video_ids = _duplicate_count(frame, "video_id")
    missing_values = _missing_values_by_column(frame)
    invalid_urls = _invalid_url_count(frame, "video_url")
    invalid_timestamps = _invalid_timestamps_by_column(frame, ["published_at"])

    return {
        "dataset_type": "video",
        "total_rows": int(len(frame)),
        "duplicate_video_ids": duplicate_video_ids,
        "missing_values_by_column": missing_values,
        "invalid_urls": invalid_urls,
        "invalid_timestamps_by_column": invalid_timestamps,
        "quality_statistics": {
            "percentage_missing_values": _percentage_missing_values(frame),
            "percentage_duplicate_video_ids": _percentage_of_rows(
                duplicate_video_ids["duplicate_rows"],
                len(frame),
            ),
        },
    }


def validate_comment_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate comment dataset shape, identity, text, and timestamp quality."""
    logger.info("Validating comment dataset with %s rows", len(frame))
    duplicate_comment_ids = _duplicate_count(frame, "comment_id")
    missing_values = _missing_values_by_column(frame)
    empty_comments = _empty_text_count(frame, "comment_text")
    whitespace_only_comments = _whitespace_only_text_count(frame, "comment_text")
    very_short_comments = _short_text_count(frame, "comment_text", minimum_length=3)
    invalid_timestamps = _invalid_timestamps_by_column(
        frame,
        ["published_at", "updated_at"],
    )

    return {
        "dataset_type": "comment",
        "total_rows": int(len(frame)),
        "duplicate_comment_ids": duplicate_comment_ids,
        "missing_values_by_column": missing_values,
        "empty_comments": empty_comments,
        "whitespace_only_comments": whitespace_only_comments,
        "comments_shorter_than_3_characters": very_short_comments,
        "invalid_timestamps_by_column": invalid_timestamps,
        "quality_statistics": {
            "percentage_missing_values": _percentage_missing_values(frame),
            "percentage_duplicate_comments": _percentage_of_rows(
                duplicate_comment_ids["duplicate_rows"],
                len(frame),
            ),
            "percentage_empty_comments": _percentage_of_rows(empty_comments, len(frame)),
            "percentage_very_short_comments": _percentage_of_rows(
                very_short_comments,
                len(frame),
            ),
        },
    }


def _duplicate_count(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {
            "column": column,
            "column_present": False,
            "duplicate_rows": 0,
            "duplicate_values": 0,
        }

    non_missing = frame[column].dropna()
    duplicate_rows = int(non_missing.duplicated(keep=False).sum())
    duplicate_values = int(non_missing[non_missing.duplicated(keep=False)].nunique())
    return {
        "column": column,
        "column_present": True,
        "duplicate_rows": duplicate_rows,
        "duplicate_values": duplicate_values,
    }


def _missing_values_by_column(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    total_rows = len(frame)
    report: dict[str, dict[str, float | int]] = {}
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        report[column] = {
            "missing_count": missing_count,
            "missing_percentage": _percentage_of_rows(missing_count, total_rows),
        }
    return report


def _invalid_url_count(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"column": column, "column_present": False, "invalid_count": 0}

    values = frame[column].dropna().astype(str).str.strip()
    valid_mask = values.str.match(r"^https?://", case=False)
    return {
        "column": column,
        "column_present": True,
        "invalid_count": int((~valid_mask).sum()),
    }


def _invalid_timestamps_by_column(
    frame: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in frame.columns:
            report[column] = {"column_present": False, "invalid_count": 0}
            continue

        values = frame[column].dropna()
        parsed_values = pd.to_datetime(values, errors="coerce", utc=True)
        report[column] = {
            "column_present": True,
            "invalid_count": int(parsed_values.isna().sum()),
        }
    return report


def _empty_text_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna("").astype(str).eq("").sum())


def _whitespace_only_text_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0

    values = frame[column].fillna("").astype(str)
    return int((values.ne("") & values.str.strip().eq("")).sum())


def _short_text_count(frame: pd.DataFrame, column: str, minimum_length: int) -> int:
    if column not in frame.columns:
        return 0

    lengths = frame[column].fillna("").astype(str).str.strip().str.len()
    return int(lengths.lt(minimum_length).sum())


def _percentage_missing_values(frame: pd.DataFrame) -> float:
    total_cells = frame.shape[0] * frame.shape[1]
    if total_cells == 0:
        return 0.0
    return round(float(frame.isna().sum().sum() / total_cells * 100), 2)


def _percentage_of_rows(count: int, total_rows: int) -> float:
    if total_rows == 0:
        return 0.0
    return round(float(count / total_rows * 100), 2)
