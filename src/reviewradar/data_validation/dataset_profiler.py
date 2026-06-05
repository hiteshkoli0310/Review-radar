"""Profiling helpers for collected ReviewRadar datasets."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def profile_video_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate summary profile statistics for a video dataset."""
    logger.info("Profiling video dataset with %s rows", len(frame))
    return {
        "dataset_type": "video",
        "number_of_videos": int(len(frame)),
        "unique_channels": _nunique(frame, "channel_name"),
        "average_views": _mean_numeric(frame, "view_count"),
        "average_likes": _mean_numeric(frame, "like_count"),
        "average_comments": _mean_numeric(frame, "comment_count"),
    }


def profile_comment_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate summary profile statistics for a comment dataset."""
    logger.info("Profiling comment dataset with %s rows", len(frame))
    comment_lengths = _comment_lengths(frame, "comment_text")

    return {
        "dataset_type": "comment",
        "total_comments": int(len(frame)),
        "unique_videos_represented": _nunique(frame, "video_id"),
        "average_comment_length": _rounded_mean(comment_lengths),
        "median_comment_length": _rounded_median(comment_lengths),
        "top_20_longest_comments": _top_longest_comments(frame, comment_lengths),
        "comments_per_video_distribution": _comments_per_video_distribution(frame),
    }


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].dropna().nunique())


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 2)


def _comment_lengths(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="int64")
    return frame[column].fillna("").astype(str).str.len()


def _rounded_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.mean()), 2)


def _rounded_median(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.median()), 2)


def _top_longest_comments(
    frame: pd.DataFrame,
    comment_lengths: pd.Series,
) -> list[dict[str, Any]]:
    if frame.empty or comment_lengths.empty:
        return []

    enriched = frame.copy()
    enriched["comment_length"] = comment_lengths
    preferred_columns = [
        "comment_id",
        "video_id",
        "video_title",
        "comment_text",
        "comment_length",
    ]
    available_columns = [column for column in preferred_columns if column in enriched.columns]
    top_comments = enriched.sort_values("comment_length", ascending=False).head(20)
    return top_comments[available_columns].to_dict(orient="records")


def _comments_per_video_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    if "video_id" not in frame.columns or frame.empty:
        return {
            "videos_with_comments": 0,
            "min_comments_per_video": 0,
            "max_comments_per_video": 0,
            "average_comments_per_video": None,
            "median_comments_per_video": None,
            "counts_by_video": {},
        }

    counts = frame["video_id"].dropna().value_counts()
    if counts.empty:
        return {
            "videos_with_comments": 0,
            "min_comments_per_video": 0,
            "max_comments_per_video": 0,
            "average_comments_per_video": None,
            "median_comments_per_video": None,
            "counts_by_video": {},
        }

    return {
        "videos_with_comments": int(counts.shape[0]),
        "min_comments_per_video": int(counts.min()),
        "max_comments_per_video": int(counts.max()),
        "average_comments_per_video": round(float(counts.mean()), 2),
        "median_comments_per_video": round(float(counts.median()), 2),
        "counts_by_video": {str(video_id): int(count) for video_id, count in counts.items()},
    }
