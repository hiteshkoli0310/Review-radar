"""Build master comment-level datasets for ReviewRadar EDA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)

MASTER_DATASET_COLUMNS = [
    "product_query",
    "video_id",
    "video_title",
    "video_url",
    "channel_name",
    "published_at",
    "view_count",
    "comment_count",
    "video_like_count",
    "comment_id",
    "author_name",
    "comment_text",
    "cleaned_comment_text",
    "comment_like_count",
    "detected_language",
    "is_empty",
    "is_deleted",
    "is_short_comment",
    "is_spam",
    "updated_at",
]


def discover_parquet_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted parquet files matching a pattern."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def load_parquet_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate parquet files, returning an empty frame when none exist."""
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        logger.info("Loading parquet file: %s", path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def build_master_dataset(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Merge processed comments with video metadata into a master comment dataset."""
    prepared_comments = _prepare_comments(comments)
    prepared_videos = _prepare_videos(videos)

    if prepared_comments.empty:
        return _ensure_master_columns(prepared_comments)

    if prepared_videos.empty or "video_id" not in prepared_videos.columns:
        return _ensure_master_columns(prepared_comments)

    merged = prepared_comments.merge(
        prepared_videos,
        on="video_id",
        how="left",
        suffixes=("_comment", "_video"),
    )
    merged = _coalesce_merged_columns(merged)
    return _ensure_master_columns(merged)


def remove_duplicate_comments(master_dataset: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate comments by comment_id while preserving the first copy."""
    if "comment_id" not in master_dataset.columns:
        return master_dataset, 0

    before = len(master_dataset)
    deduplicated = master_dataset.drop_duplicates(subset=["comment_id"], keep="first")
    removed = before - len(deduplicated)
    return deduplicated.reset_index(drop=True), int(removed)


def build_master_dataset_report(
    master_dataset: pd.DataFrame,
    videos: pd.DataFrame,
    video_files: list[Path],
    comment_files: list[Path],
    duplicate_comments_removed: int,
) -> dict[str, Any]:
    """Build summary metadata for a master dataset export."""
    products = _products_found(master_dataset)
    total_videos = int(videos["video_id"].dropna().nunique()) if "video_id" in videos else 0

    return {
        "products_found": products,
        "video_files_found": len(video_files),
        "comment_files_found": len(comment_files),
        "total_comments": int(len(master_dataset)),
        "total_videos": total_videos,
        "duplicate_comments_removed": int(duplicate_comments_removed),
    }


def _prepare_comments(comments: pd.DataFrame) -> pd.DataFrame:
    prepared = comments.copy()
    if "like_count" in prepared.columns and "comment_like_count" not in prepared.columns:
        prepared = prepared.rename(columns={"like_count": "comment_like_count"})
    if "title" in prepared.columns and "video_title" not in prepared.columns:
        prepared = prepared.rename(columns={"title": "video_title"})
    return prepared


def _prepare_videos(videos: pd.DataFrame) -> pd.DataFrame:
    prepared = videos.copy()
    rename_map = {
        "title": "video_title",
        "like_count": "video_like_count",
    }
    prepared = prepared.rename(
        columns={source: target for source, target in rename_map.items() if source in prepared}
    )
    if "video_id" in prepared.columns:
        prepared = prepared.drop_duplicates(subset=["video_id"], keep="first")

    columns = [
        "video_id",
        "product_query",
        "video_title",
        "video_url",
        "channel_name",
        "published_at",
        "view_count",
        "comment_count",
        "video_like_count",
    ]
    return prepared[[column for column in columns if column in prepared.columns]]


def _coalesce_merged_columns(merged: pd.DataFrame) -> pd.DataFrame:
    for column in ["product_query", "video_title", "video_url", "published_at"]:
        comment_column = f"{column}_comment"
        video_column = f"{column}_video"
        if video_column in merged.columns and comment_column in merged.columns:
            merged[column] = merged[video_column].combine_first(merged[comment_column])
        elif video_column in merged.columns:
            merged[column] = merged[video_column]
        elif comment_column in merged.columns:
            merged[column] = merged[comment_column]

    drop_columns = [
        column
        for column in merged.columns
        if column.endswith("_comment") or column.endswith("_video")
    ]
    return merged.drop(columns=drop_columns)


def _ensure_master_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in MASTER_DATASET_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[MASTER_DATASET_COLUMNS]


def _products_found(master_dataset: pd.DataFrame) -> list[str]:
    if "product_query" not in master_dataset.columns:
        return []
    products = master_dataset["product_query"].dropna().astype(str).unique().tolist()
    return sorted(products)
