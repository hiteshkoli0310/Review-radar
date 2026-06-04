"""Fetch detailed YouTube video metadata for searched videos."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd
from googleapiclient.errors import HttpError


logger = logging.getLogger(__name__)

VIDEO_METADATA_COLUMNS = [
    "video_id",
    "product_query",
    "title",
    "video_url",
    "channel_name",
    "published_at",
    "view_count",
    "like_count",
    "comment_count",
    "duration",
    "tags",
    "category_id",
]


def _empty_video_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=VIDEO_METADATA_COLUMNS)


def _chunked(values: list[str], chunk_size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), chunk_size):
        yield values[index : index + chunk_size]


def _safe_int(value: str | int | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_video_metadata(
    youtube_client: Any,
    video_ids: Iterable[str],
    product_query: str,
) -> pd.DataFrame:
    """Fetch detailed metadata for a collection of YouTube video IDs."""
    clean_query = product_query.strip()
    clean_video_ids = list(dict.fromkeys(video_id for video_id in video_ids if video_id))

    if not clean_query:
        raise ValueError("product_query cannot be empty.")

    if not clean_video_ids:
        return _empty_video_metadata_frame()

    records: list[dict[str, Any]] = []

    # The videos.list endpoint accepts up to 50 comma-separated video IDs.
    for batch in _chunked(clean_video_ids, chunk_size=50):
        try:
            response = (
                youtube_client.videos()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch),
                )
                .execute()
            )
        except HttpError as exc:
            logger.warning("Skipping metadata batch after YouTube API error: %s", exc)
            continue

        for item in response.get("items", []):
            video_id = item.get("id")
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})

            if not video_id:
                continue

            records.append(
                {
                    "video_id": video_id,
                    "product_query": clean_query,
                    "title": snippet.get("title"),
                    "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel_name": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                    "view_count": _safe_int(statistics.get("viewCount")),
                    "like_count": _safe_int(statistics.get("likeCount")),
                    "comment_count": _safe_int(statistics.get("commentCount")),
                    "duration": content_details.get("duration"),
                    "tags": snippet.get("tags", []),
                    "category_id": snippet.get("categoryId"),
                }
            )

    if not records:
        return _empty_video_metadata_frame()

    return pd.DataFrame.from_records(records, columns=VIDEO_METADATA_COLUMNS)

