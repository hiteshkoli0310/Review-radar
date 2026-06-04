"""Search YouTube videos for product-related review content."""

from __future__ import annotations

from typing import Any

import pandas as pd
from googleapiclient.errors import HttpError


VIDEO_SEARCH_COLUMNS = [
    "product_query",
    "video_id",
    "video_url",
    "title",
    "description",
    "channel_name",
    "channel_id",
    "published_at",
]


def _empty_video_search_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=VIDEO_SEARCH_COLUMNS)


def search_videos(
    youtube_client: Any,
    product_query: str,
    max_results: int = 20,
) -> pd.DataFrame:
    """Search YouTube for videos related to a product query.

    Results are sorted by relevance, restricted to videos, and returned in a
    stable tabular schema for downstream metadata and comment collection.
    """
    clean_query = product_query.strip()
    if not clean_query:
        raise ValueError("product_query cannot be empty.")

    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50 for one API call.")

    try:
        response = (
            youtube_client.search()
            .list(
                part="snippet",
                q=clean_query,
                type="video",
                order="relevance",
                maxResults=max_results,
            )
            .execute()
        )
    except HttpError as exc:
        raise RuntimeError(f"YouTube video search failed for query: {clean_query}") from exc

    records: list[dict[str, Any]] = []
    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})

        if not video_id:
            continue

        records.append(
            {
                "product_query": clean_query,
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channel_name": snippet.get("channelTitle"),
                "channel_id": snippet.get("channelId"),
                "published_at": snippet.get("publishedAt"),
            }
        )

    if not records:
        return _empty_video_search_frame()

    return pd.DataFrame.from_records(records, columns=VIDEO_SEARCH_COLUMNS)

