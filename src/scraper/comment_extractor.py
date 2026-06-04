"""Extract top-level YouTube comments for product review videos."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from googleapiclient.errors import HttpError


logger = logging.getLogger(__name__)

COMMENT_COLUMNS = [
    "product_query",
    "video_id",
    "video_url",
    "video_title",
    "comment_id",
    "author_name",
    "comment_text",
    "like_count",
    "published_at",
    "updated_at",
]


def _empty_comments_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMENT_COLUMNS)


def _safe_int(value: str | int | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_top_level_comments(
    youtube_client: Any,
    videos_df: pd.DataFrame,
    product_query: str,
    max_comments_per_video: int = 100,
) -> pd.DataFrame:
    """Extract top-level comments for each video in a metadata DataFrame.

    Replies are intentionally excluded by requesting only the comment thread
    snippet and reading the thread's top-level comment.
    """
    clean_query = product_query.strip()
    if not clean_query:
        raise ValueError("product_query cannot be empty.")

    if max_comments_per_video < 1:
        raise ValueError("max_comments_per_video must be at least 1.")

    if videos_df.empty:
        return _empty_comments_frame()

    records: list[dict[str, Any]] = []

    for video in videos_df.to_dict(orient="records"):
        video_id = video.get("video_id")
        if not video_id:
            continue

        video_url = video.get("video_url") or f"https://www.youtube.com/watch?v={video_id}"
        video_title = video.get("title")
        next_page_token: str | None = None
        comments_collected = 0

        while comments_collected < max_comments_per_video:
            page_size = min(100, max_comments_per_video - comments_collected)
            request_params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": page_size,
                "textFormat": "plainText",
            }

            if next_page_token:
                request_params["pageToken"] = next_page_token

            try:
                response = (
                    youtube_client.commentThreads()
                    .list(**request_params)
                    .execute()
                )
            except HttpError as exc:
                logger.warning(
                    "Skipping comments for video %s after YouTube API error: %s",
                    video_id,
                    exc,
                )
                break

            for item in response.get("items", []):
                top_level_comment = (
                    item.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet", {})
                )
                comment_id = item.get("snippet", {}).get("topLevelComment", {}).get("id")

                if not comment_id:
                    continue

                records.append(
                    {
                        "product_query": clean_query,
                        "video_id": video_id,
                        "video_url": video_url,
                        "video_title": video_title,
                        "comment_id": comment_id,
                        "author_name": top_level_comment.get("authorDisplayName"),
                        "comment_text": top_level_comment.get("textDisplay"),
                        "like_count": _safe_int(top_level_comment.get("likeCount")),
                        "published_at": top_level_comment.get("publishedAt"),
                        "updated_at": top_level_comment.get("updatedAt"),
                    }
                )
                comments_collected += 1

                if comments_collected >= max_comments_per_video:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    if not records:
        return _empty_comments_frame()

    return pd.DataFrame.from_records(records, columns=COMMENT_COLUMNS)
