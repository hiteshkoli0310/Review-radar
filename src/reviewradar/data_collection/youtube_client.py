"""YouTube Data API v3 client helpers."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from googleapiclient.discovery import build

from reviewradar.config.settings import get_settings


@dataclass(frozen=True)
class YouTubeVideoQuery:
    query: str
    max_results: int = 25


def build_search_query(product_name: str, max_results: int = 25) -> YouTubeVideoQuery:
    return YouTubeVideoQuery(query=product_name.strip(), max_results=max_results)


def get_youtube_client(api_key: str | None = None) -> Any:
    """Return an authenticated YouTube Data API v3 client."""
    settings = get_settings()
    developer_key = api_key or settings.youtube_api_key

    if not developer_key:
        raise ValueError(
            "YOUTUBE_API_KEY is missing. Add it to your .env file before running "
            "the data collection pipeline."
        )

    return build(
        settings.youtube_api_service_name,
        settings.youtube_api_version,
        developerKey=developer_key,
        cache_discovery=False,
    )
