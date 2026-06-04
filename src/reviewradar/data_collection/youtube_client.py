"""YouTube Data API v3 client helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YouTubeVideoQuery:
    query: str
    max_results: int = 25


def build_search_query(product_name: str, max_results: int = 25) -> YouTubeVideoQuery:
    return YouTubeVideoQuery(query=product_name.strip(), max_results=max_results)