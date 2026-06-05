"""YouTube data collection utilities."""

from reviewradar.data_collection.comments import extract_top_level_comments
from reviewradar.data_collection.video_metadata import fetch_video_metadata
from reviewradar.data_collection.video_search import search_videos
from reviewradar.data_collection.youtube_client import get_youtube_client

__all__ = [
    "extract_top_level_comments",
    "fetch_video_metadata",
    "get_youtube_client",
    "search_videos",
]
