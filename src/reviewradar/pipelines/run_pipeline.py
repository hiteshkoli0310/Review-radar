"""Run the ReviewRadar YouTube data collection pipeline."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from reviewradar.data_collection.comments import extract_top_level_comments
from reviewradar.data_collection.video_metadata import fetch_video_metadata
from reviewradar.data_collection.video_search import search_videos
from reviewradar.data_collection.youtube_client import get_youtube_client
from reviewradar.utils.logging import configure_logging
from reviewradar.config.settings import get_settings
from reviewradar.utils.io import save_parquet


VIDEO_SEARCH_LIMIT = 20
COMMENTS_PER_VIDEO_LIMIT = 100


def build_product_slug(product_query: str) -> str:
    """Convert a product query into a filesystem-friendly dataset name."""
    slug = re.sub(r"[^a-z0-9]+", "_", product_query.strip().lower()).strip("_")
    return slug or "product"


def collect_youtube_data(
    product_query: str,
    video_search_limit: int = VIDEO_SEARCH_LIMIT,
    comments_per_video: int = COMMENTS_PER_VIDEO_LIMIT,
    progress_callback: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search YouTube, fetch metadata, and extract top-level comments.

    Returns (video_metadata_df, comments_df). Both may be empty.
    """
    settings = get_settings()
    slug = build_product_slug(product_query)
    videos_output_path = settings.paths.raw_dir / "videos" / f"{slug}_videos.parquet"
    comments_output_path = settings.paths.raw_dir / "comments" / f"{slug}_comments.parquet"

    if progress_callback:
        progress_callback("Initializing YouTube API client...", 0, 6)

    youtube_client = get_youtube_client(settings.youtube_api_key)

    if progress_callback:
        progress_callback(f"Searching up to {video_search_limit} videos for: {product_query}", 1, 6)

    search_df = search_videos(
        youtube_client=youtube_client,
        product_query=product_query,
        max_results=video_search_limit,
    )

    if search_df.empty:
        print("No videos found. Pipeline finished without saving datasets.")
        return pd.DataFrame(), pd.DataFrame()

    if progress_callback:
        progress_callback(f"Found {len(search_df)} videos. Fetching detailed metadata...", 2, 6)

    video_metadata_df = fetch_video_metadata(
        youtube_client=youtube_client,
        video_ids=search_df["video_id"].tolist(),
        product_query=product_query,
    )

    if video_metadata_df.empty:
        print("No video metadata returned. Pipeline finished without saving datasets.")
        return pd.DataFrame(), pd.DataFrame()

    if progress_callback:
        progress_callback(f"Extracting up to {comments_per_video} top-level comments per video...", 3, 6)

    comments_df = extract_top_level_comments(
        youtube_client=youtube_client,
        videos_df=video_metadata_df,
        product_query=product_query,
        max_comments_per_video=comments_per_video,
    )

    if progress_callback:
        progress_callback("Saving parquet datasets...", 4, 6)

    save_parquet(video_metadata_df, videos_output_path)
    save_parquet(comments_df, comments_output_path)

    if progress_callback:
        progress_callback(f"Saved video metadata and comments for '{product_query}'", 5, 6)

    return video_metadata_df, comments_df


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    product_query = input("Enter product name: ").strip()
    if not product_query:
        raise ValueError("Product name cannot be empty.")

    print("Starting YouTube data collection...")
    videos_df, comments_df = collect_youtube_data(product_query)

    if videos_df.empty or comments_df.empty:
        print("Pipeline finished without saving datasets.")
        return

    print("ReviewRadar data collection complete.")


if __name__ == "__main__":
    run()
