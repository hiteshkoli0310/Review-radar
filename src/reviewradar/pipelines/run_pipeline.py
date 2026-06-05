"""Run the ReviewRadar YouTube data collection pipeline."""

from __future__ import annotations

import re

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


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    product_query = input("Enter product name: ").strip()
    if not product_query:
        raise ValueError("Product name cannot be empty.")

    dataset_slug = build_product_slug(product_query)
    videos_output_path = settings.paths.raw_dir / "videos" / f"{dataset_slug}_videos.parquet"
    comments_output_path = (
        settings.paths.raw_dir / "comments" / f"{dataset_slug}_comments.parquet"
    )

    print("Initializing YouTube API client...")
    youtube_client = get_youtube_client(settings.youtube_api_key)

    print(f"Searching up to {VIDEO_SEARCH_LIMIT} videos for: {product_query}")
    search_df = search_videos(
        youtube_client=youtube_client,
        product_query=product_query,
        max_results=VIDEO_SEARCH_LIMIT,
    )

    if search_df.empty:
        print("No videos found. Pipeline finished without saving datasets.")
        return

    print(f"Found {len(search_df)} videos. Fetching detailed metadata...")
    video_metadata_df = fetch_video_metadata(
        youtube_client=youtube_client,
        video_ids=search_df["video_id"].tolist(),
        product_query=product_query,
    )

    if video_metadata_df.empty:
        print("No video metadata returned. Pipeline finished without saving datasets.")
        return

    print(f"Extracting up to {COMMENTS_PER_VIDEO_LIMIT} top-level comments per video...")
    comments_df = extract_top_level_comments(
        youtube_client=youtube_client,
        videos_df=video_metadata_df,
        product_query=product_query,
        max_comments_per_video=COMMENTS_PER_VIDEO_LIMIT,
    )

    print("Saving parquet datasets...")
    save_parquet(video_metadata_df, videos_output_path)
    save_parquet(comments_df, comments_output_path)

    print(f"Saved video metadata: {videos_output_path}")
    print(f"Saved comments: {comments_output_path}")
    print("ReviewRadar data collection complete.")


if __name__ == "__main__":
    run()
