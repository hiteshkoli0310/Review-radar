"""Run the ReviewRadar Phase 1 YouTube data collection pipeline."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scraper.comment_extractor import extract_top_level_comments
from scraper.video_metadata import fetch_video_metadata
from scraper.video_search import search_videos
from scraper.youtube_client import get_youtube_client
from utils.save_data import save_dataframe_to_parquet


VIDEO_SEARCH_LIMIT = 20
COMMENTS_PER_VIDEO_LIMIT = 100


def build_product_slug(product_query: str) -> str:
    """Convert a product query into a filesystem-friendly dataset name."""
    slug = re.sub(r"[^a-z0-9]+", "_", product_query.strip().lower()).strip("_")
    return slug or "product"


def run_pipeline() -> None:
    """Collect YouTube videos, metadata, and top-level comments for a product."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    product_query = input("Enter product name: ").strip()
    if not product_query:
        raise ValueError("Product name cannot be empty.")

    dataset_slug = build_product_slug(product_query)
    videos_output_path = PROJECT_ROOT / "data" / "raw" / "videos" / f"{dataset_slug}_videos.parquet"
    comments_output_path = (
        PROJECT_ROOT / "data" / "raw" / "comments" / f"{dataset_slug}_comments.parquet"
    )

    print("Initializing YouTube API client...")
    youtube_client = get_youtube_client()

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
    saved_videos_path = save_dataframe_to_parquet(video_metadata_df, videos_output_path)
    saved_comments_path = save_dataframe_to_parquet(comments_df, comments_output_path)

    print(f"Saved video metadata: {saved_videos_path}")
    print(f"Saved comments: {saved_comments_path}")
    print("ReviewRadar Phase 1 data collection complete.")


if __name__ == "__main__":
    run_pipeline()
