"""Orchestrate the full ReviewRadar pipeline for a single product from the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from reviewradar.config.settings import get_settings
from reviewradar.data_audit import add_detected_language_column
from reviewradar.data_collection.youtube_client import get_youtube_client
from reviewradar.dataset_export import (
    build_master_dataset,
    build_master_dataset_report,
    discover_parquet_files,
    export_master_dataset,
    load_parquet_files,
    remove_duplicate_comments,
)
from reviewradar.dataset_export.export_master_dataset import save_master_dataset_report
from reviewradar.evaluation.insight_generator import generate_product_insights
from reviewradar.evaluation.sentiment_evaluation import DistilBertScorer as SentimentScorer
from reviewradar.pipelines.run_pipeline import (
    build_product_slug,
    collect_youtube_data,
)
from reviewradar.preprocessing import preprocess_comments
from reviewradar.translation.translation_pipeline import (
    final_cleanup,
    translate_comments,
)
from reviewradar.utils.io import load_parquet, save_parquet


ProgressCallback = Callable[[str, int, int], None] | None


def check_product_cached(slug: str) -> bool:
    """Return True if the product has already been fully processed."""
    settings = get_settings()
    processed_path = settings.paths.processed_dir / "comments" / f"{slug}_comments_processed.parquet"
    return processed_path.exists()


def _run_language_audit(
    slug: str,
    progress_callback: ProgressCallback = None,
) -> None:
    """Step 2: Language detection for a single product's raw comments."""
    settings = get_settings()
    raw_path = settings.paths.raw_dir / "comments" / f"{slug}_comments.parquet"
    output_path = settings.paths.interim_dir / "comments" / f"{slug}_comments_language_audited.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw comments not found: {raw_path}")

    if progress_callback:
        progress_callback("Running language detection...", 1, 4)

    comments = load_parquet(raw_path)
    audited = add_detected_language_column(comments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(audited, output_path)

    if progress_callback:
        progress_callback(f"Language audit complete: {len(audited)} comments", 1, 4)


def _run_preprocessing(
    slug: str,
    progress_callback: ProgressCallback = None,
) -> None:
    """Step 3: Preprocess (clean, filter spam/empty/short) for a single product."""
    settings = get_settings()
    input_path = settings.paths.interim_dir / "comments" / f"{slug}_comments_language_audited.parquet"
    output_path = settings.paths.processed_dir / "comments" / f"{slug}_comments_processed.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Language-audited comments not found: {input_path}")

    if progress_callback:
        progress_callback("Preprocessing comments (clean, filter spam/short)...", 2, 4)

    comments = load_parquet(input_path)
    processed = preprocess_comments(comments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(processed, output_path)

    if progress_callback:
        progress_callback(f"Preprocessing complete: {len(processed)} rows kept", 2, 4)


def _run_translation(
    slug: str,
    progress_callback: ProgressCallback = None,
) -> None:
    """Step 4: Translate non-English comments for a single product."""
    settings = get_settings()
    processed_dir = settings.paths.processed_dir / "comments"
    input_path = processed_dir / f"{slug}_comments_processed.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Processed comments not found: {input_path}")

    if progress_callback:
        progress_callback("Translating non-English comments (NLLB model)...", 3, 4)

    frame = load_parquet(input_path)
    frame = translate_comments(frame)
    frame = final_cleanup(frame)
    save_parquet(frame, input_path)

    if progress_callback:
        progress_callback(f"Translation complete: {len(frame)} rows after cleanup", 3, 4)


def _rebuild_master_csv(
    progress_callback: ProgressCallback = None,
) -> None:
    """Step 5: Rebuild the master CSV from ALL processed products."""
    settings = get_settings()

    if progress_callback:
        progress_callback("Rebuilding master dataset CSV...", 4, 4)

    video_dir = settings.paths.raw_dir / "videos"
    comment_dir = settings.paths.processed_dir / "comments"
    export_path = settings.paths.data_dir / "exports" / "reviewradar_master_raw.csv"
    report_path = settings.paths.data_dir / "reports" / "master_dataset_report.json"

    video_files = discover_parquet_files(video_dir, "*_videos.parquet")
    comment_files = discover_parquet_files(comment_dir, "*_comments_processed.parquet")

    videos = load_parquet_files(video_files)
    comments = load_parquet_files(comment_files)
    master_dataset = build_master_dataset(videos, comments)
    master_dataset, duplicate_comments_removed = remove_duplicate_comments(master_dataset)
    report = build_master_dataset_report(
        master_dataset=master_dataset,
        videos=videos,
        video_files=video_files,
        comment_files=comment_files,
        duplicate_comments_removed=duplicate_comments_removed,
    )

    export_master_dataset(master_dataset, export_path)
    save_master_dataset_report(report, report_path)

    if progress_callback:
        progress_callback(f"Master CSV rebuilt: {len(master_dataset)} comments", 4, 4)


def process_single_product(
    slug: str,
    progress_callback: ProgressCallback = None,
) -> None:
    """Run language audit, preprocessing, and translation for one product.

    Assumes raw YouTube data has already been collected (step 1).
    """
    _run_language_audit(slug, progress_callback)
    _run_preprocessing(slug, progress_callback)
    _run_translation(slug, progress_callback)
    _rebuild_master_csv(progress_callback)


def run_pipeline_for_product(
    product_query: str,
    sentiment_scorer: SentimentScorer,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any] | None:
    """Run the full pipeline for a product: collect, process, classify, generate insights.

    Returns the insight dict for the product (same schema as ``parse_insight_report``),
    or ``None`` if no data was collected.
    """
    slug = build_product_slug(product_query)

    if check_product_cached(slug):
        if progress_callback:
            progress_callback(f"'{product_query}' already processed — regenerating insights...", 0, 1)
        _rebuild_master_csv(progress_callback)
    else:
        if progress_callback:
            progress_callback(f"Starting pipeline for '{product_query}'...", 0, 2)

        videos_df, comments_df = collect_youtube_data(
            product_query,
            progress_callback=progress_callback,
        )

        if videos_df.empty or comments_df.empty:
            return None

        process_single_product(slug, progress_callback)

    if progress_callback:
        progress_callback("Generating consumer insight reports...", 0, 1)

    settings = get_settings()
    master_csv = settings.paths.data_dir / "exports" / "reviewradar_master_raw.csv"
    if not master_csv.exists():
        return None

    master_df = pd.read_csv(master_csv)
    all_insights = generate_product_insights(
        df=master_df,
        sentiment_scorer=sentiment_scorer,
        output_dir=str(settings.paths.data_dir / "reports" / "insights"),
    )

    return all_insights.get(product_query)
