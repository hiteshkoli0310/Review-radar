"""Run the ReviewRadar master dataset export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reviewradar.config.settings import get_settings
from reviewradar.dataset_export import (
    build_master_dataset,
    build_master_dataset_report,
    discover_parquet_files,
    export_master_dataset,
    load_parquet_files,
    remove_duplicate_comments,
)
from reviewradar.dataset_export.export_master_dataset import save_master_dataset_report
from reviewradar.utils.logging import configure_logging


def run() -> None:
    """Discover data files, build the master dataset, and export CSV/report outputs."""
    settings = get_settings()
    configure_logging(settings.log_level)

    video_files = discover_parquet_files(settings.paths.raw_dir / "videos", "*_videos.parquet")
    comment_files = discover_parquet_files(
        settings.paths.processed_dir / "comments",
        "*_comments_processed.parquet",
    )
    export_path = settings.paths.data_dir / "exports" / "reviewradar_master_raw.csv"
    report_path = settings.paths.data_dir / "reports" / "master_dataset_report.json"

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
    _print_summary(report, export_path, report_path)


def _print_summary(report: dict[str, Any], export_path: Path, report_path: Path) -> None:
    print("ReviewRadar master dataset export")
    print(f"Products found: {', '.join(report['products_found']) or 'none'}")
    print(f"Video files found: {report['video_files_found']}")
    print(f"Comment files found: {report['comment_files_found']}")
    print(f"Total videos: {report['total_videos']}")
    print(f"Total comments: {report['total_comments']}")
    print(f"Duplicate comments removed: {report['duplicate_comments_removed']}")
    print(f"\nSaved master dataset: {export_path}")
    print(f"Saved export report: {report_path}")


if __name__ == "__main__":
    run()
