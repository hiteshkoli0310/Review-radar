"""Run validation and profiling for collected ReviewRadar datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reviewradar.config.settings import get_settings
from reviewradar.config.settings import Settings
from reviewradar.data_validation import (
    profile_comment_dataset,
    profile_video_dataset,
    save_report_json,
    validate_comment_dataset,
    validate_video_dataset,
)
from reviewradar.utils.io import load_parquet
from reviewradar.utils.logging import configure_logging


def run() -> None:
    """Load latest collected datasets, validate them, profile them, and save reports."""
    settings = get_settings()
    configure_logging(settings.log_level)

    reports_dir = get_report_output_dir(settings)
    video_path = find_latest_dataset(settings.paths.raw_dir / "videos", "*_videos.parquet")
    comment_path = find_latest_dataset(
        settings.paths.raw_dir / "comments",
        "*_comments.parquet",
    )

    print("ReviewRadar data validation")
    print(f"Video dataset: {video_path}")
    print(f"Comment dataset: {comment_path}")

    video_frame = load_parquet(video_path)
    comment_frame = load_parquet(comment_path)

    video_validation = validate_video_dataset(video_frame)
    comment_validation = validate_comment_dataset(comment_frame)
    video_profile = profile_video_dataset(video_frame)
    comment_profile = profile_comment_dataset(comment_frame)

    save_report_json(video_validation, reports_dir / "video_validation_report.json")
    save_report_json(comment_validation, reports_dir / "comment_validation_report.json")
    save_report_json(video_profile, reports_dir / "video_profile_report.json")
    save_report_json(comment_profile, reports_dir / "comment_profile_report.json")

    _print_summary(video_validation, comment_validation, video_profile, comment_profile)


def find_latest_dataset(directory: Path, pattern: str) -> Path:
    """Return the newest parquet dataset matching a pattern in a directory."""
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No dataset found in {directory} matching {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_report_output_dir(settings: Settings) -> Path:
    """Return the directory where raw data validation reports are saved."""
    return settings.paths.raw_dir


def _print_summary(
    video_validation: dict[str, Any],
    comment_validation: dict[str, Any],
    video_profile: dict[str, Any],
    comment_profile: dict[str, Any],
) -> None:
    print("\nValidation summary")
    print(f"Videos: {video_validation['total_rows']}")
    print(
        "Duplicate video rows: "
        f"{video_validation['duplicate_video_ids']['duplicate_rows']}"
    )
    print(f"Invalid video URLs: {video_validation['invalid_urls']['invalid_count']}")

    print(f"Comments: {comment_validation['total_rows']}")
    print(
        "Duplicate comment rows: "
        f"{comment_validation['duplicate_comment_ids']['duplicate_rows']}"
    )
    print(f"Empty comments: {comment_validation['empty_comments']}")
    print(
        "Very short comments: "
        f"{comment_validation['comments_shorter_than_3_characters']}"
    )

    print("\nProfile summary")
    print(f"Unique channels: {video_profile['unique_channels']}")
    print(f"Average views: {video_profile['average_views']}")
    print(
        "Unique videos represented in comments: "
        f"{comment_profile['unique_videos_represented']}"
    )
    print(f"Average comment length: {comment_profile['average_comment_length']}")


if __name__ == "__main__":
    run()
