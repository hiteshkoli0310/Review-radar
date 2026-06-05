from __future__ import annotations

import pandas as pd

from reviewradar.config.settings import get_settings
from reviewradar.data_validation.dataset_profiler import (
    profile_comment_dataset,
    profile_video_dataset,
)
from reviewradar.data_validation.dataset_validator import (
    validate_comment_dataset,
    validate_video_dataset,
)
from reviewradar.pipelines.run_validation import get_report_output_dir


def test_validate_video_dataset_reports_duplicates_missing_urls_and_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "video_id": ["v1", "v1", "v2"],
            "video_url": [
                "https://www.youtube.com/watch?v=v1",
                "not-a-url",
                "http://youtube.com/watch?v=v2",
            ],
            "published_at": ["2026-01-01T00:00:00Z", "bad-date", None],
            "title": ["Review", None, "Another review"],
        }
    )

    report = validate_video_dataset(frame)

    assert report["total_rows"] == 3
    assert report["duplicate_video_ids"]["duplicate_rows"] == 2
    assert report["missing_values_by_column"]["title"]["missing_count"] == 1
    assert report["invalid_urls"]["invalid_count"] == 1
    assert report["invalid_timestamps_by_column"]["published_at"]["invalid_count"] == 1


def test_validate_comment_dataset_reports_comment_quality_issues() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c1", "c2", "c3", "c4"],
            "video_id": ["v1", "v1", "v2", "v2", "v2"],
            "comment_text": ["Great phone", "", "   ", "ok", None],
            "published_at": ["2026-01-01T00:00:00Z", "bad-date", None, None, None],
            "updated_at": ["2026-01-01T00:00:00Z", None, None, "bad-date", None],
        }
    )

    report = validate_comment_dataset(frame)

    assert report["total_rows"] == 5
    assert report["duplicate_comment_ids"]["duplicate_rows"] == 2
    assert report["empty_comments"] == 2
    assert report["whitespace_only_comments"] == 1
    assert report["comments_shorter_than_3_characters"] == 4
    assert report["invalid_timestamps_by_column"]["published_at"]["invalid_count"] == 1
    assert report["invalid_timestamps_by_column"]["updated_at"]["invalid_count"] == 1
    assert report["quality_statistics"]["percentage_duplicate_comments"] == 40.0


def test_profile_video_dataset_calculates_summary_metrics() -> None:
    frame = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "channel_name": ["A", "B"],
            "view_count": [100, 300],
            "like_count": [10, 30],
            "comment_count": [5, 15],
        }
    )

    report = profile_video_dataset(frame)

    assert report["number_of_videos"] == 2
    assert report["unique_channels"] == 2
    assert report["average_views"] == 200.0
    assert report["average_likes"] == 20.0
    assert report["average_comments"] == 10.0


def test_profile_comment_dataset_calculates_lengths_and_distribution() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3"],
            "video_id": ["v1", "v1", "v2"],
            "comment_text": ["short", "a much longer comment", "mid"],
        }
    )
    original_columns = list(frame.columns)

    report = profile_comment_dataset(frame)

    assert report["total_comments"] == 3
    assert report["unique_videos_represented"] == 2
    assert report["average_comment_length"] == 9.67
    assert report["median_comment_length"] == 5.0
    assert report["top_20_longest_comments"][0]["comment_id"] == "c2"
    assert report["comments_per_video_distribution"]["counts_by_video"] == {
        "v1": 2,
        "v2": 1,
    }
    assert list(frame.columns) == original_columns


def test_validation_reports_are_saved_under_data_raw() -> None:
    settings = get_settings()

    assert get_report_output_dir(settings) == settings.paths.raw_dir
