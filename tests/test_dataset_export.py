from __future__ import annotations

import pandas as pd

from reviewradar.dataset_export.dataset_builder import (
    MASTER_DATASET_COLUMNS,
    build_master_dataset,
    build_master_dataset_report,
    discover_parquet_files,
    load_parquet_files,
    remove_duplicate_comments,
)


def test_discover_and_load_parquet_files_includes_future_products(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed" / "comments"
    data_dir.mkdir(parents=True)
    first = data_dir / "iphone_17_comments_processed.parquet"
    second = data_dir / "future_product_comments_processed.parquet"
    pd.DataFrame({"comment_id": ["c1"]}).to_parquet(first)
    pd.DataFrame({"comment_id": ["c2"]}).to_parquet(second)

    paths = discover_parquet_files(data_dir, "*_comments_processed.parquet")
    frame = load_parquet_files(paths)

    assert paths == sorted([first, second])
    assert set(frame["comment_id"]) == {"c1", "c2"}


def test_build_master_dataset_preserves_comment_rows_and_maps_required_columns() -> None:
    videos = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "product_query": ["iphone_17", "nintendo_switch"],
            "title": ["iPhone review", "Switch review"],
            "video_url": ["https://youtube.com/watch?v=v1", "https://youtube.com/watch?v=v2"],
            "channel_name": ["A", "B"],
            "published_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "view_count": [100, 200],
            "comment_count": [10, 20],
            "like_count": [5, 6],
        }
    )
    comments = pd.DataFrame(
        {
            "product_query": ["iphone_17", "nintendo_switch", "future_product"],
            "video_id": ["v1", "v2", "missing-video"],
            "video_title": ["old title", "old title", "Future title"],
            "video_url": ["old url", "old url", "https://youtube.com/watch?v=missing"],
            "comment_id": ["c1", "c2", "c3"],
            "author_name": ["x", "y", "z"],
            "comment_text": ["good", "bad", "amazing"],
            "cleaned_comment_text": ["good", "bad", "amazing"],
            "like_count": [1, 2, 3],
            "detected_language": ["english", "english", "english"],
            "is_empty": [False, False, False],
            "is_deleted": [False, False, False],
            "is_short_comment": [False, False, False],
            "is_spam": [False, False, False],
            "updated_at": ["2026-01-03T00:00:00Z", None, None],
        }
    )

    master = build_master_dataset(videos, comments)

    assert list(master.columns) == MASTER_DATASET_COLUMNS
    assert len(master) == 3
    assert master.loc[0, "video_title"] == "iPhone review"
    assert master.loc[0, "video_like_count"] == 5
    assert master.loc[0, "comment_like_count"] == 1
    assert master.loc[2, "video_title"] == "Future title"


def test_remove_duplicate_comments_keeps_first_copy() -> None:
    master = pd.DataFrame(
        {
            "comment_id": ["c1", "c1", "c2"],
            "comment_text": ["first", "second", "third"],
        }
    )

    deduplicated, removed = remove_duplicate_comments(master)

    assert removed == 1
    assert list(deduplicated["comment_text"]) == ["first", "third"]


def test_build_master_dataset_report_summarizes_export() -> None:
    master = pd.DataFrame(
        {
            "product_query": ["nintendo_switch", "iphone_17"],
            "comment_id": ["c1", "c2"],
        }
    )
    videos = pd.DataFrame({"video_id": ["v1", "v2", "v2"]})

    report = build_master_dataset_report(
        master_dataset=master,
        videos=videos,
        video_files=[object(), object()],  # type: ignore[list-item]
        comment_files=[object()],  # type: ignore[list-item]
        duplicate_comments_removed=3,
    )

    assert report == {
        "products_found": ["iphone_17", "nintendo_switch"],
        "video_files_found": 2,
        "comment_files_found": 1,
        "total_comments": 2,
        "total_videos": 2,
        "duplicate_comments_removed": 3,
    }
