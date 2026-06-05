"""Run NLP preprocessing for language-audited ReviewRadar comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.config.settings import get_settings
from reviewradar.preprocessing import (
    build_preprocessing_report,
    preprocess_comments,
    save_preprocessing_report,
)
from reviewradar.utils.io import load_parquet, save_parquet
from reviewradar.utils.logging import configure_logging


def run() -> None:
    """Preprocess all language-audited comment datasets and save outputs."""
    settings = get_settings()
    configure_logging(settings.log_level)

    input_paths = find_language_audited_comments(settings.paths.interim_dir / "comments")
    report_path = settings.paths.data_dir / "reports" / "preprocessing_report.json"

    print("ReviewRadar preprocessing")

    processed_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    original_rows = 0
    for input_path in input_paths:
        output_path = build_processed_comments_output_path(
            input_path,
            settings.paths.processed_dir / "comments",
        )
        print(f"Input dataset: {input_path}")
        comments = load_parquet(input_path)
        processed_comments = preprocess_comments(comments)
        save_parquet(processed_comments, output_path)
        processed_frames.append(processed_comments)
        output_paths.append(output_path)
        original_rows += len(comments)

    combined_processed_comments = pd.concat(processed_frames, ignore_index=True)
    report = build_preprocessing_report(
        combined_processed_comments,
        original_rows=original_rows,
    )
    save_preprocessing_report(report, report_path)

    _print_summary(report, output_paths, report_path)


def find_latest_language_audited_comments(directory: Path) -> Path:
    """Return the newest language-audited comments parquet dataset."""
    candidates = find_language_audited_comments(directory)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_language_audited_comments(directory: Path) -> list[Path]:
    """Return all language-audited comments parquet datasets."""
    candidates = sorted(directory.glob("*_comments_language_audited.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No language-audited comments dataset found in {directory}")
    return candidates


def build_processed_comments_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build the output parquet path for a processed comments dataset."""
    dataset_name = input_path.stem
    suffix = "_comments_language_audited"
    if dataset_name.endswith(suffix):
        dataset_name = dataset_name[: -len(suffix)]
    return output_dir / f"{dataset_name}_comments_processed.parquet"


def _print_summary(report: dict[str, Any], output_paths: list[Path], report_path: Path) -> None:
    print("\nPreprocessing summary")
    print(f"Original rows: {report['original_rows']}")
    print(f"Processed rows: {report['processed_rows']}")
    print(f"Spam comments: {report['spam_count']}")
    print(f"Empty comments: {report['empty_count']}")
    print(f"Deleted comments: {report['deleted_count']}")
    print(f"Short comments: {report['short_comment_count']}")
    print("\nSaved processed comments:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print(f"Saved preprocessing report: {report_path}")


if __name__ == "__main__":
    run()
