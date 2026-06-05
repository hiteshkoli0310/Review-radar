"""Run language audit for collected ReviewRadar comments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.config.settings import get_settings
from reviewradar.data_audit import (
    add_detected_language_column,
    audit_comment_languages,
    save_language_audit_report,
)
from reviewradar.utils.io import load_parquet, save_parquet
from reviewradar.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def run() -> None:
    """Audit all collected comment datasets and save language-audited outputs."""
    settings = get_settings()
    configure_logging(settings.log_level)

    comments_paths = find_comment_datasets(settings.paths.raw_dir / "comments")
    report_path = settings.paths.data_dir / "reports" / "language_audit_report.json"

    print("ReviewRadar language audit")

    audited_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    for comments_path in comments_paths:
        output_path = build_language_audited_output_path(
            comments_path,
            settings.paths.interim_dir / "comments",
        )
        print(f"Comment dataset: {comments_path}")
        comments = load_parquet(comments_path)
        audited_comments = add_detected_language_column(comments)
        save_parquet(audited_comments, output_path)
        audited_frames.append(audited_comments)
        output_paths.append(output_path)

    combined_audited_comments = pd.concat(audited_frames, ignore_index=True)
    report = audit_comment_languages(combined_audited_comments)
    save_language_audit_report(report, report_path)

    _print_summary(report, output_paths, report_path)


def find_latest_comments_dataset(directory: Path) -> Path:
    """Return the newest collected comments parquet dataset."""
    candidates = find_comment_datasets(directory)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_comment_datasets(directory: Path) -> list[Path]:
    """Return all collected comments parquet datasets."""
    candidates = sorted(directory.glob("*_comments.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No comments dataset found in {directory}")
    return candidates


def build_language_audited_output_path(comments_path: Path, output_dir: Path) -> Path:
    """Build the output parquet path for a language-audited comments dataset."""
    dataset_name = comments_path.stem
    if dataset_name.endswith("_comments"):
        dataset_name = dataset_name[: -len("_comments")]
    return output_dir / f"{dataset_name}_comments_language_audited.parquet"


def _print_summary(report: dict[str, Any], output_paths: list[Path], report_path: Path) -> None:
    print("\nLanguage audit summary")
    print(f"Total comments: {report['total_comments']}")
    print(f"Unknown language count: {report['unknown_language_count']}")
    print("Language distribution:")
    for language, count in report["language_distribution"].items():
        percentage = report["language_percentages"].get(language, 0.0)
        print(f"  {language}: {count} ({percentage}%)")
    print("\nSaved audited comments:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print(f"Saved language audit report: {report_path}")


if __name__ == "__main__":
    run()
