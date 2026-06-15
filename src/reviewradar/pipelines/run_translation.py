"""Translate non-English comments and produce annotation-ready dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.config.settings import get_settings
from reviewradar.translation.translation_pipeline import (
    audit_translation_quality,
    build_annotation_readiness_report,
    final_cleanup,
    translate_comments,
)
from reviewradar.utils.io import load_parquet, save_parquet
from reviewradar.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def run() -> None:
    """Translate all non-English comments and build annotation-ready dataset."""
    settings = get_settings()
    configure_logging(settings.log_level)

    processed_comments_dir = settings.paths.processed_dir / "comments"
    output_dir = settings.paths.processed_dir / "comments"
    readiness_report_path = (
        settings.paths.data_dir / "reports" / "annotation_readiness_report.json"
    )

    input_paths = sorted(processed_comments_dir.glob("*_comments_processed.parquet"))
    if not input_paths:
        raise FileNotFoundError(
            f"No processed comment files found in {processed_comments_dir}"
        )

    print("ReviewRadar translation pipeline")
    print(f"Found {len(input_paths)} processed comment files.")

    all_original_frames: list[pd.DataFrame] = []
    all_translated_frames: list[pd.DataFrame] = []
    combined_before = 0
    combined_after = 0

    for input_path in input_paths:
        print(f"\nProcessing: {input_path.name}")
        original = load_parquet(input_path)
        original_rows = len(original)
        all_original_frames.append(original)

        frame = translate_comments(
            original,
            text_column="comment_text",
            detected_language_column="detected_language",
            cleaned_column="cleaned_comment_text",
            batch_size=16,
        )

        translation_report = audit_translation_quality(frame)
        print(
            f"  Translated: {translation_report['translated_comments']}, "
            f"Removed by failure: {translation_report['removed_by_translation_failure']}"
        )

        cleaned = final_cleanup(frame)
        combined_before += original_rows
        combined_after += len(cleaned)

        output_path = output_dir / input_path.name
        save_parquet(cleaned, output_path)
        all_translated_frames.append(cleaned)
        print(f"  Saved: {output_path.name} ({len(cleaned)} rows)")

    combined_all_original = (
        pd.concat(all_original_frames, ignore_index=True)
        if all_original_frames
        else pd.DataFrame()
    )
    combined = (
        pd.concat(all_translated_frames, ignore_index=True)
        if all_translated_frames
        else pd.DataFrame()
    )

    readiness = build_annotation_readiness_report(
        original_frame=combined_all_original,
        cleaned_frame=combined,
        translation_report=audit_translation_quality(combined),
    )

    readiness_report_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_report_path.write_text(
        json.dumps(readiness, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved annotation-readiness report to %s", readiness_report_path)

    _print_summary(readiness, input_paths, readiness_report_path)


def _print_summary(
    report: dict[str, Any],
    input_paths: list[Path],
    report_path: Path,
) -> None:
    print("\n" + "=" * 60)
    print("ANNOTATION READINESS REPORT")
    print("=" * 60)
    print(f"Dataset ready for annotation: {report['annotation_readiness']}")
    print(f"Total rows before: {report['rows_before_cleanup']}")
    print(f"Total rows after : {report['rows_after_cleanup']}")
    print(f"Rows removed     : {report['rows_removed']}")
    print(f"  - Translation failures : {report['removed_by_translation_failure']}")
    print(f"  - Final cleanup        : {report['removed_by_final_cleanup']}")
    print(f"All comments English     : {report['all_comments_english']}")
    print(f"Min cleaned length       : {report['minimum_cleaned_length_chars']} chars")
    print(f"Min cleaned word count   : {report['minimum_cleaned_word_count']} words")
    print()
    print("Language distribution after cleanup:")
    for lang, count in sorted(report["language_distribution"].items()):
        print(f"  {lang}: {count}")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run()
