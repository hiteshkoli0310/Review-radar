"""Language composition reporting for ReviewRadar comments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.data_audit.language_detector import LANGUAGE_COLUMN


logger = logging.getLogger(__name__)


def audit_comment_languages(
    frame: pd.DataFrame,
    text_column: str = "comment_text",
    examples_per_language: int = 5,
) -> dict[str, Any]:
    """Generate a language audit report from a language-audited comment dataset."""
    logger.info("Generating language audit report for %s comments", len(frame))
    total_comments = int(len(frame))
    language_counts = _language_counts(frame)

    return {
        "total_comments": total_comments,
        "language_distribution": language_counts,
        "language_percentages": _language_percentages(language_counts, total_comments),
        "top_examples_per_language": _top_examples_per_language(
            frame,
            text_column=text_column,
            examples_per_language=examples_per_language,
        ),
        "unknown_language_count": int(language_counts.get("unknown", 0)),
    }


def save_language_audit_report(report: dict[str, Any], output_path: Path) -> Path:
    """Save a language audit report as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved language audit report to %s", output_path)
    return output_path


def _language_counts(frame: pd.DataFrame) -> dict[str, int]:
    if LANGUAGE_COLUMN not in frame.columns:
        return {"unknown": int(len(frame))}

    counts = frame[LANGUAGE_COLUMN].fillna("unknown").value_counts()
    return {str(language): int(count) for language, count in counts.items()}


def _language_percentages(language_counts: dict[str, int], total_comments: int) -> dict[str, float]:
    if total_comments == 0:
        return {language: 0.0 for language in language_counts}

    return {
        language: round(float(count / total_comments * 100), 2)
        for language, count in language_counts.items()
    }


def _top_examples_per_language(
    frame: pd.DataFrame,
    text_column: str,
    examples_per_language: int,
) -> dict[str, list[str]]:
    if LANGUAGE_COLUMN not in frame.columns or text_column not in frame.columns:
        return {}

    examples: dict[str, list[str]] = {}
    for language, group in frame.groupby(LANGUAGE_COLUMN, dropna=False):
        values = (
            group[text_column]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda series: series.ne("")]
            .head(examples_per_language)
            .tolist()
        )
        examples[str(language) if pd.notna(language) else "unknown"] = values
    return examples
