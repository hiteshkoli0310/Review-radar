"""Statistics for manually labeled annotation datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def build_annotation_report(dataset: pd.DataFrame) -> dict[str, Any]:
    """Build label coverage and distribution statistics for annotation data."""
    total_rows = int(len(dataset))
    sentiment_counts = _value_counts(dataset, "sentiment_label")
    aspect_counts = _value_counts(dataset, "aspect_label")
    missing_sentiment = _missing_count(dataset, "sentiment_label")
    missing_aspect = _missing_count(dataset, "aspect_label")
    labeled_rows = int(
        total_rows
        - dataset[["sentiment_label", "aspect_label"]]
        .apply(lambda row: row.isna().any() or row.astype(str).str.strip().eq("").any(), axis=1)
        .sum()
        if {"sentiment_label", "aspect_label"}.issubset(dataset.columns)
        else 0
    )

    return {
        "total_rows": total_rows,
        "labeled_rows": labeled_rows,
        "missing_sentiment_labels": missing_sentiment,
        "missing_aspect_labels": missing_aspect,
        "sentiment_counts": sentiment_counts,
        "aspect_counts": aspect_counts,
        **_flatten_counts(sentiment_counts),
        **_flatten_counts(aspect_counts),
    }


def save_annotation_report(report: dict[str, Any], output_path: Path) -> Path:
    """Save annotation statistics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved annotation report to %s", output_path)
    return output_path


def _value_counts(dataset: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in dataset.columns:
        return {}

    values = dataset[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return {str(label): int(count) for label, count in values.value_counts().sort_index().items()}


def _missing_count(dataset: pd.DataFrame, column: str) -> int:
    if column not in dataset.columns:
        return int(len(dataset))

    values = dataset[column].fillna("").astype(str).str.strip()
    return int(values.eq("").sum())


def _flatten_counts(counts: dict[str, int]) -> dict[str, int]:
    return {label.lower().replace(" ", "_"): count for label, count in counts.items()}
