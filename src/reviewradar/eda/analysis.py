"""Reusable EDA helpers."""

from __future__ import annotations

import pandas as pd


def summarize_text_lengths(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    lengths = frame[column].astype(str).str.len()
    return pd.DataFrame(
        {
            "count": [lengths.count()],
            "mean_length": [lengths.mean()],
            "median_length": [lengths.median()],
        }
    )