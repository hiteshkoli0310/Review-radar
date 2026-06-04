"""Reusable parquet saving utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe_to_parquet(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a pandas DataFrame as a parquet file and create folders as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataframe.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save parquet dataset to {path}") from exc

    return path

