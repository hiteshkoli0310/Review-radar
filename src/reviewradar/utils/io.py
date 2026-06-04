"""Shared I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    frame.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)