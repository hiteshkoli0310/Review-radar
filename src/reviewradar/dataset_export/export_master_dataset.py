"""Export ReviewRadar master datasets and reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def export_master_dataset(master_dataset: pd.DataFrame, output_path: Path) -> Path:
    """Save a master dataset as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_dataset.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved master dataset to %s", output_path)
    return output_path


def save_master_dataset_report(report: dict[str, Any], output_path: Path) -> Path:
    """Save a master dataset export report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved master dataset report to %s", output_path)
    return output_path
