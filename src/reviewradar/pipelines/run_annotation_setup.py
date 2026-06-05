"""Create a human annotation dataset and labeling guidelines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reviewradar.annotation import (
    build_annotation_dataset,
    build_annotation_report,
    generate_balanced_sample,
    save_annotation_dataset,
    save_annotation_report,
    write_annotation_guidelines,
)
from reviewradar.config.settings import get_settings
from reviewradar.utils.logging import configure_logging


DEFAULT_SAMPLE_SIZE = 300
DEFAULT_RANDOM_STATE = 42


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, random_state: int = DEFAULT_RANDOM_STATE) -> None:
    """Build the default manual annotation sample and guidelines."""
    settings = get_settings()
    configure_logging(settings.log_level)

    master_path = settings.paths.data_dir / "exports" / "reviewradar_master_raw.csv"
    annotation_dir = settings.paths.data_dir / "annotation"
    sample_path = annotation_dir / "manual_review_sample.csv"
    guidelines_path = annotation_dir / "annotation_guidelines.md"
    report_path = settings.paths.data_dir / "reports" / "annotation_report.json"

    master_dataset = load_master_dataset(master_path)
    sample = generate_balanced_sample(
        master_dataset,
        sample_size=sample_size,
        random_state=random_state,
    )
    annotation_dataset = build_annotation_dataset(sample)
    report = build_annotation_report(annotation_dataset)

    save_annotation_dataset(annotation_dataset, sample_path)
    write_annotation_guidelines(guidelines_path)
    save_annotation_report(report, report_path)
    _print_summary(annotation_dataset, sample_path, guidelines_path, report_path)


def load_master_dataset(master_path: Path) -> pd.DataFrame:
    """Load the master ReviewRadar CSV dataset."""
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")
    return pd.read_csv(master_path)


def _print_summary(
    annotation_dataset: pd.DataFrame,
    sample_path: Path,
    guidelines_path: Path,
    report_path: Path,
) -> None:
    print("ReviewRadar annotation setup")
    print(f"Rows ready for annotation: {len(annotation_dataset)}")
    if "product_query" in annotation_dataset.columns:
        print("Product distribution:")
        for product, count in annotation_dataset["product_query"].value_counts().sort_index().items():
            print(f"  {product}: {count}")
    print(f"\nSaved annotation sample: {sample_path}")
    print(f"Saved annotation guidelines: {guidelines_path}")
    print(f"Saved annotation report: {report_path}")


if __name__ == "__main__":
    run()
