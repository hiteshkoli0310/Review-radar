from __future__ import annotations

import pandas as pd

from reviewradar.annotation.annotation_dataset_builder import (
    ANNOTATION_COLUMNS,
    ANNOTATION_METADATA_COLUMNS,
    ANNOTATION_NOTE_LABELS,
    build_annotation_dataset,
    write_annotation_guidelines,
)
from reviewradar.annotation.annotation_statistics import build_annotation_report
from reviewradar.annotation.sample_generator import generate_balanced_sample


def test_generate_balanced_sample_is_product_balanced_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "product_query": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "comment_id": [f"c{index}" for index in range(60)],
            "comment_text": [f"comment {index}" for index in range(60)],
        }
    )

    first = generate_balanced_sample(frame, sample_size=30, random_state=7)
    second = generate_balanced_sample(frame, sample_size=30, random_state=7)

    assert first["comment_id"].tolist() == second["comment_id"].tolist()
    assert first["product_query"].value_counts().to_dict() == {
        "A": 10,
        "B": 10,
        "C": 10,
    }


def test_generate_balanced_sample_redistributes_when_product_has_few_rows() -> None:
    frame = pd.DataFrame(
        {
            "product_query": ["A"] * 2 + ["B"] * 20 + ["C"] * 20,
            "comment_id": [f"c{index}" for index in range(42)],
        }
    )

    sample = generate_balanced_sample(frame, sample_size=12, random_state=42)

    assert len(sample) == 12
    assert sample["product_query"].value_counts()["A"] == 2


def test_build_annotation_dataset_preserves_metadata_and_adds_empty_label_columns() -> None:
    sample = pd.DataFrame(
        {
            "product_query": ["Nintendo Switch"],
            "video_id": ["v1"],
            "video_title": ["Review"],
            "channel_name": ["Channel"],
            "comment_id": ["c1"],
            "comment_text": ["OLED screen is amazing"],
            "cleaned_comment_text": ["oled screen is amazing"],
            "comment_like_count": [12],
            "detected_language": ["english"],
            "extra_column": ["ignored"],
        }
    )

    dataset = build_annotation_dataset(sample)

    assert list(dataset.columns) == ANNOTATION_METADATA_COLUMNS + ANNOTATION_COLUMNS
    assert dataset.loc[0, "sentiment_label"] == ""
    assert dataset.loc[0, "aspect_label"] == ""
    assert dataset.loc[0, "review_notes"] == ""
    for label in ANNOTATION_NOTE_LABELS:
        col = f"is_{label.lower().replace(' ', '_')}"
        assert col in dataset.columns
        assert dataset.loc[0, col] == 0 or dataset.loc[0, col] == ""
    assert "extra_column" not in dataset.columns


def test_build_annotation_report_counts_labels_and_missing_values() -> None:
    dataset = pd.DataFrame(
        {
            "sentiment_label": ["Positive", "Neutral", "Negative", ""],
            "aspect_label": ["Display", "Price", "", ""],
        }
    )

    report = build_annotation_report(dataset)

    assert report["total_rows"] == 4
    assert report["labeled_rows"] == 2
    assert report["missing_sentiment_labels"] == 1
    assert report["missing_aspect_labels"] == 2
    assert report["positive"] == 1
    assert report["neutral"] == 1
    assert report["negative"] == 1
    assert report["display"] == 1
    assert report["price"] == 1


def test_write_annotation_guidelines_creates_expected_sections(tmp_path) -> None:
    output_path = tmp_path / "annotation_guidelines.md"

    write_annotation_guidelines(output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "Allowed Sentiment Labels" in text
    assert "Positive" in text
    assert "Purchase Intent" in text
    assert "Other" in text
