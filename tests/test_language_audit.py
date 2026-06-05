from __future__ import annotations

import pandas as pd

from reviewradar.data_audit.language_audit import audit_comment_languages
from reviewradar.data_audit.language_detector import (
    add_detected_language_column,
    detect_comment_language,
)
from reviewradar.pipelines.run_language_audit import (
    build_language_audited_output_path,
    find_comment_datasets,
)


def test_detect_comment_language_handles_common_review_languages() -> None:
    assert detect_comment_language("This camera is good") == "english"
    assert detect_comment_language("यह फोन अच्छा है") == "hindi"
    assert detect_comment_language("ഈ ഫോൺ നന്നായിട്ടുണ്ട്") == "malayalam"
    assert detect_comment_language("bhai camera bahut good hai") == "hinglish"
    assert detect_comment_language("🔥🔥🔥") == "unknown"
    assert detect_comment_language("great फोन") == "mixed_hindi_english"
    assert detect_comment_language("இந்த phone நல்லது") == "mixed_tamil_english"
    assert detect_comment_language("ఈ phone బాగుంది") == "mixed_telugu_english"
    assert detect_comment_language("এই phone ভালো") == "mixed_bengali_english"


def test_add_detected_language_column_preserves_existing_columns() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c2"],
            "comment_text": ["This is good", "🔥🔥🔥"],
            "like_count": [1, 0],
        }
    )

    audited = add_detected_language_column(frame)

    assert list(audited.columns) == [
        "comment_id",
        "comment_text",
        "like_count",
        "detected_language",
    ]
    assert list(audited["detected_language"]) == ["english", "unknown"]
    assert "detected_language" not in frame.columns


def test_audit_comment_languages_generates_distribution_and_examples() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3"],
            "comment_text": ["This is good", "bhai mast phone hai", "🔥🔥🔥"],
        }
    )
    audited = add_detected_language_column(frame)

    report = audit_comment_languages(audited, examples_per_language=2)

    assert report["total_comments"] == 3
    assert report["language_distribution"] == {
        "english": 1,
        "hinglish": 1,
        "unknown": 1,
    }
    assert report["language_percentages"] == {
        "english": 33.33,
        "hinglish": 33.33,
        "unknown": 33.33,
    }
    assert report["unknown_language_count"] == 1
    assert report["top_examples_per_language"]["hinglish"] == ["bhai mast phone hai"]


def test_language_audited_output_path_uses_interim_comments_directory(tmp_path) -> None:
    comments_path = tmp_path / "data" / "raw" / "comments" / "iphone_17_comments.parquet"
    output_dir = tmp_path / "data" / "interim" / "comments"

    output_path = build_language_audited_output_path(comments_path, output_dir)

    assert output_path == output_dir / "iphone_17_comments_language_audited.parquet"


def test_find_comment_datasets_returns_all_products(tmp_path) -> None:
    comments_dir = tmp_path / "data" / "raw" / "comments"
    comments_dir.mkdir(parents=True)
    first = comments_dir / "iphone_17_comments.parquet"
    second = comments_dir / "steam_deck_comments.parquet"
    first.touch()
    second.touch()

    assert find_comment_datasets(comments_dir) == [first, second]
