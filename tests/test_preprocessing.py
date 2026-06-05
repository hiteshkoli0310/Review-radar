from __future__ import annotations

import pandas as pd

from reviewradar.pipelines.run_preprocessing import (
    build_processed_comments_output_path,
    find_language_audited_comments,
)
from reviewradar.preprocessing.preprocessing_pipeline import (
    build_preprocessing_report,
    preprocess_comments,
)
from reviewradar.preprocessing.spam_detector import is_spam_comment
from reviewradar.preprocessing.text_cleaner import clean_text


def test_clean_text_normalizes_without_removing_sentiment_words() -> None:
    text = "<b>NOT bad!!!!!!!</b> Check this out https://example.com"

    assert clean_text(text) == "not bad! check this out"


def test_clean_text_handles_unicode_whitespace_and_repeated_punctuation() -> None:
    text = "Great\u00a0phone?????   Amazing!!!!"

    assert clean_text(text) == "great phone? amazing!"


def test_spam_detector_flags_urls_promotions_emoji_and_repeated_characters() -> None:
    assert is_spam_comment("visit https://a.com and https://b.com")
    assert is_spam_comment("Subscribe to my channel for giveaway")
    assert is_spam_comment("🔥🔥🔥")
    assert is_spam_comment("goooooood")
    assert not is_spam_comment("not bad, good camera")


def test_preprocess_comments_preserves_rows_and_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3", "c4", "c5"],
            "comment_text": [
                "Great phone!!!!",
                "",
                "[deleted]",
                "ok",
                "visit https://a.com https://b.com",
            ],
            "detected_language": ["english", "unknown", "english", "english", "english"],
        }
    )
    original_columns = list(frame.columns)

    processed = preprocess_comments(frame)

    assert len(processed) == len(frame)
    assert original_columns == list(frame.columns)
    assert processed.loc[0, "cleaned_comment_text"] == "great phone!"
    assert processed.loc[1, "is_empty"]
    assert processed.loc[2, "is_deleted"]
    assert processed.loc[3, "is_short_comment"]
    assert processed.loc[4, "is_spam"]
    assert "detected_language" in processed.columns


def test_preprocessing_report_counts_flags() -> None:
    processed = pd.DataFrame(
        {
            "is_spam": [True, False, True],
            "is_empty": [False, True, False],
            "is_deleted": [False, False, True],
            "is_short_comment": [False, True, True],
        }
    )

    report = build_preprocessing_report(processed, original_rows=3)

    assert report == {
        "original_rows": 3,
        "processed_rows": 3,
        "spam_count": 2,
        "empty_count": 1,
        "deleted_count": 1,
        "short_comment_count": 2,
    }


def test_processed_output_path_uses_processed_comments_directory(tmp_path) -> None:
    input_path = (
        tmp_path
        / "data"
        / "interim"
        / "comments"
        / "nintendo_switch_comments_language_audited.parquet"
    )
    output_dir = tmp_path / "data" / "processed" / "comments"

    output_path = build_processed_comments_output_path(input_path, output_dir)

    assert output_path == output_dir / "nintendo_switch_comments_processed.parquet"


def test_find_language_audited_comments_returns_all_products(tmp_path) -> None:
    comments_dir = tmp_path / "data" / "interim" / "comments"
    comments_dir.mkdir(parents=True)
    first = comments_dir / "iphone_17_comments_language_audited.parquet"
    second = comments_dir / "steam_deck_comments_language_audited.parquet"
    first.touch()
    second.touch()

    assert find_language_audited_comments(comments_dir) == [first, second]
