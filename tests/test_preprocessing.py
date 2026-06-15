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


def test_clean_text_repairs_common_mojibake() -> None:
    mojibake = "für mixtape 🙂 spaß".encode("utf-8").decode("latin-1")

    assert clean_text(mojibake) == "für mixtape 🙂 spaß"


def test_spam_detector_flags_urls_promotions_emoji_and_repeated_characters() -> None:
    assert is_spam_comment("visit https://a.com and https://b.com")
    assert is_spam_comment("Subscribe to my channel for giveaway")
    assert is_spam_comment("🔥🔥🔥")
    assert is_spam_comment("goooooood")
    assert not is_spam_comment("not bad, good camera")


def test_spam_detector_flags_solicitation_keywords() -> None:
    assert is_spam_comment("Bhai please give steam deck please dedo")
    assert is_spam_comment("Gameloop sa pubg mobile kalka dikao plz")
    assert is_spam_comment("Anna phone kodi nanu Swiggy")
    assert is_spam_comment("Brother 16 plus comparison kudunga brother")
    assert is_spam_comment("unboxing video madi bro")
    assert not is_spam_comment("can you compare iPhone 16 vs 17")


def test_preprocess_comments_filters_low_value_rows_and_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            "comment_text": [
                "Great phone!!!!",
                "",
                "[deleted]",
                "ok",
                "visit https://a.com https://b.com",
                "Switch",
                "https://youtube.com/shorts/HWHd5mjYmto?feature=share",
            ],
            "detected_language": [
                "english",
                "unknown",
                "english",
                "english",
                "english",
                "english",
                "english",
            ],
        }
    )
    original_columns = list(frame.columns)

    processed = preprocess_comments(frame)

    assert len(processed) == 1
    assert original_columns == list(frame.columns)
    assert processed.loc[0, "cleaned_comment_text"] == "great phone!"
    assert not bool(processed.loc[0, "is_empty"])
    assert not bool(processed.loc[0, "is_deleted"])
    assert not bool(processed.loc[0, "is_short_comment"])
    assert not bool(processed.loc[0, "is_single_word"])
    assert not bool(processed.loc[0, "is_spam"])
    assert "detected_language" in processed.columns
    assert processed.attrs["preprocessing_summary"]["empty_count"] == 2
    assert processed.attrs["preprocessing_summary"]["deleted_count"] == 1
    assert processed.attrs["preprocessing_summary"]["single_word_count"] == 4
    assert processed.attrs["preprocessing_summary"]["spam_count"] == 2


def test_preprocessing_report_counts_flags() -> None:
    processed = pd.DataFrame(
        {
            "is_spam": [True, False, True],
            "is_empty": [False, True, False],
            "is_deleted": [False, False, True],
            "is_short_comment": [False, True, True],
            "is_single_word": [False, True, True],
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
        "single_word_count": 2,
        "rows_removed_by_cleaning": 0,
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
