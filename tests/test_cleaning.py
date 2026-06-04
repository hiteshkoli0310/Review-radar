from reviewradar.preprocessing.cleaning import clean_comment_text


def test_clean_comment_text_normalizes_whitespace_and_case() -> None:
    assert clean_comment_text("  Great   Product  ") == "great product"