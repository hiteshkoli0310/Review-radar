from reviewradar.preprocessing.cleaning import clean_comment_text
from reviewradar.preprocessing.text_cleaner import clean_text


def test_clean_comment_text_normalizes_whitespace_and_case() -> None:
    assert clean_comment_text("  Great   Product  ") == "great product"


def test_clean_text_preserves_emoji() -> None:
    assert clean_text("O hail Lord Gabe \U0001F64C") == "o hail lord gabe \U0001F64C"


def test_clean_text_preserves_smart_apostrophe() -> None:
    assert clean_text("I\u2019ve always taken my Switch") == "i\u2019ve always taken my switch"


def test_clean_text_preserves_french_apostrophe() -> None:
    assert clean_text("C\u2019est super de jouer") == "c\u2019est super de jouer"


def test_clean_text_repairs_synthetic_mojibake() -> None:
    mojibake = "f\u00fcr mixtape \U0001F642 spa\u00df".encode("utf-8").decode("latin-1")
    assert clean_text(mojibake) == "f\u00fcr mixtape \U0001F642 spa\u00df"