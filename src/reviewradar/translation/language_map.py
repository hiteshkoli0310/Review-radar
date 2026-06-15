"""Mappings from detected_language to NLLB FLORES-200 language codes."""

from __future__ import annotations

import re
import unicodedata

NLLB_LANGUAGE_CODES: dict[str, str] = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "hinglish": "hin_Deva",
    "arabic": "ara_Arab",
    "russian": "rus_Cyrl",
    "german": "deu_Latn",
    "french": "fra_Latn",
    "spanish": "spa_Latn",
    "malayalam": "mal_Mlym",
    "tamil": "tam_Taml",
    "kannada": "kan_Knda",
    "gujarati": "guj_Gujr",
    "mixed_malayalam_english": "mal_Mlym",
    "mixed_kannada_english": "kan_Knda",
    "mixed_tamil_english": "tam_Taml",
    "romanized_malayalam": "mal_Mlym",
}

TARGET_LANGUAGE_CODE = "eng_Latn"

TRANSLATABLE_LANGUAGES = frozenset(NLLB_LANGUAGE_CODES.keys()) - {"english"}

# Heuristic script-based language detection for "unknown" labels
_SCRIPT_TO_NLLB: dict[str, str] = {
    "Cyrillic": "rus_Cyrl",
    "Arabic": "ara_Arab",
    "Devanagari": "hin_Deva",
    "Malayalam": "mal_Mlym",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Gujarati": "guj_Gujr",
}


def detect_script_language(text: str) -> str | None:
    """Detect the script of *text* and return the best NLLB language code.

    Returns ``None`` when no script-based match is found.
    """
    if not text:
        return None

    # Check a sample of characters (first 200 non-whitespace)
    chars = re.sub(r"\s", "", text)[:200]
    script_counts: dict[str, int] = {}
    for ch in chars:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            script = _get_script_name(ch)
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return None

    dominant = max(script_counts, key=script_counts.get)
    threshold = max(2, len(chars) * 0.2)
    if script_counts[dominant] >= threshold:
        return _SCRIPT_TO_NLLB.get(dominant)
    return None


def _get_script_name(ch: str) -> str | None:
    """Return the script name of a single character."""
    try:
        name = unicodedata.name(ch, "")
        if not name:
            return None
        # Script is typically the second word in the Unicode name
        # e.g. "DEVANAGARI LETTER KA" -> "Devanagari"
        parts = name.split()
        if parts and parts[0] in _SCRIPT_RANGES:
            return parts[0].title()
        return None
    except ValueError:
        return None


# Known script identifiers from Unicode character names
_SCRIPT_RANGES = frozenset({
    "CYRILLIC",
    "ARABIC",
    "DEVANAGARI",
    "MALAYALAM",
    "TAMIL",
    "KANNADA",
    "GUJARATI",
    "GREEK",
    "HEBREW",
    "THAI",
    "CHINESE",
    "JAPANESE",
    "KOREAN",
    "HANGUL",
})
