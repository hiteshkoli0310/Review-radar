"""Orchestrate translation of non-English comments using NLLB."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from reviewradar.preprocessing.text_cleaner import clean_text
from reviewradar.preprocessing.spam_detector import (
    is_empty_comment,
    is_emoji_only,
    is_short_comment,
    is_single_word,
)
from reviewradar.translation.hinglish_normalizer import normalize as normalize_hinglish
from reviewradar.translation.language_map import (
    NLLB_LANGUAGE_CODES,
    TRANSLATABLE_LANGUAGES,
)
from reviewradar.translation.translator import NLLBTranslator

logger = logging.getLogger(__name__)

CLEANED_COL = "cleaned_comment_text"
DETECTED_LANG_COL = "detected_language"
COMMENT_TEXT_COL = "comment_text"

REMOVED_BY_TRANSLATION_COL = "is_removed_by_translation"
IS_TRANSLATED_COL = "is_translated"


def _text_similarity(a: str, b: str) -> float:
    """Compute character-level similarity ratio between two strings.

    Uses a simple approach: length of shared character n-grams.
    """
    a_clean = re.sub(r"\s+", " ", a.strip().lower())
    b_clean = re.sub(r"\s+", " ", b.strip().lower())
    if not a_clean and not b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    # Bigram overlap
    a_bigrams = {a_clean[i:i+2] for i in range(len(a_clean)-1)}
    b_bigrams = {b_clean[i:i+2] for i in range(len(b_clean)-1)}
    if not a_bigrams and not b_bigrams:
        return 1.0
    intersection = a_bigrams & b_bigrams
    union = a_bigrams | b_bigrams
    return len(intersection) / len(union)


def translate_comments(
    frame: pd.DataFrame,
    text_column: str = COMMENT_TEXT_COL,
    detected_language_column: str = DETECTED_LANG_COL,
    cleaned_column: str = CLEANED_COL,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Translate all non-English comments and update *cleaned_column* in place.

    For each row whose ``detected_language`` is not ``"english"``:

    1. The original ``comment_text`` is translated to English via NLLB.
    2. The translated result is cleaned with ``clean_text``.
    3. The result replaces the value in ``cleaned_column``.

    English rows are left untouched.

    Parameters
    ----------
    frame : pd.DataFrame
        Comment dataset with at minimum ``comment_text`` and
        ``detected_language`` columns.
    text_column : str
        Column holding the raw comment text to translate.
    detected_language_column : str
        Column holding the detected language label.
    cleaned_column : str
        Column to overwrite with the translated-and-cleaned text.
    batch_size : int
        How many comments to translate per NLLB call per language.

    Returns
    -------
    pd.DataFrame
        The modified frame with an added ``is_translated`` boolean column.
    """
    translator = NLLBTranslator()
    frame = frame.copy()
    frame[IS_TRANSLATED_COL] = False
    frame[REMOVED_BY_TRANSLATION_COL] = False

    if detected_language_column not in frame.columns:
        logger.warning(
            "Missing detected_language column — skipping translation"
        )
        return frame

    if cleaned_column not in frame.columns:
        frame[cleaned_column] = ""

    non_english_mask = frame[detected_language_column].str.lower() != "english"
    to_translate = frame.loc[non_english_mask].copy()

    if to_translate.empty:
        logger.info("No non-English comments to translate.")
        return frame

    logger.info(
        "Translating %s non-English comments ...", len(to_translate)
    )

    # Group by detected language so we can batch per source language
    lang_map: dict[str, list[str]] = defaultdict(list)
    lang_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in to_translate.iterrows():
        lang = str(row.get(detected_language_column, "")).strip().lower()
        lang_map[lang].append(str(row.get(text_column, "")))
        lang_indices[lang].append(idx)

    failed_count = 0
    for lang, texts in lang_map.items():
        nllb_code = NLLB_LANGUAGE_CODES.get(lang)
        if nllb_code is None and lang not in TRANSLATABLE_LANGUAGES:
            # Try script-based detection for "unknown" language
            sample_texts = [t for t in texts if t]
            if sample_texts:
                from reviewradar.translation.language_map import detect_script_language
                nllb_code = detect_script_language(sample_texts[0])

        if nllb_code is None:
            logger.warning(
                "No NLLB language code for '%s' — marking %s comments as removed",
                lang,
                len(texts),
            )
            for idx in lang_indices[lang]:
                frame.at[idx, REMOVED_BY_TRANSLATION_COL] = True
                frame.at[idx, cleaned_column] = ""
                frame.at[idx, IS_TRANSLATED_COL] = False
            failed_count += len(texts)
            continue

        translations: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                results = translator.translate_batch(batch, nllb_code)
                translations.extend(results)
            except Exception:
                logger.exception(
                    "Batch translation failed for %s (batch %s)", lang, i
                )
                translations.extend(batch)

        for idx, result_text in zip(lang_indices[lang], translations):
            original_raw = str(frame.at[idx, text_column])
            result_text = str(result_text).strip()

            if not result_text or is_empty_comment(result_text):
                frame.at[idx, REMOVED_BY_TRANSLATION_COL] = True
                frame.at[idx, cleaned_column] = ""
                frame.at[idx, IS_TRANSLATED_COL] = False
                failed_count += 1
                continue

            # Check if NLLB actually transformed the text
            similarity = _text_similarity(original_raw, result_text)
            if similarity > 0.85:
                # NLLB failed to translate (e.g., Romanized input)
                if lang == "hinglish":
                    result_text = normalize_hinglish(
                        original_raw, nllb_failed=True
                    )
                elif lang == "romanized_malayalam":
                    # Try native-script Malayalam transliteration
                    result_text = normalize_hinglish(
                        original_raw, nllb_failed=True
                    )
                else:
                    # Mark as failed for other languages
                    frame.at[idx, REMOVED_BY_TRANSLATION_COL] = True
                    frame.at[idx, cleaned_column] = ""
                    frame.at[idx, IS_TRANSLATED_COL] = False
                    failed_count += 1
                    continue

            cleaned_result = clean_text(result_text)
            frame.at[idx, cleaned_column] = cleaned_result
            frame.at[idx, IS_TRANSLATED_COL] = True

            if is_empty_comment(cleaned_result):
                frame.at[idx, REMOVED_BY_TRANSLATION_COL] = True

    logger.info(
        "Translation complete. %s succeeded, %s failed.",
        len(to_translate) - failed_count,
        failed_count,
    )
    return frame


def audit_translation_quality(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build a quality report on the translation pass."""
    total = len(frame)
    translated = frame[IS_TRANSLATED_COL].sum() if IS_TRANSLATED_COL in frame.columns else 0
    removed_by_translation = frame[REMOVED_BY_TRANSLATION_COL].sum() if REMOVED_BY_TRANSLATION_COL in frame.columns else 0

    language_counts: dict[str, int] = {}
    if DETECTED_LANG_COL in frame.columns:
        lang_counts = frame.loc[frame[IS_TRANSLATED_COL] == True, DETECTED_LANG_COL].value_counts()
        language_counts = {
            k: int(v) for k, v in lang_counts.items()
        }

    return {
        "total_comments": int(total),
        "translated_comments": int(translated),
        "removed_by_translation_failure": int(removed_by_translation),
        "translation_language_breakdown": language_counts,
    }


def final_cleanup(
    frame: pd.DataFrame,
    cleaned_column: str = CLEANED_COL,
    min_word_count: int = 2,
    min_character_count: int = 3,
) -> pd.DataFrame:
    """Remove remaining low-quality comments after translation.

    Drops rows where the cleaned/translated text is:
    - Empty
    - Emoji-only (no alphanumeric characters)
    - URL-only
    - Single word
    - Shorter than *min_character_count* characters
    - Shorter than *min_word_count* words

    Parameters
    ----------
    frame : pd.DataFrame
    cleaned_column : str
    min_word_count : int
    min_character_count : int

    Returns
    -------
    pd.DataFrame
        Filtered frame with an added ``is_removed_in_final_cleanup`` column.
    """
    frame = frame.copy()
    removal_col = "is_removed_in_final_cleanup"

    if cleaned_column not in frame.columns:
        frame[removal_col] = False
        return frame

    texts = frame[cleaned_column].astype(str)

    is_empty = texts.apply(is_empty_comment)
    is_emoji = texts.apply(lambda t: is_emoji_only(t))
    is_url_only = texts.apply(lambda t: t.strip().startswith("http"))
    short_char = texts.apply(lambda t: len(t.strip()) < min_character_count)
    short_word = texts.apply(lambda t: is_single_word(t))

    frame[removal_col] = (
        is_empty
        | is_emoji
        | is_url_only
        | short_char
        | short_word
    )

    kept = frame[~frame[removal_col]].reset_index(drop=True)
    logger.info(
        "Final cleanup: %s removed, %s kept",
        frame[removal_col].sum(),
        len(kept),
    )
    return kept


def build_annotation_readiness_report(
    original_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    translation_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a comprehensive annotation-readiness report."""
    before = len(original_frame)
    after = len(cleaned_frame)
    removed_total = before - after

    removed_by_translation = translation_report.get(
        "removed_by_translation_failure", 0
    )
    removed_by_final_cleanup = removed_total - removed_by_translation

    language_dist: dict[str, int] = {}
    if DETECTED_LANG_COL in cleaned_frame.columns:
        language_dist = {
            k: int(v)
            for k, v in cleaned_frame[DETECTED_LANG_COL]
            .value_counts()
            .items()
        }

    only_english = (
        cleaned_frame[DETECTED_LANG_COL].str.lower() == "english"
    ).all() if DETECTED_LANG_COL in cleaned_frame.columns else False

    min_chars = int(cleaned_frame[CLEANED_COL].astype(str).str.len().min())
    min_words = int(
        cleaned_frame[CLEANED_COL]
        .astype(str)
        .str.split()
        .apply(len)
        .min()
    )

    return {
        "annotation_readiness": "READY" if after > 0 else "EMPTY_DATASET",
        "rows_before_cleanup": before,
        "rows_after_cleanup": after,
        "rows_removed": removed_total,
        "removed_by_translation_failure": removed_by_translation,
        "removed_by_final_cleanup": removed_by_final_cleanup,
        "all_comments_english": only_english,
        "language_distribution": language_dist,
        "minimum_cleaned_length_chars": min_chars,
        "minimum_cleaned_word_count": min_words,
    }
