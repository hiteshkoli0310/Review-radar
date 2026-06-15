"""Translation package for multilingual comment processing."""

from reviewradar.translation.language_map import (
    NLLB_LANGUAGE_CODES,
    TARGET_LANGUAGE_CODE,
    TRANSLATABLE_LANGUAGES,
)
from reviewradar.translation.translator import NLLBTranslator

__all__ = [
    "NLLB_LANGUAGE_CODES",
    "NLLBTranslator",
    "TARGET_LANGUAGE_CODE",
    "TRANSLATABLE_LANGUAGES",
]
