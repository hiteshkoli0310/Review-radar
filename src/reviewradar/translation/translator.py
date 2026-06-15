"""NLLB-based offline translator for multilingual comments."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from reviewradar.translation.language_map import (
    NLLB_LANGUAGE_CODES,
    TARGET_LANGUAGE_CODE,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 512


class NLLBTranslator:
    """Offline translator using Meta's NLLB-200 model.

    Loads the model on first call (lazy) and caches it module-wide.
    """

    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._model is not None and cls._tokenizer is not None:
            return

        logger.info("Loading NLLB translation model %s ...", DEFAULT_MODEL_NAME)
        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        dtype = torch.float16 if cls._device == "cuda" else torch.float32
        cls._model = AutoModelForSeq2SeqLM.from_pretrained(
            DEFAULT_MODEL_NAME,
            dtype=dtype,
        )
        cls._model = cls._model.to(cls._device)
        cls._model.eval()
        logger.info("NLLB model loaded on %s", cls._device)

    def translate(self, text: str, source_lang: str) -> str:
        """Translate a single text from *source_lang* to English.

        Parameters
        ----------
        text : str
            The text to translate.
        source_lang : str
            The NLLB source language code (e.g. ``"hin_Deva"``).

        Returns
        -------
        str
            Translated text, or the original text on failure.
        """
        return self.translate_batch([text], source_lang)[0]

    def translate_batch(self, texts: list[str], source_lang: str) -> list[str]:
        """Translate a batch of texts from *source_lang* to English."""
        self._ensure_loaded()

        if not texts:
            return []

        cleaned = [str(t) for t in texts]
        try:
            self._tokenizer.src_lang = source_lang
            inputs = self._tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to(self._device)

            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(
                TARGET_LANGUAGE_CODE
            )
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=MAX_OUTPUT_LENGTH,
                    num_beams=4,
                )

            results = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            return results
        except Exception:
            logger.exception(
                "NLLB translation failed for batch (lang=%s)", source_lang
            )
            return cleaned
