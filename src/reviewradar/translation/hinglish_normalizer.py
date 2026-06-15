"""Normalize Romanized Hindi (Hinglish) text into English.

Uses a dictionary-based approach to map common Hinglish words and
patterns to their English equivalents. Falls back to NLLB for larger
pure-Hindi phrases.
"""

from __future__ import annotations

import re
from typing import Any

_HINGLISH_WORD_MAP: dict[str, str] = {
    # Pronouns
    "mein": "i",
    "mujhe": "i",
    "maine": "i",
    "hum": "we",
    "tum": "you",
    "aap": "you",
    "tera": "your",
    "teri": "your",
    "tere": "your",
    "mera": "my",
    "meri": "my",
    "mere": "my",
    "uska": "his",
    "uski": "his",
    "unka": "their",
    "unki": "their",
    "kisika": "someone",
    "kisiki": "someone",
    # Question words
    "kya": "what",
    "kyo": "why",
    "kab": "when",
    "kaha": "where",
    "kahan": "where",
    "kaise": "how",
    "konsa": "which",
    "konse": "which",
    "kis": "which",
    "kitna": "how much",
    "kitne": "how many",
    "kitni": "how many",
    "kaun": "who",
    "kaunsa": "which",
    # Verbs
    "hai": "is",
    "hain": "are",
    "ho": "are",
    "hu": "am",
    "tha": "was",
    "the": "were",
    "thi": "was",
    "thhi": "was",
    "thhe": "were",
    "hoga": "will be",
    "hogee": "will be",
    "hogi": "will be",
    "honge": "will be",
    "rahega": "will be",
    "rahegi": "will be",
    "rahenge": "will be",
    "rahe": "stay",
    "chahiye": "want",
    "chahta": "want",
    "chahti": "want",
    "chahte": "want",
    "karta": "do",
    "karti": "do",
    "karte": "do",
    "karo": "do",
    "kar": "do",
    "kiya": "did",
    "kare": "do",
    "karenge": "will do",
    "karna": "to do",
    "de": "give",
    "dega": "will give",
    "dene": "to give",
    "le": "take",
    "lena": "to take",
    "lo": "take",
    "aana": "to come",
    "aata": "comes",
    "aati": "comes",
    "aate": "come",
    "jaana": "to go",
    "jaata": "goes",
    "jaati": "goes",
    "jaate": "go",
    "jata": "goes",
    "jati": "goes",
    "jate": "go",
    "bol": "say",
    "bolo": "say",
    "bolta": "says",
    "bolti": "says",
    "bolte": "say",
    "bata": "tell",
    "batao": "tell",
    "bataye": "tell",
    "samajh": "understand",
    "samajhta": "understands",
    "samajhte": "understand",
    "rakha": "keep",
    "rakhe": "keep",
    "rakho": "keep",
    "rakhta": "keeps",
    "rakhti": "keeps",
    "rakhte": "keep",
    "lagta": "feels",
    "lagti": "feels",
    "lagte": "feel",
    "lagna": "to feel",
    "hoja": "happen",
    "hojata": "happens",
    "hojayega": "will happen",
    # Conjunctions & postpositions
    "aur": "and",
    "ya": "or",
    "lekin": "but",
    "magar": "but",
    "par": "but",
    "toh": "so",
    "tab": "then",
    "agar": "if",
    "kyuki": "because",
    "kyunki": "because",
    "isliye": "so",
    "warna": "otherwise",
    "ke liye": "for",
    "ki liye": "for",
    "ke saath": "with",
    "ke baad": "after",
    "ke pehle": "before",
    "ke andar": "inside",
    "ke bahar": "outside",
    "ke uper": "above",
    "ke upar": "above",
    "ke neeche": "below",
    "ke niche": "below",
    "ke sath": "with",
    "ki tarah": "like",
    "ki taraf": "towards",
    # Adverbs
    "bahut": "very",
    "kaafi": "quite",
    "thoda": "a little",
    "thodi": "a little",
    "thode": "a little",
    "zyada": "too much",
    "zada": "too much",
    "jada": "too much",
    "kam": "less",
    "kuch": "some",
    "kuch nahi": "nothing",
    "kuch bhi": "anything",
    "sab": "all",
    "pure": "full",
    "puri": "full",
    "pura": "full",
    "abhi": "now",
    "ab": "now",
    "aaj": "today",
    "kal": "tomorrow",
    "parson": "day after",
    "hamesha": "always",
    "kabhi": "sometimes",
    "kabhi kabhi": "sometimes",
    "aksar": "often",
    "generally": "generally",
    # Negations
    "nahi": "not",
    "na": "no",
    "mat": "dont",
    "bina": "without",
    # Demonstratives
    "yeh": "this",
    "ye": "this",
    "yee": "this",
    "woh": "that",
    "wo": "that",
    "voh": "that",
    "un": "those",
    "yaha": "here",
    "yahan": "here",
    "waha": "there",
    "wahan": "there",
    "idhar": "here",
    "udhar": "there",
    # Nouns / misc
    "baat": "thing",
    "chij": "thing",
    "cheez": "thing",
    "kaam": "work",
    "samay": "time",
    "vakt": "time",
    "waqt": "time",
    "baar": "time",
    "dafa": "time",
    "sawaal": "question",
    "jawab": "answer",
    "sach": "true",
    "sahi": "correct",
    "galat": "wrong",
    "accha": "good",
    "acha": "good",
    "acchi": "good",
    "acche": "good",
    "kharab": "bad",
    "bura": "bad",
    "buri": "bad",
    "bure": "bad",
    "chota": "small",
    "choti": "small",
    "chote": "small",
    "bada": "big",
    "badi": "big",
    "bade": "big",
    "naya": "new",
    "naye": "new",
    "nayi": "new",
    "purana": "old",
    "purane": "old",
    "purani": "old",
    "sasta": "cheap",
    "saste": "cheap",
    "sasti": "cheap",
    "mehanga": "expensive",
    "mehenga": "expensive",
    "mahanga": "expensive",
    "mast": "great",
    "maza": "fun",
    "mazaa": "fun",
    "maje": "fun",
    "shandar": "excellent",
    "ganda": "bad",
    "gandi": "bad",
    "gande": "bad",
    "behtar": "better",
    "behtareen": "best",
    "accha hai": "is good",
    "bura hai": "is bad",
    "sahi hai": "is correct",
    "galat hai": "is wrong",
    # Social / address
    "bhai": "brother",
    "dost": "friend",
    "yaar": "friend",
    "bhaiya": "brother",
}

# Sort by length descending so longer phrases match first
_HINGLISH_ITEMS = sorted(_HINGLISH_WORD_MAP.items(), key=lambda x: -len(x[0]))


def normalize(text: str, nllb_failed: bool = True) -> str:
    """Normalize Hinglish text to English.

    Parameters
    ----------
    text : str
        Lowercased Hinglish comment text.
    nllb_failed : bool
        Whether NLLB was unable to translate this text (i.e. the output
        was the same as input).  When True, applies word-level
        replacement.

    Returns
    -------
    str
        Normalized English text.
    """
    if not nllb_failed:
        return text

    words = text.strip().split()
    if not words:
        return text

    # Replace multi-word phrases first
    result = text.lower()
    for hinglish_word, english_word in _HINGLISH_ITEMS:
        # Word-boundary replacement for each phrase/word
        pattern = re.compile(r"\b" + re.escape(hinglish_word) + r"\b", re.IGNORECASE)
        result = pattern.sub(english_word, result)

    return result


def is_hinglish_dominated(text: str) -> bool:
    """Check if text has a significant proportion of Hinglish vocabulary."""
    if not text:
        return False
    words = text.strip().lower().split()
    if not words:
        return False
    hinglish_count = sum(
        1 for w in words if w in _HINGLISH_WORD_MAP
    )
    ratio = hinglish_count / len(words)
    return ratio >= 0.15
