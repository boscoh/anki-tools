"""Greek sentence ranking and text helpers."""

from __future__ import annotations

import re
import unicodedata

from anki_tools.ranking.rank import jaccard_similarity

GREEK_LETTER_PATTERN = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+")


def _tokenize_words_el(text: str) -> list[str]:
    """Tokenize Greek text into words (letters only, casefolded, ς to σ).

    :param text: Sentence or phrase.
    :returns: List of non-empty word tokens.
    """
    cleaned = re.sub(r"<[^>]+>", "", text)
    tokens = GREEK_LETTER_PATTERN.findall(unicodedata.normalize("NFC", cleaned))
    return [token.casefold().replace("ς", "σ") for token in tokens if token]


def greek_text_dedup_key(text: str) -> str:
    """Normalize Greek text for exact deduplication.

    Punctuation and surrounding quotes are ignored, the text is casefolded, NFC
    normalized, and the final sigma maps to sigma.

    :param text: Sentence or phrase.
    :returns: Stable string key for dictionary lookups.
    """
    tokens = _tokenize_words_el(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def word_similarity_el(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity between word sets of two Greek sentences.

    :param sent_a: First sentence.
    :param sent_b: Second sentence.
    :returns: Similarity between 0 and 1.
    """
    words_a = set(_tokenize_words_el(sent_a))
    words_b = set(_tokenize_words_el(sent_b))
    return jaccard_similarity(words_a, words_b)


__all__ = ["greek_text_dedup_key", "word_similarity_el"]
