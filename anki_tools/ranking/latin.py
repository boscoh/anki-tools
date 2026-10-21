"""Shared text helpers of the Latin-script languages.

The tokenizer serves the languages that use a Latin script: French and Spanish.
A language package uses it for its word set, its complexity score, and its
dedupe key. The helpers hold no language data.
"""

from __future__ import annotations

import re

LETTER_PATTERN = re.compile(r"[a-zA-Z\u00c0-\u024f]+")


def tokenize_latin(text: str) -> list[str]:
    """Split text into lowercase letter tokens.

    Markup tags are removed first, so a card field keeps only its words.

    :param text: Sentence or phrase.
    :returns: Non-empty word tokens, lowercased.
    """

    cleaned = re.sub(r"<[^>]+>", "", text)
    tokens = LETTER_PATTERN.findall(cleaned)
    return [token.lower() for token in tokens if token]


def latin_text_dedup_key(text: str) -> str:
    """Give the exact dedupe key of a Latin-script sentence.

    Punctuation and surrounding quotes are ignored, so ``Hi!`` and ``Hi`` give
    the same key. Text with no letter token falls back to the casefolded
    whitespace-collapsed text.

    :param text: Sentence or phrase.
    :returns: Stable string key for a dictionary lookup.
    """

    tokens = tokenize_latin(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(text.split()).casefold()


def latin_word_similarity(sent_a: str, sent_b: str) -> float:
    """Give the Jaccard similarity of two Latin-script word sets.

    :param sent_a: First sentence.
    :param sent_b: Second sentence.
    :returns: Similarity between 0 and 1.
    """
    words_a = set(tokenize_latin(sent_a))
    words_b = set(tokenize_latin(sent_b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
