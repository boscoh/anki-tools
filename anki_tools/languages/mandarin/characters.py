"""Han character helpers and similarity of the Mandarin package.

Cantonese shares the Han script, so it imports these helpers from this module.
The helpers hold script data only. The per-language word metrics stay in the
``rank.py`` module of each language package.
"""

from __future__ import annotations

from anki_tools.ranking.rank import (
    Sentence,
    compute_similarity_penalties,
    jaccard_similarity,
)


def is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character.

    :param char: Single character.
    :returns: True if in CJK Unified Ideographs range.
    """
    return "\u4e00" <= char <= "\u9fff"


def get_chinese_chars(text: str) -> list[str]:
    """Extract Chinese characters from text.

    :param text: Arbitrary text.
    :returns: List of Chinese characters in order.
    """
    return [c for c in text if is_chinese_char(c)]


def is_repeated_zh_sentence(text: str) -> bool:
    """Check if Chinese chars in text form a pure repetition pattern.

    Catches '是 是', '走吧走吧', '好了，好了', '谢谢你，谢谢你', etc.

    :param text: Sentence text.
    :returns: True if the Chinese character sequence is a repeated sub-sequence.
    """
    chars = get_chinese_chars(text)
    n = len(chars)
    if n < 2:
        return False
    for period in range(1, n // 2 + 1):
        if n % period == 0 and chars == chars[:period] * (n // period):
            return True
    return False


def char_similarity_zh(sent_a: str, sent_b: str) -> float:
    """Calculate Jaccard similarity between character sets.

    :param sent_a: First Chinese sentence.
    :param sent_b: Second Chinese sentence.
    :returns: Similarity between 0 (no overlap) and 1 (identical).
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))
    return jaccard_similarity(chars_a, chars_b)


def char_similarity_zh_same_len(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity of character sets, only for equal unique-character counts.

    The unique-character count avoids stutter anchors: '是 是' does not match
    '是的' only because both have 2 Chinese characters.

    :param sent_a: First Chinese sentence.
    :param sent_b: Second Chinese sentence.
    :returns: Similarity between 0 (no overlap or different sizes) and 1.
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))
    if len(chars_a) != len(chars_b) or not chars_a:
        return 0.0
    return jaccard_similarity(chars_a, chars_b)


def compute_similarity_penalties_zh(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None], list[str | None]]:
    """Compute similarity penalties (character-based, ZH and YUE).

    :param sentences: Sentences in rank order.
    :param top_n: Number of higher-ranked sentences to compare.
    :returns: Penalties, similar ranks, and similar texts.
    """
    return compute_similarity_penalties(sentences, char_similarity_zh, top_n)


__all__ = [
    "char_similarity_zh",
    "char_similarity_zh_same_len",
    "compute_similarity_penalties_zh",
    "get_chinese_chars",
    "is_chinese_char",
    "is_repeated_zh_sentence",
]
