"""Shared text helpers and word metrics of the Latin-script languages.

The tokenizer serves the languages that use a Latin script: French and
Spanish. A language package uses it for its word set, its complexity score,
its frequency score, and its dedupe key. The helpers hold no language data.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anki_tools.ranking.rank import FrequencyData

LETTER_PATTERN = re.compile(r"[a-zA-Z\u00c0-\u024f]+")

#: Rank of a word that the frequency list does not hold.
DEFAULT_WORD_RANK = 100_000


def complexity_score_latin(sentence: str) -> float:
    """Give the complexity score of a Latin-script sentence, 0 to 100.

    The score blends the word count, the unique word count, the mean word
    length, and the character count.

    :param sentence: Sentence or phrase.
    :returns: Score from 0 to 100, higher for a more complex sentence.
    """
    words = tokenize_latin(sentence)
    if not words:
        return 0.0

    word_count = len(words)
    unique_words = len(set(words))
    avg_word_len = sum(len(w) for w in words) / len(words)
    char_count = sum(len(w) for w in words)

    word_norm = min(word_count / 25, 1.0)
    unique_norm = min(unique_words / 20, 1.0)
    avg_len_norm = min(avg_word_len / 8, 1.0)
    char_norm = min(char_count / 150, 1.0)

    score = (
        word_norm * 0.25
        + unique_norm * 0.25
        + avg_len_norm * 0.25
        + char_norm * 0.25
    )
    return score * 100


def get_word_rank(
    word: str, freq_data: FrequencyData, max_rank: int = DEFAULT_WORD_RANK
) -> int:
    """Give the frequency rank of one Latin-script word.

    :param word: Word to look up, in any case.
    :param freq_data: Loaded frequency data.
    :param max_rank: Rank of a word that the list does not hold.
    :returns: Frequency rank, lower for a more common word.
    """
    return freq_data.word_freq.get(word.lower(), max_rank)


def frequency_score_latin(sentence: str, freq_data: FrequencyData) -> float:
    """Give the frequency score of a Latin-script sentence, 0 to 100.

    :param sentence: Sentence or phrase.
    :param freq_data: Loaded frequency data.
    :returns: Score from 0 to 100, higher for more common words.
    """
    words = tokenize_latin(sentence)
    if not words:
        return 0.0
    if not freq_data.word_freq:
        return 50.0

    ranks = [get_word_rank(w, freq_data) for w in words]
    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / DEFAULT_WORD_RANK, 1.0)
    top_5k = sum(1 for r in ranks if r <= 5000) / len(ranks)
    return (avg_score * 0.6 + top_5k * 0.4) * 100

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
