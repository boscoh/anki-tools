"""Mandarin sentence ranking.

The module holds the Mandarin word frequency table, the Mandarin complexity and
frequency scores, and the Mandarin ranking driver. The shared helpers of
:mod:`anki_tools.ranking.rank` hold the sentence data class, the frequency
tables, and the similarity engine. :mod:`anki_tools.languages.mandarin.
characters` holds the Han character helpers.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import jieba

from anki_tools.languages.mandarin.characters import (
    compute_similarity_penalties_zh,
    get_chinese_chars,
    is_chinese_char,
    is_repeated_zh_sentence,
)
from anki_tools.ranking.rank import (
    FrequencyData,
    Sentence,
    load_wordfreq_data,
    rank_character_based,
)


def load_frequency_data_zh() -> FrequencyData:
    """Load Chinese word frequency data from wordfreq.

    :returns: Filled :class:`FrequencyData`.
    """
    return load_wordfreq_data("zh", char_filter=is_chinese_char)


def complexity_score_zh(sentence: str) -> float:
    """Calculate complexity score (0-100) based on structural features.

    Metrics (see SPEC_RANK_SENTENCES.md):
        - Character count, unique characters, word count, average word length.

    :param sentence: Chinese sentence.
    :returns: Score from 0 to 100 (higher = more complex).
    """
    chars = get_chinese_chars(sentence)
    if not chars:
        return 0.0

    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    char_count = len(chars)
    unique_chars = len(set(chars))
    word_count = len(chinese_words)
    avg_word_len = sum(len(w) for w in chinese_words) / max(len(chinese_words), 1)

    char_norm = min(char_count / 30, 1.0)
    unique_norm = min(unique_chars / 20, 1.0)
    word_norm = min(word_count / 15, 1.0)
    avg_len_norm = min(avg_word_len / 4, 1.0)

    score = (
        char_norm * 0.25 + unique_norm * 0.25 + word_norm * 0.30 + avg_len_norm * 0.20
    )

    return score * 100


def get_word_rank_zh(word: str, freq_data: FrequencyData, max_rank: int = 50000) -> int:
    """Get frequency rank for a word, falling back to character average if not found.

    :param word: Word to look up.
    :param freq_data: Loaded frequency data.
    :param max_rank: Default rank when not found.
    :returns: Frequency rank (lower = more common).
    """
    if word in freq_data.word_freq:
        return freq_data.word_freq[word]

    chars = get_chinese_chars(word)
    if not chars:
        return max_rank

    char_ranks = [freq_data.char_freq.get(c, max_rank) for c in chars]
    return int(sum(char_ranks) / len(char_ranks))


def frequency_score_zh(sentence: str, freq_data: FrequencyData) -> float:
    """Calculate frequency score (0-100) based on word commonality.

    Higher score = more common vocabulary = easier to learn first.

    :param sentence: Chinese sentence.
    :param freq_data: Loaded frequency data.
    :returns: Score from 0 to 100.
    """
    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    if not chinese_words:
        return 0.0

    if not freq_data.word_freq:
        return 50.0

    max_rank = 50000
    ranks = [get_word_rank_zh(w, freq_data, max_rank) for w in chinese_words]

    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)

    top_1000 = sum(1 for r in ranks if r <= 1000) / len(ranks)
    top_5000 = sum(1 for r in ranks if r <= 5000) / len(ranks)

    if freq_data.vocab_levels:
        hsk_count = 0
        for w in chinese_words:
            if freq_data.vocab_levels.get(w, 7) <= 4:
                hsk_count += 1
            else:
                chars = get_chinese_chars(w)
                if chars and all(freq_data.vocab_levels.get(c, 7) <= 4 for c in chars):
                    hsk_count += 1
        hsk_coverage = hsk_count / len(chinese_words)
    else:
        hsk_coverage = 0.5

    score = avg_score * 0.40 + top_1000 * 0.25 + top_5000 * 0.15 + hsk_coverage * 0.20

    return score * 100


def rank_sentences_zh(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """Rank sentences by combined score (Chinese pipeline).

    Default weights: complexity 0.3, frequency 0.5, similarity_penalty 0.2.
    Complexity is inverted (lower complexity = higher score for learning).

    :param sentences: Sentences to rank.
    :param weights: Optional dict with complexity, frequency, similarity_penalty.
    :returns: Same list sorted by final_score (best first).
    """
    if weights is None:
        weights = {
            "complexity": 0.3,
            "frequency": 0.5,
            "similarity_penalty": 0.2,
        }

    return rank_character_based(
        sentences,
        weights,
        complexity=complexity_score_zh,
        frequency=frequency_score_zh,
        penalties=compute_similarity_penalties_zh,
        freq_loader=load_frequency_data_zh,
        repeated_filter=is_repeated_zh_sentence,
    )


__all__ = [
    "complexity_score_zh",
    "frequency_score_zh",
    "get_word_rank_zh",
    "load_frequency_data_zh",
    "rank_sentences_zh",
]
