"""Cantonese sentence ranking.

The ranking reuses the Han character scoring of
:mod:`anki_tools.languages.mandarin.characters` and the Cantonese frequency
sources of :mod:`anki_tools.languages.cantonese.frequency`. The Apple Daily
list is written Hong Kong Chinese; the ``spoken`` and ``blend`` sources add the
Cifu spoken counts. See ``specs/cantonese-frequency-list.md``.
"""
from __future__ import annotations


from anki_tools.languages.cantonese.frequency import (
    CIFU_ATTRIBUTION,
    YUE_FREQUENCY_ATTRIBUTION,
    YUE_FREQUENCY_MAX_WORDS,
    load_yue_frequency,
    to_frequency_data,
)
from anki_tools.languages.cantonese.jyutping import (
    get_jyutping,
    simplified_to_traditional,
)
from anki_tools.languages.mandarin.characters import (
    compute_similarity_penalties_zh,
    is_repeated_zh_sentence,
)
from anki_tools.languages.mandarin.rank import (
    complexity_score_zh,
    frequency_score_zh,
)
from anki_tools.ranking.rank import FrequencyData, Sentence, rank_character_based


def load_frequency_data_yue(
    *, max_words: int = YUE_FREQUENCY_MAX_WORDS
) -> FrequencyData:
    """Load the Cantonese word frequency data.

    The list blends the Cifu spoken counts with the written counts. See
    :mod:`anki_tools.languages.cantonese.frequency`. The download runs when a
    cached file is absent.

    :param max_words: Number of most common words to keep.
    :returns: Filled :class:`FrequencyData`.
    """
    data = to_frequency_data(load_yue_frequency(), max_words)
    print(f"Loaded {len(data.word_freq)} words (Cantonese blend)")
    print(f"  {CIFU_ATTRIBUTION}; {YUE_FREQUENCY_ATTRIBUTION}")
    return data


def rank_sentences_yue(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """Rank sentences for Cantonese by combined score.

    Converts simplified to traditional Chinese and generates jyutping romanization.
    Uses Chinese character-based complexity and similarity (like ZH)
    with Cantonese-specific weighting. The frequency term uses the Apple Daily
    list measures Mandarin only. The list blends the Cifu spoken counts with
    the Apple Daily written counts, because the writing is written news text.

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

    def preprocess_cantonese(sentences: list[Sentence]) -> None:
        print("Converting to traditional Chinese and generating jyutping...")
        for sent in sentences:
            sent.text = simplified_to_traditional(sent.text)
            sent.romanization = get_jyutping(sent.text)

    return rank_character_based(
        sentences,
        weights,
        complexity=complexity_score_zh,
        frequency=frequency_score_zh,
        penalties=compute_similarity_penalties_zh,
        preprocess=preprocess_cantonese,
        freq_loader=lambda: load_frequency_data_yue(),
        repeated_filter=is_repeated_zh_sentence,
    )


__all__ = ["load_frequency_data_yue", "rank_sentences_yue"]
