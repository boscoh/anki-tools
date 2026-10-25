"""Cantonese sentence ranking.

The ranking reuses the Han character scoring of
:mod:`anki_tools.languages.mandarin.characters` and the Cantonese frequency
sources of :mod:`anki_tools.languages.cantonese.frequency`. The Apple Daily
list is written Hong Kong Chinese; the ``spoken`` and ``blend`` sources add the
Cifu spoken counts. See ``specs/yue-sentence-selection.md``.
"""
from __future__ import annotations

from pathlib import Path

from anki_tools.languages.cantonese.frequency import (
    CIFU_ATTRIBUTION,
    YUE_FREQUENCY_ATTRIBUTION,
    YUE_FREQUENCY_MAX_WORDS,
    _build_frequency_data,
    _read_word_count_tsv,
    download_yue_frequency,
    load_yue_frequency,
    read_variants,
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
    path: Path | None = None,
    *,
    max_words: int = YUE_FREQUENCY_MAX_WORDS,
    source: str = "written",
    spoken_weight: float = 0.7,
) -> FrequencyData:
    """Load Cantonese word frequency data.

    The ``written`` source is the Apple Daily list. It holds 275,492 types from
    493,922,982 tokens of Hong Kong news text and covers Cantonese-only forms
    that the Mandarin ``zh`` list misses. The genre is written Chinese.

    The ``spoken`` and ``blend`` sources use the Cifu spoken counts. See
    :mod:`anki_tools.languages.cantonese.frequency`.

    The download runs when a cached file is absent.

    :param path: Apple Daily frequency file (``written`` source only).
    :param max_words: Number of most common words to keep.
    :param source: ``written``, ``spoken``, or ``blend``.
    :param spoken_weight: Weight of the spoken term in the ``blend`` source.
    :returns: Filled :class:`FrequencyData`.
    """
    if source != "written":
        data = to_frequency_data(
            load_yue_frequency(freq=source, spoken_weight=spoken_weight), max_words
        )
        print(f"Loaded {len(data.word_freq)} words ({source} frequency)")
        print(f"  {CIFU_ATTRIBUTION}; {YUE_FREQUENCY_ATTRIBUTION}")
        return data
    freq_path = Path(path) if path else download_yue_frequency()
    counts = _read_word_count_tsv(freq_path)
    if not counts:
        raise ValueError(f"No word counts found in {freq_path}")
    data = _build_frequency_data(counts, max_words, read_variants())
    print(f"Loaded {len(data.word_freq)} words from {YUE_FREQUENCY_ATTRIBUTION}")
    return data


def rank_sentences_yue(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
    *,
    freq: str = "blend",
    spoken_weight: float = 0.7,
) -> list[Sentence]:
    """Rank sentences for Cantonese by combined score.

    Converts simplified to traditional Chinese and generates jyutping romanization.
    Uses Chinese character-based complexity and similarity (like ZH)
    with Cantonese-specific weighting. The frequency term uses the Apple Daily
    Hong Kong Chinese list, because wordfreq has no `yue` list and its `zh`
    list measures Mandarin only. The Apple Daily list is written news text,
    not speech. The ``blend`` and ``spoken`` sources add Cifu spoken counts.

    :param sentences: Sentences to rank.
    :param weights: Optional dict with complexity, frequency, similarity_penalty.
    :param freq: Frequency term: ``blend``, ``spoken``, or ``written``.
    :param spoken_weight: Weight of the spoken term in the ``blend`` term.
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
        freq_loader=lambda: load_frequency_data_yue(
            source=freq, spoken_weight=spoken_weight
        ),
        repeated_filter=is_repeated_zh_sentence,
    )


__all__ = ["load_frequency_data_yue", "rank_sentences_yue"]
