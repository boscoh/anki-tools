"""Spanish sentence ranking.

Spanish is a word-based language, so the module reuses the shared Latin-script
word metrics of :mod:`anki_tools.ranking.latin` and the similarity engine of
:mod:`anki_tools.ranking.rank`. Only the grammar heuristics and the ranking
driver are Spanish-specific.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import re

from anki_tools.ranking.rank import (
    STRUCTURAL_COMPLEXITY_BLEND,
    WORD_WEIGHTS,
    FrequencyData,
    Sentence,
    load_wordfreq_data,
    rank_word_based,
)

#: The word-set Jaccard value at which two Spanish sentences are duplicates.
SIMILARITY_THRESHOLD = 0.5

def load_frequency_data_es() -> FrequencyData:
    """Load Spanish word frequency data from wordfreq.

    :returns: Filled :class:`FrequencyData`.
    """
    return load_wordfreq_data("es", lower_keys=True)


def grammar_score_es(sentence: str) -> float:
    """Calculate the grammar score (0-100) of a Spanish sentence.

    The score counts the subjunctive, the relative clauses, the conditionals,
    and the commas.

    :param sentence: Spanish sentence.
    :returns: Score from 0 to 100.
    """
    cleaned = re.sub(r"<[^>]+>", "", sentence)
    lower = cleaned.lower()
    score = 0.0

    subjunctive_triggers = [
        "es necesario que", "es importante que", "es posible que",
        "para que", "aunque", "antes de que", "sin que", "con tal de que",
        "ojalá", "tal vez", "quizás", "puede que",
    ]
    for trigger in subjunctive_triggers:
        if trigger in lower:
            score += 12
            break

    relative_markers = ["el cual", "la cual", "los cuales", "las cuales", "cuyo", "cuya"]
    for m in relative_markers:
        if m in lower:
            score += 10
            break

    if re.search(r"\bque\b", lower):
        score += 3

    if re.search(r"\bsi\b.*\b(fuera|hubiera|sería|haría|tendría)", lower):
        score += 8

    comma_count = lower.count(",") + lower.count(";")
    if comma_count >= 2:
        score += min(comma_count * 3, 15)
    elif comma_count == 1:
        score += 3

    return min(score, 100.0)


def rank_sentences_es(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
) -> list[Sentence]:
    """Rank sentences for Spanish by combined score.

    Uses word-based complexity, frequency, and grammar scoring.
    Combines complexity and grammar into structural score.

    :param sentences: Sentences to rank (text = main sentence).
    :param weights: Optional dict with structural, frequency, similarity_penalty.
    :param structural_blend: Fraction of structural from complexity (rest from grammar).
    :returns: Same list sorted by final_score (best first).
    """
    return rank_word_based(
        sentences,
        WORD_WEIGHTS if weights is None else weights,
        grammar=grammar_score_es,
        freq_loader=load_frequency_data_es,
        structural_blend=structural_blend,
    )


__all__ = [
    "grammar_score_es",
    "load_frequency_data_es",
    "rank_sentences_es",
]
