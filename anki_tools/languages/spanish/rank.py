"""Spanish sentence ranking.

Spanish is a word-based language like French, so the module reuses the French
word metrics and the shared Latin tokenizer. Only the grammar heuristics and
the ranking driver are Spanish-specific.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import re

from anki_tools.languages.french.rank import (
    complexity_score_fr,
    get_word_rank_fr,
    word_similarity_fr,
)
from anki_tools.ranking.latin import tokenize_latin
from anki_tools.ranking.rank import (
    STRUCTURAL_COMPLEXITY_BLEND,
    FrequencyData,
    Sentence,
    _compute_similarity_penalties,
    _load_frequency_data_wordfreq,
    _structural_score,
)


def load_frequency_data_es() -> FrequencyData:
    """Load Spanish word frequency data from wordfreq.

    :returns: Filled :class:`FrequencyData`.
    """
    return _load_frequency_data_wordfreq("es", lower_keys=True)


def complexity_score_es(sentence: str) -> float:
    """Calculate the complexity score (0-100) of a Spanish sentence.

    :param sentence: Spanish sentence.
    :returns: Score from 0 to 100, higher for a more complex sentence.
    """
    return complexity_score_fr(sentence)


def frequency_score_es(sentence: str, freq_data: FrequencyData) -> float:
    """Calculate the frequency score (0-100) of a Spanish sentence.

    :param sentence: Spanish sentence.
    :param freq_data: Loaded frequency data.
    :returns: Score from 0 to 100, higher for more common words.
    """
    words = tokenize_latin(sentence)
    if not words:
        return 0.0
    if not freq_data.word_freq:
        return 50.0

    max_rank = 100000
    ranks = [get_word_rank_fr(w, freq_data, max_rank) for w in words]
    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)
    top_5k = sum(1 for r in ranks if r <= 5000) / len(ranks)
    return (avg_score * 0.6 + top_5k * 0.4) * 100


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


def word_similarity_es(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity between word sets (for Spanish).

    Reuses word tokenization from French (both are Romance languages).

    :param sent_a: First sentence.
    :param sent_b: Second sentence.
    :returns: Similarity between 0 and 1.
    """
    return word_similarity_fr(sent_a, sent_b)


def compute_similarity_penalties_es(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None], list[str | None]]:
    """Compute similarity penalties (word-based, ES)."""
    return _compute_similarity_penalties(sentences, word_similarity_es, top_n)


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
    if weights is None:
        weights = {
            "structural": 0.35,
            "frequency": 0.45,
            "similarity_penalty": 0.2,
        }

    freq_data = load_frequency_data_es()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score_es(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score_es(sent.text, freq_data)

    print("Calculating grammar scores...")
    for sent in sentences:
        sent.grammar_score = grammar_score_es(sent.text)

    for sent in sentences:
        structural = _structural_score(
            sent.complexity_score, sent.grammar_score, structural_blend
        )
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_rank_indices, similar_to_texts = (
        compute_similarity_penalties_es(sentences)
    )
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalties, similar_to_rank_indices, similar_to_texts
    ):
        sent.similarity_penalty = penalty
        sent.similar_to_rank = similar_to_rank
        sent.similar_to_text = sim_text

    for sent in sentences:
        structural = _structural_score(
            sent.complexity_score, sent.grammar_score, structural_blend
        )
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)
    return sentences


__all__ = [
    "complexity_score_es",
    "compute_similarity_penalties_es",
    "frequency_score_es",
    "grammar_score_es",
    "load_frequency_data_es",
    "rank_sentences_es",
    "word_similarity_es",
]
