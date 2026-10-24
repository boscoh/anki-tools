"""French sentence ranking.

The module holds the French frequency loader, the French grammar score, and
the French ranking driver. The shared helpers of
:mod:`anki_tools.ranking.latin` hold the Latin-script word metrics. The
shared helpers of :mod:`anki_tools.ranking.rank` hold the sentence data
class, the frequency tables, and the similarity engine.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from anki_tools.ranking.rank import (
    STRUCTURAL_COMPLEXITY_BLEND,
    WORD_WEIGHTS,
    FrequencyData,
    Sentence,
    load_wordfreq_data,
    rank_word_based,
)


def load_frequency_data_fr() -> FrequencyData:
    """Load French word frequency data from wordfreq.

    :returns: Filled :class:`FrequencyData`.
    """
    return load_wordfreq_data("fr", lower_keys=True)




@lru_cache(maxsize=1)
def _load_grammar_nlp() -> Any | None:
    """Load the French spaCy pipeline one time for the grammar score.

    The entity model is not part of the score, so it is left out. A missing
    spaCy package or model gives ``None``, and the caller uses the marker
    fallback.

    :returns: The loaded pipeline, or None when spaCy or the model is absent.
    """
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("fr_core_news_sm", exclude=["ner"])
    except OSError:
        return None


def _grammar_score_fr_spacy(sentence: str) -> float | None:
    """Give the spaCy grammar score of a French sentence.

    :param sentence: French sentence.
    :returns: Score from 0 to 100, or None when spaCy or the model is absent.
    """
    nlp = _load_grammar_nlp()
    if nlp is None:
        return None
    cleaned = re.sub(r"<[^>]+>", "", sentence)
    if not cleaned.strip():
        return 0.0
    doc = nlp(cleaned)
    if not doc:
        return 0.0
    max_depth = 0
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            depth += 1
            t = t.head
        max_depth = max(max_depth, depth)
    verb_count = sum(1 for t in doc if t.pos_ == "VERB" or t.pos_ == "AUX")
    subj_count = sum(1 for t in doc if t.dep_ == "nsubj" or t.dep_ == "csubj")
    score = min(100.0, max_depth * 8 + verb_count * 5 + subj_count * 3)
    return score


def grammar_score_fr(sentence: str) -> float:
    """Calculate the grammar score (0-100) of a French sentence.

    The score uses spaCy when the model is present. Otherwise it counts the
    subjunctive, the relative pronouns, the conditionals, and the commas.

    :param sentence: French sentence.
    :returns: Score from 0 to 100.
    """
    spacy_score = _grammar_score_fr_spacy(sentence)
    if spacy_score is not None:
        return min(100.0, spacy_score)

    cleaned = re.sub(r"<[^>]+>", "", sentence)
    lower = cleaned.lower()
    score = 0.0

    subjunctive_triggers = [
        "il faut que", "il faut qu'", "pour que", "bien que", "avant que",
        "quoique", "à moins que", "jusqu'à ce que", "pourvu que", "afin que",
        "bien qu'", "quoiqu'", "pour qu'", "avant qu'", "à moins qu'",
    ]
    for trigger in subjunctive_triggers:
        if trigger in lower:
            score += 12
            break

    relative_markers = [
        "lequel", "laquelle", "lesquels", "lesquelles", "dont",
        " auquel", " à laquelle", " auxquels", " auxquelles",
    ]
    for m in relative_markers:
        if m in lower:
            score += 10
            break

    if re.search(r"\bqui\b", lower):
        score += 3
    que_count = lower.count(" que ") + lower.count(" qu'")
    if que_count >= 2:
        score += 6
    elif que_count >= 1:
        score += 2
    if re.search(r"\bsi\b.*\b(était|avait|pouvait|faisait|allait|venait|voulait)", lower):
        score += 8
    comma_count = lower.count(",") + lower.count(";")
    if comma_count >= 2:
        score += min(comma_count * 3, 15)
    elif comma_count == 1:
        score += 3

    return min(score, 100.0)


def rank_sentences_fr(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
) -> list[Sentence]:
    """Rank French sentences by the combined score.

    The structural term blends the complexity score and the grammar score, so
    the two correlated measures count once.

    :param sentences: Sentences to rank.
    :param weights: Weights of structural, frequency, and similarity_penalty.
    :param structural_blend: Share of the complexity term in the structural
        score.
    :returns: The same list, sorted by final score, best first.
    """
    return rank_word_based(
        sentences,
        WORD_WEIGHTS if weights is None else weights,
        grammar=grammar_score_fr,
        freq_loader=load_frequency_data_fr,
        structural_blend=structural_blend,
    )


__all__ = [
    "grammar_score_fr",
    "load_frequency_data_fr",
    "rank_sentences_fr",
]
