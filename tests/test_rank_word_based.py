"""Tests for the shared word-based ranking driver."""

from __future__ import annotations

import pytest

from anki_tools.ranking.latin import complexity_score_latin, frequency_score_latin
from anki_tools.ranking.rank import (
    SIMILARITY_PENALTY_SCALE,
    WORD_WEIGHTS,
    FrequencyData,
    Sentence,
    rank_word_based,
    structural_score,
)

FREQUENCY = FrequencyData(
    word_freq={
        "le": 1,
        "chat": 2,
        "mange": 3,
        "chien": 4,
        "dort": 5,
        "vite": 6,
        "très": 7,
    }
)


def _sentences() -> list[Sentence]:
    texts = ["le chat mange", "le chien dort vite", "le chat mange très vite"]
    return [
        Sentence(note_id=index, text=text, original_order=index)
        for index, text in enumerate(texts)
    ]


def _rank(sentences: list[Sentence]) -> list[Sentence]:
    return rank_word_based(
        sentences,
        WORD_WEIGHTS,
        grammar=lambda text: 40.0,
        freq_loader=lambda: FREQUENCY,
    )


def test_scores_are_filled() -> None:
    sentences = _rank(_sentences())

    for sent in sentences:
        assert sent.complexity_score == pytest.approx(complexity_score_latin(sent.text))
        assert sent.frequency_score == pytest.approx(
            frequency_score_latin(sent.text, FREQUENCY)
        )
        assert sent.grammar_score == 40.0


def test_final_score_follows_the_weights() -> None:
    sentences = _rank(_sentences())

    for sent in sentences:
        structural = structural_score(sent.complexity_score, sent.grammar_score)
        expected = (
            sent.frequency_score * WORD_WEIGHTS["frequency"]
            + (100 - structural) * WORD_WEIGHTS["structural"]
            - sent.similarity_penalty * WORD_WEIGHTS["similarity_penalty"]
        )
        assert sent.final_score == pytest.approx(expected)


def test_sentences_are_sorted_best_first() -> None:
    sentences = _rank(_sentences())

    scores = [sent.final_score for sent in sentences]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_gets_a_similarity_penalty() -> None:
    texts = ["le chat mange", "le chat mange", "chien dort vite"]
    sentences = [
        Sentence(note_id=index, text=text, original_order=index)
        for index, text in enumerate(texts)
    ]

    ranked = _rank(sentences)

    penalized = [sent for sent in ranked if sent.similarity_penalty > 0]
    assert len(penalized) == 1
    assert penalized[0].similarity_penalty == pytest.approx(SIMILARITY_PENALTY_SCALE)
    assert penalized[0].similar_to_rank in (1, 2)
    assert penalized[0].similar_to_text == "le chat mange"


def test_empty_list() -> None:
    assert _rank([]) == []
