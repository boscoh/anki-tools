"""Test for the ranking CSV that the similarity penalty and the structural score feed."""

from __future__ import annotations

import csv
from pathlib import Path

from anki_tools.ranking.rank import (
    Sentence,
    compute_similarity_penalties,
    write_ranking_csv,
)


def _jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b)


def test_the_ranking_csv_flags_a_near_duplicate_and_blends_the_structure(
    tmp_path: Path,
) -> None:
    sentences = [
        Sentence(note_id=11 + index, text=text, original_order=index)
        for index, text in enumerate(["a b c", "a b c d", "x y z"])
    ]
    penalties, ranks, texts = compute_similarity_penalties(sentences, _jaccard)
    for sentence, penalty, rank, text in zip(sentences, penalties, ranks, texts):
        sentence.similarity_penalty = penalty
        sentence.similar_to_rank = rank
        sentence.similar_to_text = text
        sentence.complexity_score = 50.0
        sentence.grammar_score = 10.0
        sentence.romanization = "r"
        sentence.kept = True
    output = tmp_path / "rank.csv"

    write_ranking_csv(sentences, str(output))

    with output.open(encoding="utf-8", newline="") as handle:
        rows = [
            (row["rank"], row["sentence"], row["structural"], row["similarity"], row["similar_to"])
            for row in csv.DictReader(handle)
        ]
    assert rows == [
        ("1", "a b c", "34.0", "0.0", ""),
        ("2", "a b c d", "34.0", "15.0", "a b c"),
        ("3", "x y z", "34.0", "0.0", ""),
    ]
