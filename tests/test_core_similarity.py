"""Test that the core build and gate K6 use the same lemma set."""

from __future__ import annotations

from anki_tools.ranking.core import CoreCandidate, CoreOptions, select
from anki_tools.ranking.corelist import CONTENT_POS, lemma_sets


def test_cards_with_the_same_content_lemmas_are_not_both_kept() -> None:
    tokens = [
        [("a", "NOUN"), ("b", "VERB"), ("c", "NOUN"), ("the", "DET")],
        [("a", "NOUN"), ("b", "VERB"), ("c", "NOUN"), ("of", "ADP")],
    ]
    full_sets = lemma_sets(tokens, CONTENT_POS)
    candidates = [
        CoreCandidate(0, ("F:a",), ("b", "c"), 4, frozenset(full_sets[0])),
        CoreCandidate(1, ("F:a",), (), 4, frozenset(full_sets[1])),
    ]

    result = select(
        candidates,
        positions={"F:a": 1},
        quotas={"F:a": 2},
        ranks={},
        options=CoreOptions(max_cards=10, completion_cards=0),
    )

    assert len(result.kept) == 1
