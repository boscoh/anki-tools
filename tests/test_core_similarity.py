"""Test that the core selection keeps one of two cards with close lemma sets."""

from __future__ import annotations

import pytest

from anki_tools.ranking.core import CoreCandidate, CoreOptions, select


@pytest.mark.parametrize(
    ("lemmas", "kept"),
    [
        pytest.param(({"a", "b", "c"}, {"a", "b", "c"}), [0], id="the same lemmas"),
        pytest.param(({"a", "b", "c"}, {"a", "b", "c", "d"}), [0], id="three of four"),
        pytest.param(({"a", "b"}, {"a", "c"}), [0, 1], id="one of three"),
    ],
)
def test_cards_with_close_content_lemmas_are_not_both_kept(
    lemmas: tuple[set[str], set[str]], kept: list[int]
) -> None:
    candidates = [
        CoreCandidate(index, ("F:a",), (), 4, frozenset(words))
        for index, words in enumerate(lemmas)
    ]

    result = select(
        candidates,
        positions={"F:a": 1},
        quotas={"F:a": 2},
        ranks={},
        options=CoreOptions(max_cards=10, completion_cards=0),
    )

    assert result.kept == kept
