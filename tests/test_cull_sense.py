"""Test that the cull drops a lemma that fails the sense test of a card."""

from __future__ import annotations

from anki_tools.sentence_pack.english_sense import SenseRow
from tests.cull_fixture import card, select, target

SENSES = {
    "car": SenseRow(
        senses=frozenset({"expensive", "dear"}),
        false_friends=frozenset({"face"}),
    )
}
TARGETS = [target("car", 5, deck_count=2)]
RANKS = {"car": 5}


def test_a_false_friend_card_does_not_count_for_the_lemma() -> None:
    cards = [
        card("phrase 0", 0, "The shoes are expensive."),
        card("phrase 1", 1, "I face the wall."),
        card("phrase 2", 2, "Nothing is said."),
    ]
    tokens = [[("car", "NOUN"), (f"f{index}", "NOUN")] for index in range(3)]

    result = select(cards, tokens, TARGETS, RANKS, senses=SENSES)

    assert result.card_targets == {0: ("car",)}
    assert result.uses["car"] == 1
    assert result.kept == [0]
    assert result.reasons == {1: "no_core", 2: "no_core"}


def test_a_lemma_without_a_row_counts() -> None:
    cards = [card("phrase 0", 0, "I face the wall.")]

    result = select(cards, [[("car", "NOUN")]], TARGETS, RANKS)

    assert result.card_targets == {0: ("car",)}


def test_similarity_uses_the_full_lemma_sets() -> None:
    cards = [
        card("phrase 0", 0, "The shoes are expensive."),
        card("phrase 1", 1, "I face the wall."),
    ]
    tokens = [
        [(lemma, "NOUN") for lemma in ("car", "a", "b", "c", "d", "e")],
        [(lemma, "NOUN") for lemma in ("car", "a", "b", "c", "d", "f")],
    ]

    result = select(
        cards, tokens, [*TARGETS, target("a", 6)], RANKS, senses=SENSES
    )

    assert result.kept == [0]
    assert result.reasons == {1: "similar"}
