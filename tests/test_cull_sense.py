"""Test that the cull drops a lemma that fails the sense test of a card."""

from __future__ import annotations

import math

from anki_tools.ranking import corelist, coreflow, cullflow
from anki_tools.sentence_pack.english_sense import SenseRow

CONTENT_POS = corelist.CONTENT_POS
SENSES = {
    "car": SenseRow(
        senses=frozenset({"expensive", "dear"}),
        false_friends=frozenset({"face"}),
    )
}


def _card(index: int, english: str) -> coreflow.InputCard:
    return coreflow.InputCard(
        note_id=1000 + index,
        sentence=f"phrase {index}",
        english=english,
        audio="a.mp3",
        index=index,
    )


def _target(lemma: str, rank: int) -> corelist.TargetWord:
    return corelist.TargetWord(
        lemma=lemma,
        rank=rank,
        deck_count=2,
        content_share=1.0,
        weight=1 / math.log2(rank + 1),
    )


def test_a_false_friend_card_does_not_count_for_the_lemma() -> None:
    cards = [
        _card(0, "The shoes are expensive."),
        _card(1, "I face the wall."),
        _card(2, "Nothing is said."),
    ]
    tokens = [
        [("car", "NOUN"), ("f0", "NOUN")],
        [("car", "NOUN"), ("f1", "NOUN")],
        [("car", "NOUN"), ("f2", "NOUN")],
    ]

    result = cullflow.select_cards(
        cards,
        tokens,
        [_target("car", 5)],
        {"car": 5},
        max_cards=10,
        exempt_top=0,
        content_pos=CONTENT_POS,
        senses=SENSES,
    )

    assert result.card_targets == {0: ("car",)}
    assert result.uses["car"] == 1
    assert result.kept == [0, 1, 2]
    assert result.gain[0] > result.gain[1] == result.gain[2]


def test_a_lemma_without_a_row_counts() -> None:
    cards = [_card(0, "I face the wall.")]
    tokens = [[("car", "NOUN")]]

    result = cullflow.select_cards(
        cards,
        tokens,
        [_target("car", 5)],
        {"car": 5},
        max_cards=10,
        exempt_top=0,
        content_pos=CONTENT_POS,
    )

    assert result.card_targets == {0: ("car",)}
