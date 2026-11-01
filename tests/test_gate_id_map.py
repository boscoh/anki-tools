"""Tests that pin the pack, cull, and core rule ids to one semantic set."""

import pytest

from anki_tools.ranking import cardfilter, coreflow, cullflow
from anki_tools.sentence_pack import pool

SHAPE_KEYS = ("length_window", "digit_marks", "script", "repeat", "marks", "unique")


@pytest.mark.parametrize(
    "keys, expected",
    [
        pytest.param(tuple(pool.FILTER_KEYS), (*SHAPE_KEYS, "sense"), id="pack"),
        pytest.param(cullflow.FILTER_RULES, SHAPE_KEYS, id="cull"),
        pytest.param(coreflow.FILTER_KEYS, SHAPE_KEYS, id="core"),
    ],
)
def test_the_filter_keys_keep_their_order(keys: tuple[str, ...], expected: tuple[str, ...]) -> None:
    assert tuple(keys) == expected


def test_every_card_filter_rule_is_a_pack_cull_and_core_key() -> None:
    rules = set(cardfilter.LENGTH_RULES + cardfilter.TEXT_RULES + cardfilter.MARK_RULES)
    assert rules <= set(pool.FILTER_KEYS) & set(cullflow.FILTER_RULES) & set(coreflow.FILTER_KEYS)
