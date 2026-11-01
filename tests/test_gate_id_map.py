"""Tests that pin the pack, cull, and core rule ids to one semantic set."""

from anki_tools.ranking import cardfilter, coreflow, cullflow
from anki_tools.sentence_pack import pool

PACK_KEYS = (
    "english_link",
    "length_window",
    "digit_marks",
    "script",
    "repeat",
    "final_mark",
    "one_final_mark",
    "quote",
    "internal_mark",
    "predicate",
    "orphan",
    "english_words",
    "unique",
    "unknown",
    "sense",
)
CULL_KEYS = (
    "english_words",
    "length_window",
    "digit_marks",
    "script",
    "final_mark",
    "one_final_mark",
    "quote",
    "internal_mark",
    "propn",
    "unique",
)
CORE_RULE_KEYS = CULL_KEYS + ("C1", "C2", "C3", "C4", "C5")


def test_pack_keys_keep_the_pack_order() -> None:
    assert tuple(pool.FILTER_KEYS) == PACK_KEYS


def test_cull_keys_keep_the_cull_order() -> None:
    assert cullflow.FILTER_RULES == CULL_KEYS


def test_core_keys_keep_the_core_order() -> None:
    assert coreflow.FILTER_KEYS == CORE_RULE_KEYS


def test_the_guard_tables_cover_the_pack_keys_after_english_link() -> None:
    shape_keys = {key for key, _ in pool.SHAPE_GUARDS}
    fact_keys = {key for key, _ in pool.FACT_GUARDS}
    assert shape_keys | fact_keys == set(pool.FILTER_KEYS[1:])


def test_the_cull_mark_keys_are_the_pack_mark_keys() -> None:
    pack_mark_keys = {"final_mark", "one_final_mark", "quote", "internal_mark"}
    assert set(cardfilter.MARK_RULES) == pack_mark_keys
    assert cardfilter.LENGTH_RULES == ("length_window",)
