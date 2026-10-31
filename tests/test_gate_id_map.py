"""Tests that pin the pack and ranking rule keys to the ADR 0027 map."""

from anki_tools.ranking import cullflow
from anki_tools.sentence_pack import pool

# ADR 0027: every pack key and every rank key has one row in the map.
SAME_ID = ("F1", "F2", "F3")
SHIFTED = {"F5": "F4", "F6": "F5", "F7": "F6", "F8": "F7"}
RENUMBERED = {"F12": "F9"}
RANK_ONLY = ("F8",)
PACK_ONLY = ("F4", "F9", "F10", "F11", "unknown", "sense")


def test_mark_rules_is_the_shift_of_the_map() -> None:
    assert cullflow.MARK_RULES == SHIFTED


def test_every_pack_key_has_a_map_row() -> None:
    mapped = set(SAME_ID) | set(SHIFTED) | set(RENUMBERED) | set(PACK_ONLY)
    assert mapped == set(pool.FILTER_KEYS)


def test_every_rank_key_has_a_map_row() -> None:
    mapped = (
        set(SAME_ID)
        | set(SHIFTED.values())
        | set(RENUMBERED.values())
        | set(RANK_ONLY)
    )
    assert mapped == set(cullflow.FILTER_RULES)
