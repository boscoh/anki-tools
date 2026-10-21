"""Tests for the ordered guard tables and the language guard hooks."""

from types import SimpleNamespace
from typing import Any

from anki_tools.sentence_pack import audit, pool
from anki_tools.sentence_pack.guards import (
    check_sentence_shape,
    compound_spans,
    inside_spans,
    occurrences,
)
from anki_tools.sentence_pack.languages.cantonese import CANTONESE
from anki_tools.languages.french.profile import FRENCH
from anki_tools.sentence_pack.languages.greek import GREEK
from anki_tools.sentence_pack.languages.mandarin import MANDARIN


def test_guard_table_matches_the_filter_order() -> None:
    keys = [key for key, _ in pool.SHAPE_GUARDS]
    keys += [key for key, _ in pool.FACT_GUARDS]
    assert keys == pool.FILTER_KEYS[1:]


def test_scaffold_keys_skip_the_relaxed_filters() -> None:
    assert pool.SCAFFOLD_SHAPE_KEYS == frozenset({"F2", "F3", "F4", "F5", "F7"})
    assert pool.SCAFFOLD_SHAPE_FAILURES == frozenset({"F6", "F8"})


def test_record_pass_counts_every_guard_before_the_failure() -> None:
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, "F5")
    assert set(pool.FILTER_KEYS[:4]) == {key for key, value in counts.items() if value}
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, None)
    assert all(value == 1 for value in counts.values())


def test_shape_failure_applies_only_the_requested_keys() -> None:
    sentence = "我要喝水"
    assert check_sentence_shape(sentence, MANDARIN, 4, 7) == "F5"
    assert (
        pool.shape_failure(
            sentence, MANDARIN, 4, 7, keys=pool.SCAFFOLD_SHAPE_KEYS
        )
        == "F5"
    )
    assert (
        pool.shape_failure(sentence, MANDARIN, 4, 7, keys=frozenset({"F3"})) is None
    )


def test_fact_failure_can_skip_the_sense_guard() -> None:
    measured = pool.FactInput(
        predicate=True,
        orphan=False,
        english_words=5,
        english_bounds=(2, 12),
        duplicate=False,
        unknown_words=0,
        max_unknown=2,
        hits=("火",),
        sense_hits=(),
    )
    assert pool.fact_failure(measured) == "sense"
    assert pool.fact_failure(measured, skip=pool.SCAFFOLD_SKIP_GUARDS) is None


class FakeInputs:
    def __init__(
        self, rows: list[dict[str, str]], table: list[Any]
    ) -> None:
        self._rows = rows
        self._table = table

    def rows(self) -> list[dict[str, str]]:
        return self._rows

    def table(self) -> list[Any]:
        return self._table


def test_occurrences_finds_every_start() -> None:
    assert occurrences("aXaXa", "X") == [1, 3]
    assert occurrences("abc", "") == []
    assert occurrences("abc", "z") == []


def test_compound_spans_and_inside_spans() -> None:
    spans = compound_spans("聽日聽日", ("聽日",))
    assert spans == [(0, 2), (2, 4)]
    assert inside_spans(0, spans)
    assert not inside_spans(4, spans)


def test_false_friend_hook_is_set_for_the_chinese_packs() -> None:
    assert MANDARIN.false_friend_spans is not None
    assert CANTONESE.false_friend_spans is not None
    assert FRENCH.false_friend_spans is None
    assert GREEK.false_friend_spans is None


def test_gate_no_false_friend_uses_the_profile_hook() -> None:
    bad = FakeInputs([{"sentence": "聽日係我生日。", "target": "聽"}], [])
    passed, _ = audit.gate_no_false_friend(bad, CANTONESE)
    assert not passed
    good = FakeInputs([{"sentence": "我聽音樂。", "target": "聽"}], [])
    passed, _ = audit.gate_no_false_friend(good, CANTONESE)
    assert passed


def test_gate_no_sense_mismatch_uses_the_profile_filter() -> None:
    table = [SimpleNamespace(cantonese="燒", english="to burn")]
    bad = FakeInputs(
        [
            {
                "sentence": "我發燒。",
                "target": "燒",
                "english": "She has a temperature.",
            }
        ],
        table,
    )
    passed, _ = audit.gate_no_sense_mismatch(bad, CANTONESE)
    assert not passed
    good = FakeInputs(
        [
            {
                "sentence": "我燒水。",
                "target": "燒",
                "english": "My hand was burned.",
            }
        ],
        table,
    )
    passed, _ = audit.gate_no_sense_mismatch(good, CANTONESE)
    assert passed
