"""Tests for the ordered guard tables and the language guard hooks."""

from types import SimpleNamespace
from typing import Any

from anki_tools.sentence_pack import audit, pool
from anki_tools.sentence_pack.guards import (
    check_sentence_shape,
    compound_spans,
    form_compounds,
    inside_spans,
    occurrences,
)
from anki_tools.languages.cantonese.profile import CANTONESE
from anki_tools.languages.french.profile import FRENCH
from anki_tools.languages.greek.profile import GREEK
from anki_tools.languages.mandarin.profile import MANDARIN


def test_guard_table_matches_the_filter_order() -> None:
    keys = [key for key, _ in pool.SHAPE_GUARDS]
    keys += [key for key, _ in pool.FACT_GUARDS]
    assert keys == pool.FILTER_KEYS[1:]


def test_scaffold_relaxes_only_the_count_checks_of_marks() -> None:
    two_final = "我要喝水。。"
    assert pool.shape_failure(two_final, MANDARIN, 4, 7) == "marks"
    assert pool.shape_failure(two_final, MANDARIN, 4, 7, scaffold=True) is None
    quoted = "他说“我要喝水”。"
    assert pool.shape_failure(quoted, MANDARIN, 4, 9, scaffold=True) == "marks"


def test_record_pass_counts_every_guard_before_the_failure() -> None:
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, "marks")
    assert set(pool.FILTER_KEYS[:5]) == {key for key, value in counts.items() if value}
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, None)
    assert all(value == 1 for value in counts.values())


def test_digit_marks_is_one_guard_for_every_language() -> None:
    for profile, sentence in (
        (MANDARIN, "我想喝水2杯。"),
        (GREEK, "Η γάτα έχει 3 μήνες."),
        (FRENCH, "Le chat [noir] mange vite."),
        (FRENCH, "Le chat mange…"),
    ):
        assert (
            pool.shape_failure(sentence, profile, 1, 20, keys=frozenset({"digit_marks"}))
            == "digit_marks"
        )


def test_script_tests_only_the_alphabet() -> None:
    assert MANDARIN.script_ok("我想喝水2杯。")
    assert not MANDARIN.script_ok("我想喝tea。")
    assert FRENCH.script_ok("Il a 42 ans.")


def test_shape_failure_applies_only_the_requested_keys() -> None:
    sentence = "我要喝水"
    assert check_sentence_shape(sentence, MANDARIN, 4, 7) == "marks"
    assert pool.shape_failure(sentence, MANDARIN, 4, 7, scaffold=True) == "marks"
    assert (
        pool.shape_failure(sentence, MANDARIN, 4, 7, keys=frozenset({"script"}))
        is None
    )


def test_fact_failure_can_skip_the_sense_guard() -> None:
    measured = pool.FactInput(
        duplicate=False,
        unknown_words=0,
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


def test_compounds_come_from_the_sense_file_of_the_chinese_packs() -> None:
    assert form_compounds(MANDARIN.sense_path, "\u6765") == ("\u51fa\u6765", "\u8d77\u6765")


def test_gate_no_false_friend_reads_the_compounds_of_the_record() -> None:
    bad = FakeInputs([{"sentence": "聽日係我生日。", "target": "聽"}], [])
    passed, _ = audit.gate_no_false_friend(bad, CANTONESE)  # type: ignore[arg-type]
    assert not passed
    good = FakeInputs([{"sentence": "我聽音樂。", "target": "聽"}], [])
    passed, _ = audit.gate_no_false_friend(good, CANTONESE)  # type: ignore[arg-type]
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
    passed, _ = audit.gate_no_sense_mismatch(bad, CANTONESE)  # type: ignore[arg-type]
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
    passed, _ = audit.gate_no_sense_mismatch(good, CANTONESE)  # type: ignore[arg-type]
    assert passed
