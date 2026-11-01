"""Tests for the ordered guard tables and the language guard hooks."""

import pytest

from anki_tools.sentence_pack import pool
from anki_tools.sentence_pack.guards import (
    check_sentence_shape,
    compound_spans,
    form_compounds,
    inside_spans,
    occurrences,
)
from anki_tools.languages.cantonese.profile import form_offsets
from anki_tools.languages.french.profile import hits as french_hits
from anki_tools.languages.mandarin.profile import MANDARIN, count_form_occurrences
from anki_tools.languages.spanish.profile import hits as spanish_hits
from tests.shape_fixture import PROFILES, shape_rules


def test_guard_table_matches_the_filter_order() -> None:
    keys = [key for key, _ in pool.SHAPE_GUARDS]
    keys += [key for key, _ in pool.FACT_GUARDS]
    assert keys == pool.FILTER_KEYS


def test_scaffold_relaxes_only_the_count_checks_of_marks() -> None:
    two_final = "我要喝水。。"
    assert pool.shape_failure(two_final, MANDARIN, 4, 7) == "marks"
    assert pool.shape_failure(two_final, MANDARIN, 4, 7, scaffold=True) is None
    quoted = "他说“我要喝水”。"
    assert pool.shape_failure(quoted, MANDARIN, 4, 9, scaffold=True) == "marks"


def test_record_pass_counts_every_guard_before_the_failure() -> None:
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, "marks")
    assert set(pool.FILTER_KEYS[:4]) == {key for key, value in counts.items() if value}
    counts = dict.fromkeys(pool.FILTER_KEYS, 0)
    pool.record_pass(counts, None)
    assert all(value == 1 for value in counts.values())


@pytest.mark.parametrize("language, sentence, expected", shape_rules("french", "greek", "mandarin"))
def test_a_sentence_gets_the_first_failed_shape_guard(
    language: str, sentence: str, expected: str
) -> None:
    profile = PROFILES[language]

    failure = pool.shape_failure(sentence, profile, profile.min_units, profile.max_units)

    assert (failure or "") == expected


def test_shape_failure_applies_only_the_requested_keys() -> None:
    sentence = "我要喝水"
    assert check_sentence_shape(sentence, MANDARIN, 4, 7) == "marks"
    assert pool.shape_failure(sentence, MANDARIN, 4, 7, scaffold=True) == "marks"
    assert pool.shape_failure(sentence, MANDARIN, 4, 7, keys=frozenset({"script"})) is None
    assert (
        pool.shape_failure("我想喝水2杯。", MANDARIN, 4, 7, keys=frozenset({"script"})) is None
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


def test_compound_rows_block_insect_and_can_hear() -> None:
    assert form_offsets("\u6606\u87f2\u597d\u7d30", "\u87f2") == []
    assert form_offsets("\u87f2\u597d\u7d30", "\u87f2") == [0]
    assert count_form_occurrences("\u6211\u542c\u5f97\u5230\u3002", "\u5f97\u5230") == 0
    assert count_form_occurrences("\u6211\u5f97\u5230\u4e86\u3002", "\u5f97\u5230") == 1


@pytest.mark.parametrize(
    ("hits", "sentence", "form", "expected"),
    [
        (french_hits, "Quelque chose ne va pas.", "quelques", {}),
        (french_hits, "Je lis quelque livre.", "quelques", {"quelques": 1}),
        (french_hits, "Il a quelques amis.", "quelques", {"quelques": 1}),
        (spanish_hits, "Lo hice otra vez.", "otro", {}),
        (spanish_hits, "Lo hice otra vez.", "otra", {}),
        (spanish_hits, "Quiero otra casa.", "otra", {"otra": 1}),
    ],
)
def test_latin_hits_block_the_compounds_column(hits, sentence, form, expected) -> None:
    assert hits(sentence, frozenset({form})) == expected

