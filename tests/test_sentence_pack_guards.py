"""Tests for the ordered guard tables and the language guard hooks."""

import pytest

from anki_tools.sentence_pack import pool
from anki_tools.sentence_pack.guards import (
    check_sentence_shape,
    occurrences,
)
from anki_tools.languages.mandarin.profile import MANDARIN
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

