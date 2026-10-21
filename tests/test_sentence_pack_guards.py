"""Tests for the ordered guard table of the S2 candidate pool."""

from anki_tools.sentence_pack import pool
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
    assert pool.check_sentence_shape(sentence, MANDARIN, 4, 7) == "F5"
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
