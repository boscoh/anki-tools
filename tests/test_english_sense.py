"""Tests for the shared English gloss sense test."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.sentence_pack.english_sense import (
    GlossSense,
    SenseRow,
    english_tokens,
    load_senses,
    read_senses,
    sense_failure,
    sense_row,
)


def test_every_swadesh_label_has_a_sense_entry() -> None:
    labels = [
        line.strip()
        for line in Path("vocab/swadesh.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    senses = load_senses()
    missing = [label for label in labels if label.casefold() not in senses]
    assert missing == []


def test_contractions_expand_to_words() -> None:
    assert english_tokens("I'm one of you.") == ("i", "am", "one", "of", "you")
    assert "not" in english_tokens("Don't go.")
    assert english_tokens("Let's go.") == ("let", "us", "go")


GLOSSES = {
    "i": GlossSense(frozenset({"i"}), ()),
    "to hold": GlossSense(frozenset({"hold"}), ()),
    "belly": GlossSense(frozenset({"belly", "stomach"}), ()),
    "tree bark": GlossSense(frozenset(), ((frozenset({"tree"}), frozenset({"bark"})),)),
}
HOLD = SenseRow(senses=frozenset({"hold"}), false_friends=frozenset({"drive"}))
TIME = SenseRow(senses=frozenset({"time"}), false_friends=frozenset({"weather"}))
BELLY = SenseRow(senses=frozenset({"stomachache"}))


@pytest.mark.parametrize(
    ("english", "row", "gloss", "expected"),
    [
        ("I'm one of you.", None, "i", None),
        ("Hold the candle.", None, "to hold", None),
        ("Could you drive more slowly?", None, "to hold", "sense"),
        ("The tree bark is rough.", None, "tree bark", None),
        ("The bark of this tree is rough.", None, "tree bark", "sense"),
        ("", None, "to hold", "sense"),
        ("anything", None, "not a swadesh gloss", None),
        ("I drive a car.", HOLD, "to hold", "false_friend"),
        ("Hold the drive.", HOLD, "to hold", None),
        ("I have a stomach.", None, "belly", None),
        ("I have a stomachache.", None, "belly", "sense"),
        ("I have a stomachache.", BELLY, "belly", None),
        ("I have a headache.", BELLY, "belly", "sense"),
        ("What time is it?", TIME, None, None),
        ("Time and weather.", TIME, None, "false_friend"),
        ("What day is it?", TIME, None, "sense"),
        ("anything", SenseRow(), None, None),
        ("anything", None, None, None),
    ],
)
def test_a_sentence_passes_or_fails_the_english_sense_test(
    english: str, row: SenseRow | None, gloss: str | None, expected: str | None
) -> None:
    assert sense_failure(english, row, gloss, senses=GLOSSES) == expected


def test_read_senses_keeps_a_tier_s_row_apart_from_a_tier_f_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "senses.tsv"
    path.write_text(
        "word\tsenses\tfalse_friends\ttier\n"
        "Voler\tFly\tSteal\tS\n"
        "voler\tsteal,rob\t\t\n",
        encoding="utf-8",
    )
    assert read_senses(path) == {
        ("S", "voler"): SenseRow(frozenset({"fly"}), frozenset({"steal"}), "fly"),
        ("F", "voler"): SenseRow(frozenset({"steal", "rob"}), frozenset(), "steal"),
    }
    assert sense_row(path, "S", "VOLER") == read_senses(path)["S", "voler"]
    assert sense_row(path, "F", "absent") is None


def test_read_senses_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_senses(tmp_path / "absent.tsv") == {}
