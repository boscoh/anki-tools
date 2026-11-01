"""Tests for the shared English gloss sense test."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.sentence_pack import english_sense
from anki_tools.sentence_pack.english_sense import (
    SenseRow,
    english_tokens,
    load_senses,
    read_senses,
    sense_failure,
    sense_mismatches,
    sense_row,
)
from anki_tools.sentence_pack.pool import FILTER_KEYS, FILTER_NAMES


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


@pytest.mark.parametrize(
    ("gloss", "english"),
    [
        ("I", "I'm one of you."),
        ("we", "It's ours."),
        ("we", "Let's go by taxi."),
        ("tongue", "I bit my tongue."),
        ("to hold", "Hold the candle."),
        ("to die", "He passed away."),
        ("blood", "I was bleeding a lot."),
        ("fingernail", "I broke a nail."),
        ("big", "What a huge dog!"),
        ("mother", "Is your mum at home?"),
        ("sun", "Solar power is clean."),
        ("tree bark", "The bark of this tree is rough."),
        ("to squeeze", "Squeeze the lemon."),
    ],
)
def test_holds_sense_accepts_the_gloss_sense(gloss: str, english: str) -> None:
    assert sense_failure(english, None, gloss) is None


@pytest.mark.parametrize(
    ("gloss", "english"),
    [
        ("tongue", "I love this language."),
        ("to hold", "Could you drive more slowly?"),
        ("to squeeze", "She elbowed her way onto the train."),
        ("tree bark", "We squeeze the orange and throw away the rind."),
        ("to burn", "Does she still have a temperature?"),
        ("big", "He is as tall as her."),
    ],
)
def test_holds_sense_rejects_a_different_sense(gloss: str, english: str) -> None:
    assert sense_failure(english, None, gloss) == "sense"


def test_holds_sense_ignores_a_gloss_without_an_entry() -> None:
    assert sense_failure("anything", None, "not a swadesh gloss") is None


def test_a_tier_s_false_friend_needs_no_sense_word() -> None:
    row = SenseRow(
        senses=frozenset({"hold"}), false_friends=frozenset({"drive"})
    )
    assert sense_failure("I drive a car.", row, "to hold") == "false_friend"
    assert sense_failure("Hold the drive.", row, "to hold") is None
    assert sense_failure("Hold it.", None, "to hold") is None


def test_a_tier_f_row_needs_a_sense_word_and_no_false_friend() -> None:
    row = SenseRow(
        senses=frozenset({"time"}), false_friends=frozenset({"weather"})
    )
    assert sense_failure("What time is it?", row) is None
    assert sense_failure("Time and weather.", row) == "false_friend"
    assert sense_failure("What day is it?", row) == "sense"
    assert sense_failure("anything", SenseRow()) is None
    assert sense_failure("anything") is None


def test_read_senses_keeps_a_tier_s_row_apart_from_a_tier_f_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "senses.tsv"
    path.write_text(
        "word\tsenses\tfalse_friends\tcompounds\ttier\n"
        "Voler\tFly\tSteal\t\tS\n"
        "voler\tsteal,rob\t\t\n"
        "\u6570\t\t\t\u6570\u5b66,\u4eba\u6570\tS\n",
        encoding="utf-8",
    )
    assert read_senses(path) == {
        ("S", "voler"): SenseRow(
            frozenset({"fly"}), frozenset({"steal"}), (), "fly"
        ),
        ("F", "voler"): SenseRow(
            frozenset({"steal", "rob"}), frozenset(), (), "steal"
        ),
        ("S", "\u6570"): SenseRow(
            frozenset(), frozenset(), ("\u6570\u5b66", "\u4eba\u6570"), ""
        ),
    }
    assert sense_row(path, "S", "VOLER") == read_senses(path)["S", "voler"]
    assert sense_row(path, "F", "absent") is None


def test_read_senses_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_senses(tmp_path / "absent.tsv") == {}

def test_sense_mismatches_reports_bad_cards() -> None:
    senses = {
        "tongue": english_sense.GlossSense(
            frozenset({"tongue"}), ()
        ),
    }
    rows = [
        {"target": "langue", "sentence": "J'ai mordu ma langue.", "english": "I bit my tongue."},
        {"target": "langue", "sentence": "J'aime cette langue.", "english": "I love this language."},
    ]
    bad = sense_mismatches(
        rows, {"langue": "tongue"}, Path("absent.tsv"), senses=senses
    )
    assert bad == ["langue in J'aime cette langue."]


def test_sense_stage_is_in_the_filter_report() -> None:
    assert "sense" in FILTER_KEYS
    assert FILTER_NAMES["sense"] == "english sense"
