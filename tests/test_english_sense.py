"""Tests for the shared English gloss sense test."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.sentence_pack.english_sense import (
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


HOLD = SenseRow(senses=frozenset({"hold"}), false_friends=frozenset({"drive"}))
TIME = SenseRow(senses=frozenset({"time"}), false_friends=frozenset({"weather"}))


@pytest.mark.parametrize(
    ("english", "row", "gloss", "expected"),
    [
        ("I'm one of you.", None, "I", None),
        ("It's ours.", None, "we", None),
        ("Let's go by taxi.", None, "we", None),
        ("I bit my tongue.", None, "tongue", None),
        ("Hold the candle.", None, "to hold", None),
        ("He passed away.", None, "to die", None),
        ("I was bleeding a lot.", None, "blood", None),
        ("I broke a nail.", None, "fingernail", None),
        ("What a huge dog!", None, "big", None),
        ("Is your mum at home?", None, "mother", None),
        ("Solar power is clean.", None, "sun", None),
        ("The bark of this tree is rough.", None, "tree bark", None),
        ("Squeeze the lemon.", None, "to squeeze", None),
        ("I love this language.", None, "tongue", "sense"),
        ("Could you drive more slowly?", None, "to hold", "sense"),
        ("She elbowed her way onto the train.", None, "to squeeze", "sense"),
        ("We squeeze the orange and throw away the rind.", None, "tree bark", "sense"),
        ("Does she still have a temperature?", None, "to burn", "sense"),
        ("He is as tall as her.", None, "big", "sense"),
        ("anything", None, "not a swadesh gloss", None),
        ("I drive a car.", HOLD, "to hold", "false_friend"),
        ("Hold the drive.", HOLD, "to hold", None),
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
    assert sense_failure(english, row, gloss) == expected


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


GLOSS_SIZE_BOUND = 125


@pytest.mark.parametrize(
    ("gloss", "loose"),
    [
        ("big", ["bad", "adult", "great", "heavy"]),
        ("fish", ["angle", "angling"]),
        ("sun", ["insolate", "sunday"]),
    ],
)
def test_gloss_list_rejects_loose_words(gloss: str, loose: list[str]) -> None:
    words = load_senses()[gloss].words
    assert not words.intersection(loose)


def test_gloss_list_size_is_bounded() -> None:
    sizes = {
        gloss: len(sense.words) + len(sense.phrases)
        for gloss, sense in load_senses().items()
    }
    assert max(sizes.values()) <= GLOSS_SIZE_BOUND, sizes
