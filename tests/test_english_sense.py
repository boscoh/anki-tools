"""Tests for the shared English gloss sense test."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.sentence_pack import english_sense
from anki_tools.sentence_pack.english_sense import (
    english_tokens,
    holds_sense,
    load_senses,
    sense_mismatches,
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
    assert holds_sense(gloss, english)


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
    assert not holds_sense(gloss, english)


def test_holds_sense_ignores_a_gloss_without_an_entry() -> None:
    assert holds_sense("not a swadesh gloss", "anything")


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
    bad = sense_mismatches(rows, {"langue": "tongue"}, senses=senses)
    assert bad == ["langue in J'aime cette langue."]


def test_sense_stage_is_in_the_filter_report() -> None:
    assert "sense" in FILTER_KEYS
    assert FILTER_NAMES["sense"] == "english sense"
