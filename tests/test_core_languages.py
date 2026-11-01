"""Tests for the core run of the Mandarin and Cantonese profiles on a three-card deck.

The segmenter, the matcher, the unit count, and the romanizer are fakes, so the
run checks that the adapters wire the input fields to the output note.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.cantonese import core as cantonese
from anki_tools.languages.mandarin import core as mandarin
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.scan import CoreProfile
from tests.cull_fixture import tiny_deck

OPTIONS = CoreOptions(max_cards=10, per_word=1, completion_cards=0, tail_cards=1)

CORES = [
    pytest.param(
        mandarin.MANDARIN_CORE_PROFILE,
        ["Sentence", "Sentence (Latin)", "Sentence (Translation)", "[a]"],
        ["我们都喝水。", "他们都喝水。", "我们吃鱼肉。"],
        "我们都喝<b>水</b>.",
        "pinyin-field",
        id="mandarin",
    ),
    pytest.param(
        cantonese.CANTONESE_CORE_PROFILE,
        ["Front", "Back", "Audio", "Jyutping"],
        ["我哋都飲水。", "佢哋都飲水。", "我哋食魚肉。"],
        "我哋都飲<b>水</b>.",
        "jyutping:我哋都飲水",
        id="cantonese",
    ),
]


@pytest.mark.usefixtures("fake_jyutping")
@pytest.mark.parametrize(("profile", "fields", "sentences", "bold", "romanization"), CORES)
def test_the_core_run_keeps_the_card_that_holds_the_core_word(
    tmp_path: Path,
    profile: CoreProfile,
    fields: list[str],
    sentences: list[str],
    bold: str,
    romanization: str,
) -> None:
    rows = [
        {
            profile.input_sentence_field: sentence,
            profile.input_english_field: "We all drink water.",
            profile.input_audio_field: f"[sound:a{index}.mp3]",
            profile.romanization_source or "": "pinyin-field",
        }
        for index, sentence in enumerate(sentences)
    ]
    source = tiny_deck(
        tmp_path,
        fields,
        rows,
        profile.input_model_name,
        [f"a{index}.mp3" for index in range(len(rows))],
    )
    profile = replace(
        profile,
        tokenize_deck=lambda cards: [
            [(char, "NOUN") for char in re.findall(r"\w", card.sentence)] for card in cards
        ],
        hits=lambda sentence, forms: {form: 1 for form in forms if sentence.count(form) == 1},
        highlight=lambda sentence, word: sentence.replace(word, f"<b>{word}</b>", 1),
        unit_count=lambda sentence: len(re.findall(r"\w", sentence)),
        load_frequency=list,
        frequency_ranks=lambda _rows, _weight: {},
        read_names=set,
        word_table=lambda: [(1, "水", "water")],
        sense_path=tmp_path / "absent_senses.tsv",
        default_options=OPTIONS,
    )

    run = coreflow.run_core(
        coreflow.CoreRequest(
            input_path=source,
            output_path=tmp_path / "core.apkg",
            core_words=0,
            max_cards=OPTIONS.max_cards,
        ),
        profile,
    )

    assert run.manifest_path is not None
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    with AnkiPackage(tmp_path / "core.apkg") as package:
        notes = [str(note["flds"]).split("\x1f") for note in package.get_notes()]
    assert manifest["counts"]["cards"] == len(notes) == 1
    assert notes[0][1] == bold
    assert notes[0][-2:] == ["水", "water"]
    assert notes[0][0].startswith("[sound:a")
    assert notes[0][3] == romanization
