"""Tests for the core run of the Mandarin and Cantonese profiles on a three-card deck."""

from __future__ import annotations

import json
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
        (1, "水", "water"),
        "我们都喝<b>水</b>.",
        "",
        id="mandarin",
    ),
    pytest.param(
        cantonese.CANTONESE_CORE_PROFILE,
        ["Front", "Back", "Audio", "Jyutping"],
        ["我哋都飲水。", "佢哋都飲水。", "我哋食魚肉。"],
        (1, "水", "water"),
        "我哋都飲<b>水</b>.",
        "ngo5 dei6 dou1 jam2 seoi2",
        id="cantonese",
    ),
]


@pytest.mark.parametrize(
    ("profile", "fields", "sentences", "target", "bold", "romanization"), CORES
)
def test_the_core_run_keeps_the_card_that_holds_the_core_word(
    tmp_path: Path,
    profile: CoreProfile,
    fields: list[str],
    sentences: list[str],
    target: tuple[int, str, str],
    bold: str,
    romanization: str,
) -> None:
    rows = [
        {
            profile.input_sentence_field: sentence,
            profile.input_english_field: "We all drink water.",
            profile.input_audio_field: f"[sound:a{index}.mp3]",
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
        load_frequency=list,
        frequency_ranks=lambda _rows, _weight: {},
        read_names=set,
        word_table=lambda: [target],
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
    assert notes[0][-2:] == [target[1], target[2]]
    assert notes[0][0].startswith("[sound:a")
    if romanization:
        assert notes[0][3] == romanization
