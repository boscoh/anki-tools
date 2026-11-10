"""Tests for the Cantonese core adapter and its output note type."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.cantonese import core
from anki_tools.ranking.deckreader import read_cards
from tests.cull_fixture import tiny_deck, write_core_package


def _build_output(tmp_path: Path, jyutping: str) -> tuple[Path, int]:
    """Write the core package of one note and give its path and note id."""
    source = tiny_deck(
        tmp_path,
        ["Front", "Back", "Audio", "Jyutping"],
        [
            {
                "Front": "死開!",
                "Back": "Get lost!",
                "Audio": "[sound:a.mp3]",
                "Jyutping": jyutping,
            }
        ],
        core.MODEL_NAME,
        ["a.mp3"],
    )
    cards, _dropped = read_cards(source, core.CANTONESE_CORE_PROFILE)
    output = tmp_path / "core.apkg"
    write_core_package(source, output, cards, [0], core.CANTONESE_CORE_PROFILE)
    return output, cards[0].note_id


def test_the_jyutping_field_survives_the_new_model(tmp_path: Path) -> None:
    output, _note_id = _build_output(tmp_path, "séihōi ！")

    with AnkiPackage(output) as package:
        model = next(iter(package.get_models().values()))
        names = [field["name"] for field in model["flds"]]
        values = str(package.get_notes()[0]["flds"]).split("\x1f")

    assert names == ["Audio", "Sentence", "English", "Romanization", "Core", "CoreGloss"]
    assert values[3] == "séihōi ！"


@pytest.mark.usefixtures("fake_jyutping")
def test_finalize_writes_the_jyutping_of_the_sentence(tmp_path: Path) -> None:
    """The LTL deck holds Yale in its Jyutping field, so the fix derives it."""
    output, note_id = _build_output(tmp_path, "séihōi ！")
    core_csv = tmp_path / "core.csv"
    core_csv.write_text(
        f"note_id,kept,sentence\n{note_id},1,死開!\n", encoding="utf-8"
    )

    core.finalize_output(output, core_csv)

    with AnkiPackage(output) as package:
        values = str(package.get_notes()[0]["flds"]).split("\x1f")

    assert values[1] == "死開!"
    assert values[3] == "jyutping:死開"
