"""One shape test on an anki2 fixture and on an anki21b fixture.

The written package must hold one deck, one standard note type with the stated
fields, and one card per note. The two package formats are the adapters of the
seam.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from anki_tools.apkg import deck_writer
from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.ranking.identity import DeckIdentity
from tests.deck_fixture import MODEL_NAME as ANKI2_INPUT_MODEL
from tests.deck_fixture import build_contract_deck
from tests.test_apkg_package import (
    ANKI21B_NOTE_TYPE_NAME,
    build_anki21b_deck,
)

FIELDS = ("Audio", "Sentence", "English", "Core", "CoreGloss")
TEMPLATES = (("{{Audio}}", "{{Audio}}<hr id=answer>{{Sentence}}"),)
CSS = ".card { color: black; }"


def _shape(input_model_name: str, deck_name: str) -> OutputShape:
    return OutputShape(
        deck_name=deck_name,
        model_name=deck_name,
        input_model_name=input_model_name,
        fields=FIELDS,
        templates=TEMPLATES,
        css=CSS,
    )


def _values(input_path: Path, count: int) -> dict[int, list[str]]:
    with AnkiPackage(input_path) as package:
        note_ids = [int(row["id"]) for row in package.get_notes()][:count]
    return {
        note_id: ["[sound:a.mp3]", "Le chat dort.", "The cat sleeps.", "chat", "cat"]
        for note_id in note_ids
    }


def _write(input_path: Path, output: Path, input_model: str) -> OutputShape:
    shape = _shape(input_model, DeckIdentity(output).model_name)
    write_package(input_path, output, shape, _values(input_path, 2))
    return shape


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _config_field(config: bytes, wanted: int) -> int | None:
    pos = 0
    while pos < len(config):
        tag = config[pos]
        pos += 1
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _varint(config, pos)
            if field == wanted:
                return value
        elif wire == 2:
            length, pos = _varint(config, pos)
            pos += length
        else:
            return None
    return None


def _assert_shape(output: Path, shape: OutputShape) -> None:
    with AnkiPackage(output) as package:
        models = package.get_models(include_templates=True)
        decks = {deck["name"] for deck in package.get_decks().values()}
        cards = package.get_cards()

    assert decks == {shape.deck_name}
    assert [model["name"] for model in models.values()] == [shape.model_name]
    model = next(iter(models.values()))
    assert [field["name"] for field in model["flds"]] == list(FIELDS)
    assert [(tmpl["qfmt"], tmpl["afmt"]) for tmpl in model["tmpls"]] == list(TEMPLATES)
    assert len(cards) == 2
    assert {int(row["ord"]) for row in cards} == {0}


def test_the_shape_holds_on_an_anki2_fixture(tmp_path: Path) -> None:
    source = tmp_path / "input.apkg"
    shutil.copy(build_contract_deck(tmp_path), source)
    output = tmp_path / "shape.apkg"

    shape = _write(source, output, ANKI2_INPUT_MODEL)

    _assert_shape(output, shape)
    with AnkiPackage(output) as package:
        model = next(iter(package.get_models().values()))
    assert model["type"] == 0
    assert model["kind"] == 0


def test_the_shape_holds_on_an_anki21b_fixture(tmp_path: Path) -> None:
    pytest.importorskip("zstandard")
    source = build_anki21b_deck(tmp_path / "input.apkg")
    output = tmp_path / "shape.apkg"

    shape = _write(source, output, ANKI21B_NOTE_TYPE_NAME)

    _assert_shape(output, shape)
    with AnkiPackage(output) as package:
        assert package.db_format == "anki21b"
        row = package.conn.execute(
            "SELECT config FROM notetypes WHERE name = ?", (shape.model_name,)
        ).fetchone()
        model = next(iter(package.get_models().values()))
    assert _config_field(row["config"] or b"", 1) == 0
    assert _config_field(row["config"] or b"", 2) == 1
    assert model["kind"] == 0


def test_the_new_package_adapter_states_the_same_shape(tmp_path: Path) -> None:
    output = tmp_path / "new.apkg"
    files = deck_writer.write_deck(
        deck_writer.DeckWrite(
            output_path=output,
            shape=deck_writer.NoteTypeShape(
                deck_name="New Deck",
                model_name="New Deck",
                fields=FIELDS,
                templates=TEMPLATES,
                css=CSS,
            ),
            cards=[
                deck_writer.Card(
                    values={
                        "Audio": "[sound:a.mp3]",
                        "Sentence": "Le chat dort.",
                        "English": "The cat sleeps.",
                        "Core": "chat",
                        "CoreGloss": "cat",
                    }
                )
            ],
        )
    )

    assert files.output_path == output
    with AnkiPackage(output) as package:
        model = next(iter(package.get_models(include_templates=True).values()))
        cards = package.get_cards()
    assert [field["name"] for field in model["flds"]] == list(FIELDS)
    assert [(t["qfmt"], t["afmt"]) for t in model["tmpls"]] == list(TEMPLATES)
    assert len(cards) == 1


def test_the_edit_adapter_rejects_media(tmp_path: Path) -> None:
    source = tmp_path / "input.apkg"
    shutil.copy(build_contract_deck(tmp_path), source)

    with pytest.raises(ValueError, match="media"):
        deck_writer.write_deck(
            deck_writer.DeckWrite(
                output_path=tmp_path / "out.apkg",
                input_path=source,
                shape=deck_writer.NoteTypeShape(
                    deck_name="Edited",
                    model_name="Edited",
                    fields=FIELDS,
                    templates=TEMPLATES,
                    css=CSS,
                    input_model_name=ANKI2_INPUT_MODEL,
                ),
                cards=[],
                media=[deck_writer.MediaFile(source, "a.mp3")],
            )
        )
