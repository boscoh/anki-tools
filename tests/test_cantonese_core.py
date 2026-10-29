"""Tests for the Cantonese core adapter and its output note type."""

from __future__ import annotations
from pathlib import Path

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.languages.cantonese import core
from anki_tools.ranking import assemble, coreflow


def test_profile_holds_the_deck_paths():
    """The profile names the LTL deck files and the Cantonese hooks."""
    profile = core.CANTONESE_CORE_PROFILE
    assert profile.input_model_name == "LTL Note Type - Cantonese"
    assert profile.input_sentence_field == "Front"
    assert profile.input_english_field == "Back"
    assert profile.input_audio_field == "Audio"
    assert profile.romanization_source == "Jyutping"
    assert profile.min_units < profile.max_units
    assert profile.templates
    assert profile.card_css().startswith("/*")


def _build_output(tmp_path, jyutping: str) -> tuple[Path, int]:
    """Write the core package of one note and give its path and note id."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=["Front", "Back", "Audio", "Jyutping"],
        cards=[
            {
                "Front": "死開!",
                "Back": "Get lost!",
                "Audio": "[sound:a.mp3]",
                "Jyutping": jyutping,
            }
        ],
        media_files=[str(audio)],
        model_name=core.MODEL_NAME,
    )
    cards = core.load_input_deck(source)
    output = tmp_path / "core.apkg"
    profile = core.CANTONESE_CORE_PROFILE
    identity = DeckIdentity(output)
    note_fields, _bolded, _unbolded = coreflow.note_values(
        source, cards, [0], profile, {}
    )
    write_package(
        source,
        output,
        OutputShape(
            deck_name=identity.title,
            model_name=identity.model_name,
            input_model_name=profile.input_model_name,
            fields=assemble.output_fields(profile),
            templates=assemble.deck_templates(profile),
            css=profile.card_css(),
        ),
        note_fields,
    )
    return output, cards[0].note_id


def test_the_jyutping_field_survives_the_new_model(tmp_path) -> None:
    """The extra field sits at position 4 and keeps the input value."""
    output, _note_id = _build_output(tmp_path, "séihōi ！")

    with AnkiPackage(output) as package:
        model = next(iter(package.get_models().values()))
        names = [field["name"] for field in model["flds"]]
        values = str(package.get_notes()[0]["flds"]).split("\x1f")

    assert names == ["Audio", "Sentence", "English", "Romanization", "Core", "CoreGloss"]
    assert values[3] == "séihōi ！"


def test_finalize_writes_the_jyutping_of_the_sentence(tmp_path) -> None:
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
    assert values[3] == "sei2 hoi1"
