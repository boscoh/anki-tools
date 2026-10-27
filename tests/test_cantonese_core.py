"""Tests for the Cantonese core adapter and its output note type."""

from __future__ import annotations

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


def test_the_jyutping_field_survives_the_new_model(tmp_path) -> None:
    """The extra field sits at position 4 and keeps the input value."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=["Front", "Back", "Audio", "Jyutping"],
        cards=[
            {
                "Front": "我睇咗一本書.",
                "Back": "I read a book.",
                "Audio": "[sound:a.mp3]",
                "Jyutping": "ngo5 tai2 zo2 jat1 bun2 syu1",
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

    with AnkiPackage(output) as package:
        model = next(iter(package.get_models().values()))
        names = [field["name"] for field in model["flds"]]
        values = str(package.get_notes()[0]["flds"]).split("\x1f")

    assert names == ["Audio", "Sentence", "English", "Romanization", "Core", "CoreGloss"]
    assert values[3] == "ngo5 tai2 zo2 jat1 bun2 syu1"
