"""Readers of one sentence deck from an APKG file.

A reader opens the package once. It resolves the note type by name, indexes the
fields, and returns one card per note in the input order with fresh indices. A
note without a sound tag is dropped.

The module also holds the output field names, because the assembler and the
readers state the same five names. A reader that imports them back from the
flow would close an import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import find_model_id, find_only_model_id

OUTPUT_AUDIO_FIELD = "Audio"
OUTPUT_SENTENCE_FIELD = "Sentence"
OUTPUT_ENGLISH_FIELD = "English"
OUTPUT_CORE_FIELD = "Core"
OUTPUT_GLOSS_FIELD = "CoreGloss"


@dataclass(frozen=True)
class InputCard:
    """One card of an input sentence deck.

    :param note_id: Anki note id.
    :param sentence: Sentence text after the language clean step.
    :param english: English translation.
    :param audio: Name of the audio file in the card.
    :param index: Position of the card in the input order.
    """

    note_id: int
    sentence: str
    english: str
    audio: str
    index: int


_SOUND_PATTERN = re.compile(r"\[sound:([^\]]+)\]")
_WRAPPED_QUOTES = re.compile(r'^["“](.*)["”]$', re.DOTALL)


def clean_english(text: str) -> str:
    """Strip the spaces and one pair of wrapping double quotes from English text.

    :param text: Raw English field value.
    :returns: The text without the wrapping quotes. Inner quotes stay.
    """
    return _WRAPPED_QUOTES.sub(r"\1", text.strip())


def sound_name(field_value: str) -> str:
    """Read the audio file name from a sound field.

    :param field_value: Raw field text, for example ``[sound:a.mp3]``.
    :returns: The file name, or an empty string when the field has no sound tag.
    """
    match = _SOUND_PATTERN.search(field_value)
    return match.group(1) if match else ""


def _input_cards(
    package: AnkiPackage,
    model_name: str,
    sentence_field: str,
    english_field: str,
    audio_field: str,
    clean_sentence: Callable[[str], str],
) -> list[InputCard]:
    """Read one card per note of the sentence model of an open package.

    A note without a sound tag is skipped.

    :param package: Open input package.
    :param model_name: Name of the note model.
    :param sentence_field: Name of the sentence field.
    :param english_field: Name of the English field.
    :param audio_field: Name of the audio field.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :raises RuntimeError: The package has no model with that name.
    :returns: One card per note, in the input order.
    """
    models = package.get_models()
    model_id = find_model_id(models, model_name)
    fields = [field["name"] for field in models[model_id]["flds"]]
    index = {name: position for position, name in enumerate(fields)}
    sentence = index[sentence_field]
    english = index[english_field]
    audio = index[audio_field]

    chosen: dict[int, Any] = {}
    for row in package.get_cards():
        if str(row["mid"]) != model_id:
            continue
        if len(row["flds"].split("\x1f")) != len(fields):
            continue
        note_id = int(row["nid"])
        current = chosen.get(note_id)
        if current is None or int(row["ord"]) < int(current["ord"]):
            chosen[note_id] = row

    cards: list[InputCard] = []
    for note_id, row in chosen.items():
        values = row["flds"].split("\x1f")
        sound = _SOUND_PATTERN.search(values[audio])
        if sound is None:
            continue
        cards.append(
            InputCard(
                note_id=note_id,
                sentence=clean_sentence(values[sentence]),
                english=clean_english(values[english]),
                audio=sound.group(1),
                index=len(cards),
            )
        )
    return cards


def read_input_deck(
    path: Path,
    model_name: str,
    sentence_field: str,
    english_field: str,
    audio_field: str,
    clean_sentence: Callable[[str], str],
) -> list[InputCard]:
    """Read one card per note of the sentence model in an APKG file.

    A note without a sound tag is skipped.

    :param path: Input APKG file.
    :param model_name: Name of the note model.
    :param sentence_field: Name of the sentence field.
    :param english_field: Name of the English field.
    :param audio_field: Name of the audio field.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :raises RuntimeError: The package has no model with that name.
    :returns: One card per note, in the input order.
    """
    with AnkiPackage(path) as package:
        return _input_cards(
            package,
            model_name,
            sentence_field,
            english_field,
            audio_field,
            clean_sentence,
        )


def drop_silent_cards(
    cards: Sequence[InputCard], media: Collection[str]
) -> list[InputCard]:
    """Keep the input cards whose audio file is in the media map of the package.

    The kept cards get new indices, so an index stays the position in the list.

    :param cards: Input cards, as :func:`read_input_deck` returns them.
    :param media: File names of the media map of the package.
    :returns: The cards that have an audio file.
    """
    heard = [card for card in cards if card.audio in media]
    return [replace(card, index=index) for index, card in enumerate(heard)]


def read_cards(path: Path, profile: Any) -> tuple[list[InputCard], int]:
    """Read the heard cards of the input deck of a profile.

    The function opens the package once. It filters the cards that have no
    audio file in the media map, and it counts them.

    :param path: Input APKG file.
    :param profile: Core or cull profile, which states the input field names
        and the sentence cleaner.
    :raises RuntimeError: The package has no model with the profile name.
    :returns: The cards that have an audio file, and the count of the dropped
        cards.
    """
    with AnkiPackage(path) as package:
        media = set(package.get_media_mapping().values())
        cards = _input_cards(
            package,
            profile.input_model_name,
            profile.input_sentence_field,
            profile.input_english_field,
            profile.input_audio_field,
            profile.clean_sentence,
        )
    kept = drop_silent_cards(cards, media)
    return kept, len(cards) - len(kept)


_BOLD_TAG = re.compile(r"</?b>")


def read_output_deck(path: Path) -> list[InputCard]:
    """Read one card per note of the only note type of an output APKG.

    The function reads the ``Audio``, ``Sentence``, and ``English`` fields, and
    it removes the bold tags from the sentence. A note without audio is
    skipped.

    :param path: Output APKG file.
    :raises RuntimeError: The package holds two or more note types.
    :returns: One card per note, in the input order.
    """
    cards: list[InputCard] = []
    with AnkiPackage(path) as package:
        models = package.get_models()
        model_id = find_only_model_id(models)
        names = [field["name"] for field in models[model_id]["flds"]]
        index = {name: position for position, name in enumerate(names)}
        audio_index = index[OUTPUT_AUDIO_FIELD]
        sentence_index = index[OUTPUT_SENTENCE_FIELD]
        english_index = index[OUTPUT_ENGLISH_FIELD]

        chosen: dict[int, Any] = {}
        for row in package.get_cards():
            if str(row["mid"]) != model_id:
                continue
            if len(row["flds"].split("\x1f")) != len(names):
                continue
            note_id = int(row["nid"])
            current = chosen.get(note_id)
            if current is None or int(row["ord"]) < int(current["ord"]):
                chosen[note_id] = row

        for note_id, row in chosen.items():
            values = row["flds"].split("\x1f")
            sound = _SOUND_PATTERN.search(values[audio_index])
            if sound is None:
                continue
            cards.append(
                InputCard(
                    note_id=note_id,
                    sentence=_BOLD_TAG.sub("", values[sentence_index]),
                    english=values[english_index].strip(),
                    audio=sound.group(1),
                    index=len(cards),
                )
            )
    return cards


def read_output(path: Path) -> dict[str, Any]:
    """Read the notes, the decks, and the media of an output APKG.

    :param path: Output APKG file.
    :raises RuntimeError: The package holds two or more note types.
    :returns: Cards, note count, field names, model names, deck names, media
        names, the card ordinals, the card templates and CSS per model, and the
        note type kind.
    """
    cards = read_output_deck(path)
    with AnkiPackage(path) as package:
        full = package.get_models(include_templates=True)
        model = full[find_only_model_id(full)]
        name = model.get("name")
        notes = package.get_notes()
        media = package.get_media_mapping()
        decks = {deck.get("name") for deck in package.get_decks().values()}
        ords = [int(row["ord"]) for row in package.get_cards()]
    return {
        "cards": cards,
        "notes": len(notes),
        "fields": [field["name"] for field in model.get("flds", ())],
        "note_models": {name},
        "decks": decks,
        "media": set(media.values()),
        "ords": ords,
        "templates": {
            name: [
                (template.get("qfmt", ""), template.get("afmt", ""))
                for template in model.get("tmpls", ())
            ]
        },
        "css": {name: model.get("css", "")},
        "kind": model.get("kind"),
    }
