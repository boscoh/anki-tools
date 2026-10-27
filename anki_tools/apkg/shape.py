"""The stated shape of a written package.

An output package holds one deck, one standard note type with the stated
fields, and one card per note. This module states that shape. The caller states
the values. The ``anki2`` and ``anki21b`` adapters sit behind the seam: the
``anki2`` note type holds the output deck link, and the ``anki21b`` note type
has no such field. Both set the standard kind. ADR 0015 records the decision.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from anki_tools.apkg.package import AnkiPackage

#: Kind of the standard note type. Anki calls the other kind ``cloze``.
STANDARD_KIND = 0

#: Tag of the reserve notes.
RESERVE_TAGS = " swadesh "



def find_model_id(models: Mapping[str, Mapping[str, Any]], name: str) -> str:
    """Find the id of a note type by name.

    :param models: Note types of a package, as :meth:`AnkiPackage.get_models`
        gives them.
    :param name: Note type name.
    :raises RuntimeError: No note type has the name.
    :returns: Model id as a string.
    """
    for model_id, model in models.items():
        if model.get("name") == name:
            return model_id
    raise RuntimeError(f"The input APKG has no model {name!r}.")


def find_only_model_id(models: Mapping[str, Mapping[str, Any]]) -> str:
    """Find the id of the only note type of a package.

    :param models: Note types of a package, as :meth:`AnkiPackage.get_models`
        gives them.
    :raises RuntimeError: The package holds two or more note types.
    :returns: Model id as a string.
    """
    if len(models) != 1:
        raise RuntimeError(f"The package holds {len(models)} note types, not one.")
    return next(iter(models))


@dataclass(frozen=True)
class OutputShape:
    """One deck, one standard note type with the stated fields, one card per note.

    :param deck_name: Title of the output deck.
    :param model_name: Name of the output note type, which equals the deck name.
    :param input_model_name: Name of the input note type that the run reads.
    :param fields: Ordered field names of the output note type.
    :param templates: ``(qfmt, afmt)`` pair per card template.
    :param css: CSS of the output note type.
    """

    deck_name: str
    model_name: str
    input_model_name: str
    fields: tuple[str, ...]
    templates: tuple[tuple[str, str], ...]
    css: str


@dataclass(frozen=True)
class _DeckLink:
    """Adapter of one package format across the note type seam.

    :param linked: True when the note type holds the output deck link.
    """

    linked: bool

    def note_type(
        self,
        package: AnkiPackage,
        source_model_id: str,
        shape: OutputShape,
        deck_id: int,
    ) -> str:
        """Give the output note type of the stated shape.

        :param package: Open output package.
        :param source_model_id: Model id that the new note type copies.
        :param shape: Stated shape of the output note type.
        :param deck_id: Deck id of the output deck.
        :returns: Model id of the new note type.
        """
        extra: dict[str, Any] = {"deck_id": deck_id} if self.linked else {}
        model_id = package.clone_model(
            source_model_id,
            shape.model_name,
            list(shape.fields),
            list(shape.templates),
            shape.css,
            **extra,
        )
        package.set_model_kind(model_id, STANDARD_KIND)
        return model_id


ANKI2 = _DeckLink(linked=True)
"""The ``anki2`` adapter: the note type holds the output deck link."""

ANKI21B = _DeckLink(linked=False)
"""The ``anki21b`` adapter: this format has no deck link field."""

FORMAT_ADAPTERS: dict[str, _DeckLink] = {"anki2": ANKI2, "anki21b": ANKI21B}


def rename_deck(package: AnkiPackage, deck_name: str) -> int:
    """Rename the deck of the kept cards and delete the decks that stay empty.

    :param package: Open package.
    :param deck_name: New deck name.
    :raises RuntimeError: The collection has no card.
    :returns: The deck id of the renamed deck.
    """
    cards = package.get_cards()
    counts = Counter(str(row["did"]) for row in cards)
    if not counts:
        raise RuntimeError("The collection has no card to place in a deck.")
    main = int(counts.most_common(1)[0][0])
    for row in cards:
        if int(row["did"]) != main:
            package.move_card(int(row["id"]), main)
    package.set_deck_name(main, deck_name)
    for deck_id in list(package.get_decks()):
        if int(deck_id) != main:
            package.delete_deck(int(deck_id))
    return main


def insert_reserve(
    package: AnkiPackage,
    reserve: Sequence[Any],
    reserve_pack_path: Path,
    deck_id: int,
    model_id: str,
    reserve_values: Callable[[Any], Sequence[str]],
) -> None:
    """Insert the reserve notes and cards into an open package.

    :param package: Open output package after the non-kept cards are deleted.
    :param reserve: Reserve cards to insert.
    :param reserve_pack_path: Pack with the reserve media files.
    :param deck_id: Target deck id.
    :param model_id: Model id of the output note type.
    :param reserve_values: Callable that gives the field values of one card.
    :raises RuntimeError: A reserve audio file is not in the pack.
    """
    mod = int(time.time() * 1000)
    with AnkiPackage(reserve_pack_path) as swadesh:
        media_dir = swadesh.media_dir()
        if media_dir is None:
            raise RuntimeError("The reserve pack has no temporary directory.")
        media = swadesh.get_media_mapping()
        reverse = {name: key for key, name in media.items()}
        for card in reserve:
            key = reverse.get(card.audio)
            if key is None:
                raise RuntimeError(
                    f"The reserve audio {card.audio} is not in {reserve_pack_path}."
                )
            package.add_media_file(Path(media_dir) / key, card.audio)
    for card in reserve:
        package.add_note_and_card(
            reserve_values(card),
            model_id,
            deck_id,
            RESERVE_TAGS,
            guid=str(uuid.uuid4()),
            mod=mod,
            sfld=card.sentence,
        )


def write_package(
    input_path: Path,
    output_path: Path,
    shape: OutputShape,
    values: Mapping[int, Sequence[str]],
    *,
    reserve: Sequence[Any] = (),
    reserve_pack_path: Path | None = None,
    reserve_values: Callable[[Any], Sequence[str]] | None = None,
) -> dict[str, int]:
    """Write the kept notes as a package of the stated shape.

    The cards of every note that the run does not keep are deleted, and the
    media files that no kept card uses are removed with them. The kept notes
    move to a clone of the input note type, and every other note type goes.

    :param input_path: Input APKG file.
    :param output_path: Output APKG file.
    :param shape: Stated shape of the output package.
    :param values: Output field values of each kept note, keyed by note id.
    :param reserve: Reserve cards to add to the output note type.
    :param reserve_pack_path: Pack with the reserve media files.
    :param reserve_values: Callable that gives the field values of one reserve
        card. A run with reserve cards needs it.
    :raises RuntimeError: The output note count differs from the kept count.
    :returns: Note, media, and card counts of the output.
    """
    kept_notes = set(values)
    if not kept_notes:
        raise ValueError("A written package needs one kept note.")
    sort_index = shape.fields.index("Sentence") if "Sentence" in shape.fields else 0
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with AnkiPackage(input_path) as package:
        doomed = [
            int(row["id"])
            for row in package.get_cards()
            if int(row["nid"]) not in kept_notes
        ]
        if doomed:
            package.delete_cards(doomed, cleanup_audio=False)
        models = package.get_models()
        input_id = find_model_id(models, shape.input_model_name)
        package.prune_extra_cards(input_id)
        deck_id = rename_deck(package, shape.deck_name)
        adapter = FORMAT_ADAPTERS[package.db_format or "anki2"]
        output_id = adapter.note_type(package, input_id, shape, deck_id)
        for row in package.get_notes():
            package.update_note(
                int(row["id"]), values[int(row["id"])], output_id, sort_index
            )
        for model_id in list(models):
            package.delete_model(model_id)
        if reserve:
            if reserve_pack_path is None or reserve_values is None:
                raise RuntimeError(
                    "A run with reserve cards needs a pack path and values."
                )
            insert_reserve(
                package,
                reserve,
                reserve_pack_path,
                int(next(iter(package.get_decks()))),
                output_id,
                reserve_values,
            )
        package.prune_media(package.used_media())
        package.touch()
        package.save(target)

    expected = len(kept_notes) + len(reserve)
    with AnkiPackage(target) as package:
        notes = len(package.get_notes())
        media = len(package.get_media_mapping())
    if notes != expected:
        raise RuntimeError(f"The output APKG has {notes} notes, not {expected}.")
    return {"notes": notes, "media": media, "items": expected}
