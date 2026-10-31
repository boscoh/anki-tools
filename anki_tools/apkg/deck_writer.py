"""One writer for every output package.

A caller states the shape, the cards, the media, the side tables, the manifest,
and the report lines. This module selects the adapter: a new package, or an edit
of an input package. The caller states the output path and the manifest path;
this module states no name. ADR 0024 records the interface.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package


def sha256_value(path: Path) -> str:
    """Build the manifest hash value of a file.

    :param path: File to hash.
    :returns: ``sha256:`` followed by the hex digest.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class NoteTypeShape:
    """The note type of a written package.

    :param deck_name: Title of the output deck.
    :param model_name: Name of the output note type.
    :param fields: Ordered field names.
    :param templates: ``(qfmt, afmt)`` pair per card template.
    :param css: CSS of the note type.
    :param sort_field_index: Field that the browser sorts on.
    :param deck_description: Deck description text.
    :param input_model_name: Input note type that the edit adapter clones, or
        None for a new package.
    """

    deck_name: str
    model_name: str
    fields: tuple[str, ...]
    templates: tuple[tuple[str, str], ...]
    css: str
    sort_field_index: int = 0
    deck_description: str = ""
    input_model_name: str | None = None


@dataclass(frozen=True)
class Card:
    """One note of the output.

    :param values: Field value per field name.
    :param tags: Tags of the note.
    :param note_id: Input note id that the edit adapter rewrites, or None.
    """

    values: Mapping[str, str]
    tags: Sequence[str] = ()
    note_id: int | None = None


@dataclass(frozen=True)
class MediaFile:
    """One media file of the output.

    :param source: Path of the file on disk.
    :param name: Archive name of the file.
    """

    source: Path
    name: str


@dataclass(frozen=True)
class SideTable:
    """One side CSV of a run.

    :param suffix: File suffix after the output stem, for example ``core.csv``.
    :param columns: Column names, in file order.
    :param rows: One row mapping per line.
    """

    suffix: str
    columns: tuple[str, ...]
    rows: Sequence[Mapping[str, Any]]

    @property
    def label(self) -> str:
        """Give the manifest key of the file, for example ``core_csv``."""
        return f"{self.suffix.split('.')[0]}_csv"


@dataclass(frozen=True)
class ReserveCards:
    """Reserve cards that the edit adapter adds from another package.

    :param cards: Reserve cards of the run.
    :param pack_path: Package that holds the reserve media files.
    :param values: Callable that gives the field values of one reserve card.
    """

    cards: Sequence[Any]
    pack_path: Path
    values: Callable[[Any], Sequence[str]]


@dataclass(frozen=True)
class DeckWrite:
    """One write of one output package.

    :param output_path: Output APKG file.
    :param shape: Note type of the output.
    :param cards: Notes of the output.
    :param input_path: Input APKG file, or None for a new package.
    :param media: Media files of a new package.
    :param sides: Side CSV files of the run.
    :param manifest: Manifest payload. The writer fills its ``files`` key.
    :param manifest_path: Manifest JSON file, or None for no manifest.
    :param manifest_files: Files to hash into ``manifest.files``, keyed by the
        manifest key of each file.
    :param reserve: Reserve cards that the edit adapter adds.
    """

    output_path: Path
    shape: NoteTypeShape
    cards: Sequence[Card]
    input_path: Path | None = None
    media: Sequence[MediaFile] = ()
    sides: Sequence[SideTable] = ()
    manifest: Mapping[str, Any] | None = None
    manifest_path: Path | None = None
    manifest_files: Mapping[str, Path] = field(default_factory=dict)
    reserve: ReserveCards | None = None


@dataclass(frozen=True)
class RunFiles:
    """The files that one write produced.

    :param output_path: Output APKG file.
    :param manifest_path: Manifest JSON file, or None.
    :param sides: Side CSV files, keyed by suffix, for example ``core.csv``.
    """

    output_path: Path
    manifest_path: Path | None
    sides: Mapping[str, Path]


def write_deck(job: DeckWrite) -> RunFiles:
    """Write the package, the side files, and the manifest.

    :param job: Stated shape, cards, files, and manifest of one run.
    :raises ValueError: The job states a value that the adapter cannot honour.
    :returns: The written paths.
    """
    output = Path(job.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if job.input_path is None:
        _write_new(job, output)
    else:
        _write_edited(job, output)
    sides = write_sides(job)
    return RunFiles(
        output_path=output,
        manifest_path=Path(job.manifest_path) if job.manifest_path else None,
        sides=sides,
    )


def write_sides(job: DeckWrite) -> Mapping[str, Path]:
    """Write the side CSVs and the manifest of one run, beside the output.

    :param job: Stated shape, sides, and manifest of one run.
    :returns: The side path per suffix.
    """
    sides = {side.suffix: _write_csv(side, job.output_path) for side in job.sides}
    if job.manifest is not None and job.manifest_path is not None:
        hashes = {
            key: sha256_value(path) for key, path in job.manifest_files.items()
        }
        payload = dict(job.manifest)
        payload["files"] = {**payload.get("files", {}), **hashes}
        _write_json(payload, job.manifest_path)
    return sides


def _write_new(job: DeckWrite, output: Path) -> None:
    """Write a package from scratch.

    :param job: Stated write of one run.
    :param output: Output APKG file.
    :raises ValueError: The shape holds two templates, the media states another
        archive name than the file name, or the job states reserve cards.
    """
    shape = job.shape
    if len(shape.templates) != 1:
        raise ValueError("A new package holds one card template.")
    if job.reserve is not None:
        raise ValueError("Only the edit adapter adds reserve cards.")
    for media in job.media:
        if Path(media.name).name != media.name:
            raise ValueError(f"The archive name {media.name} holds a directory.")
    question, answer = shape.templates[0]
    AnkiPackage.create(
        output,
        shape.deck_name,
        fields=list(shape.fields),
        cards=[
            {**dict(card.values), "tags": list(card.tags)} for card in job.cards
        ],
        media_files=[str(media.source) for media in job.media],
        model_name=shape.model_name,
        question_format=question,
        answer_format=answer,
        css=shape.css,
        sort_field_index=shape.sort_field_index,
        deck_description=shape.deck_description,
    )


def _write_edited(job: DeckWrite, output: Path) -> None:
    """Write a package by editing the input package.

    :param job: Stated write of one run.
    :param output: Output APKG file.
    :raises ValueError: A card has no note id, the shape has no input model
        name, or the job states media files.
    """
    shape = job.shape
    input_path = job.input_path
    if input_path is None:
        raise ValueError("The edit adapter needs the input path.")
    if shape.input_model_name is None:
        raise ValueError("The edit adapter needs the input model name.")
    if job.media:
        raise ValueError("The edit adapter keeps the media of the input package.")
    values: dict[int, list[str]] = {}
    for card in job.cards:
        if card.note_id is None:
            raise ValueError("The edit adapter needs the note id of each card.")
        values[card.note_id] = [card.values.get(name, "") for name in shape.fields]
    reserve = ()
    reserve_pack_path = None
    reserve_values = None
    if job.reserve is not None:
        reserve = job.reserve.cards
        reserve_pack_path = job.reserve.pack_path
        reserve_values = job.reserve.values
    write_package(
        input_path,
        output,
        OutputShape(
            deck_name=shape.deck_name,
            model_name=shape.model_name,
            input_model_name=shape.input_model_name,
            fields=shape.fields,
            templates=shape.templates,
            css=shape.css,
        ),
        values,
        reserve=reserve,
        reserve_pack_path=reserve_pack_path,
        reserve_values=reserve_values,
    )


def _write_csv(side: SideTable, output_path: Path) -> Path:
    """Write one side CSV beside the output file.

    :param side: Side table of the run.
    :param output_path: Output APKG file.
    :returns: The written path.
    """
    target = Path(output_path).with_name(f"{Path(output_path).stem}.{side.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(side.columns))
        writer.writeheader()
        for row in side.rows:
            writer.writerow(row)
    return target


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write one JSON file.

    :param payload: JSON object.
    :param path: Output JSON file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_run(lines: Sequence[tuple[str | None, str]], files: RunFiles) -> None:
    """Print the report of one run and the files that it wrote.

    :param lines: Report lines of the run: a label and a value, or a raw line
        when the label is None.
    :param files: The written paths.
    """
    for label, value in lines:
        if label is None:
            print(value)
        else:
            print(f"{label + ':':<21}{value}")
    print()
    print(f"wrote {files.output_path}")
    for path in files.sides.values():
        print(f"wrote {path}")
    if files.manifest_path is not None:
        print(f"wrote {files.manifest_path}")
