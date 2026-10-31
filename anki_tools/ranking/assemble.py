"""One assembler for the deck, the side files, the manifest, and the report.

The core selection and the cull selection are the two adapters across this
seam. Each adapter states the values, the tables, and the report lines of its
mode. This module holds the output shape, the write order, the manifest schema,
and the report layout once. The identity module names every artifact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from anki_tools.apkg import deck_writer
from anki_tools.apkg.templates import TemplateSpec, assemble_templates
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.reorder import reorder_deck

#: Columns of the rank CSV that ``reorder_deck`` reads.
RANK_COLUMNS = ("rank", "note_id", "sentence", "kept")

#: Field names of the output note type, in order, without the romanization.
OUTPUT_FIELDS: tuple[str, ...] = ("Audio", "Sentence", "English", "Core", "CoreGloss")
OUTPUT_ROMANIZATION_FIELD = "Romanization"


class DeckProfile(Protocol):
    """Contract of the output values and the written note type."""

    @property
    def input_model_name(self) -> str:
        """Give the name of the input note type."""
        ...

    @property
    def input_sentence_field(self) -> str:
        """Give the name of the input sentence field."""
        ...

    @property
    def input_english_field(self) -> str:
        """Give the name of the input English field."""
        ...

    @property
    def input_audio_field(self) -> str:
        """Give the name of the input audio field."""
        ...

    @property
    def romanization_source(self) -> str | None:
        """Give the input field of the romanization line, or None."""
        ...

    @property
    def templates(self) -> tuple[tuple[str, str], ...]:
        """Give the card templates of the output."""
        ...

    @property
    def highlight(self) -> Callable[[str, str], str] | None:
        """Give the bold wrapper of the target tokens, or None."""
        ...

    def card_css(self) -> str:
        """Give the CSS of the output note type."""
        ...


def deck_templates(profile: DeckProfile) -> tuple[tuple[str, str], ...]:
    """Give the templates of a run, minus the romanization block when the
    language has no romanization field.

    :param profile: Language hooks of the run.
    :returns: The template pairs of the output note type.
    """
    fields = output_fields(profile)
    pairs: list[tuple[str, str]] = []
    for qfmt, afmt in profile.templates:
        pair = assemble_templates(
            TemplateSpec(
                front=qfmt,
                back=afmt,
                fields=fields,
                optional_blocks=("Romanization",),
            )
        )
        pairs.append((pair.front, pair.back))
    return tuple(pairs)


def output_fields(profile: DeckProfile) -> tuple[str, ...]:
    """Give the output field names of a deck profile.

    :param profile: Language hooks of the run.
    :returns: ``Audio``, ``Sentence``, ``English``, the optional
        ``Romanization``, ``Core``, and ``CoreGloss``.
    """
    names = [*OUTPUT_FIELDS[:3]]
    if profile.romanization_source is not None:
        names.append(OUTPUT_ROMANIZATION_FIELD)
    names.extend(OUTPUT_FIELDS[3:])
    return tuple(names)



@dataclass(frozen=True)
class ManifestValues:
    """The manifest values that only a mode knows.

    :param input_model: Name of the input note type of the run.
    :param params: Options of the run. The assembler adds ``name``.
    :param counts: Card, lemma, and target counts.
    :param frequency: Frequency source path and the spoken weight.
    :param kept_order: Hash of the kept sentences in their final order.
    :param licenses: License text per data source, or None for no block.
    :param reserve_pack: Reserve pack of the run, empty without a reserve.
    :param swadesh_exceptions: Reserve concepts that no source sentence covers.
    """

    input_model: str
    params: Mapping[str, Any]
    counts: Mapping[str, int]
    frequency: Mapping[str, Any]
    kept_order: str
    licenses: Mapping[str, str] | None = None
    reserve_pack: str = ""
    swadesh_exceptions: Sequence[str] = ()


@dataclass(frozen=True)
class DeckRun:
    """The selected cards of one run and the files that the assembler writes.

    :param identity: Output name of the run.
    :param input_path: Input APKG file.
    :param values: Output field values of each kept note, keyed by note id.
    :param card_table: Card CSV of the mode.
    :param coverage_table: Coverage CSV of the mode.
    :param manifest: Manifest values of the mode.
    :param report: Report lines of the mode: a label and a value, or a raw
        line when the label is None.
    :param rank_rows: ``(rank, note_id, sentence, kept)`` row per kept card for
        a run that reorders the deck, or None.
    :param reserve: Reserve cards to add to the output note type.
    :param reserve_pack_path: Pack with the reserve media files.
    :param reserve_values: Callable that gives the field values of one reserve
        card.
    """

    identity: DeckIdentity
    input_path: Path
    values: Mapping[int, Sequence[str]]
    card_table: deck_writer.SideTable
    coverage_table: deck_writer.SideTable
    manifest: ManifestValues
    report: Sequence[tuple[str | None, str]]
    rank_rows: Sequence[tuple[int, int, str, int]] | None = None
    reserve: Sequence[Any] = ()
    reserve_pack_path: Path | None = None
    reserve_values: Callable[[Any], Sequence[str]] | None = None


def write_run(run: DeckRun, profile: DeckProfile) -> deck_writer.RunFiles:
    """Write the deck, the side files, the manifest, and the report of one run.

    The output shape of the package comes from the output identity and the
    language hooks, so a mode states only the values that it alone knows. The
    core mode and the cull mode state a DeckRun, and this function is their
    adapter over the deck writer.

    :param run: Selected cards and stated values of the run.
    :param profile: Language hooks of the run.
    :returns: The written paths.
    """
    identity = run.identity
    fields = output_fields(profile)
    shape = deck_writer.NoteTypeShape(
        deck_name=identity.title,
        model_name=identity.model_name,
        fields=fields,
        templates=deck_templates(profile),
        css=profile.card_css(),
        input_model_name=profile.input_model_name,
    )
    job = deck_writer.DeckWrite(
        output_path=identity.output_path,
        input_path=run.input_path,
        shape=shape,
        cards=[
            deck_writer.Card(values=dict(zip(fields, values)), note_id=note_id)
            for note_id, values in run.values.items()
        ],
        sides=(run.card_table, run.coverage_table),
        manifest=manifest_payload(run),
        manifest_path=identity.side("manifest.json"),
        manifest_files={
            run.card_table.label: identity.side(run.card_table.suffix),
            run.coverage_table.label: identity.side(run.coverage_table.suffix),
            "input_apkg": run.input_path,
        },
        reserve=_reserve(run),
    )
    files = _write_reordered(job, run)
    deck_writer.print_run(run.report, files)
    return files


def _reserve(run: DeckRun) -> deck_writer.ReserveCards | None:
    """Give the reserve block of a run, or None when the run has none.

    :param run: Selected cards and stated values of the run.
    :raises RuntimeError: The run has reserve cards but no pack path or no
        value callable.
    """
    if not run.reserve:
        return None
    if run.reserve_pack_path is None or run.reserve_values is None:
        raise RuntimeError(
            "A run with reserve cards needs a pack path and values."
        )
    return deck_writer.ReserveCards(
        cards=run.reserve,
        pack_path=run.reserve_pack_path,
        values=run.reserve_values,
    )


def _write_reordered(
    job: deck_writer.DeckWrite, run: DeckRun
) -> deck_writer.RunFiles:
    """Write the run, then reorder the deck when the run states a rank table.

    :param job: Stated write of the run.
    :param run: Selected cards and stated values of the run.
    :returns: The written paths.
    """
    if run.rank_rows is None:
        return deck_writer.write_deck(job)
    rank_csv = run.identity.side("rank.csv")
    write_rank_csv(run.rank_rows, rank_csv)
    temporary = job.output_path.with_name(job.output_path.stem + ".tmp.apkg")
    deck_writer.write_deck(
        replace(job, output_path=temporary, sides=(), manifest=None)
    )
    reorder_deck(str(temporary), str(job.output_path), str(rank_csv))
    temporary.unlink(missing_ok=True)
    sides = deck_writer.write_sides(job)
    return deck_writer.RunFiles(
        output_path=job.output_path,
        manifest_path=job.manifest_path,
        sides=sides,
    )


def write_rank_csv(rows: Sequence[tuple[int, int, str, int]], path: Path) -> None:
    """Write the ranking CSV that ``reorder_deck`` reads.

    :param rows: ``(rank, note_id, sentence, kept)`` row per kept card.
    :param path: Output CSV file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RANK_COLUMNS)
        for row in rows:
            writer.writerow(row)



def manifest_payload(run: DeckRun) -> dict[str, Any]:
    """Give the manifest payload of a run, in its stated key order.

    :param run: Selected cards and stated values of the run.
    :returns: The manifest object, with an empty ``files`` block that the deck
        writer fills.
    """
    identity = run.identity
    values = run.manifest
    payload: dict[str, Any] = {
        "build_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "input": str(run.input_path),
            "deck": identity.title,
            "model": identity.model_name,
            "input_model": values.input_model,
        },
        "params": {**values.params, "name": identity.title},
        "counts": dict(values.counts),
        "frequency": dict(values.frequency),
        "files": {},
    }
    if values.licenses is not None:
        payload["licenses"] = dict(values.licenses)
    payload["kept_order"] = values.kept_order
    if values.reserve_pack:
        payload["reserve_pack"] = values.reserve_pack
        payload["swadesh_exceptions"] = list(values.swadesh_exceptions)
    return payload



