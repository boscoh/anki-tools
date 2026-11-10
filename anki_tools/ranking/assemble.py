"""One assembler for the deck, the side files, the manifest, and the report.

The core selection and the cull selection are the two adapters across this
seam. Each adapter states the values, the tables, and the report lines of its
mode. This module holds the output shape, the write order, the manifest schema,
and the report layout once. The identity module names every artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from anki_tools.apkg import deck_writer
from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import TemplateSpec, assemble_templates
from anki_tools.ranking.deckreader import (
    OUTPUT_AUDIO_FIELD,
    OUTPUT_CORE_FIELD,
    OUTPUT_ENGLISH_FIELD,
    OUTPUT_GLOSS_FIELD,
    OUTPUT_SENTENCE_FIELD,
)
from anki_tools.ranking.identity import DeckIdentity

#: Field names of the output note type, in order, without the romanization.
OUTPUT_FIELDS: tuple[str, ...] = (
    OUTPUT_AUDIO_FIELD,
    OUTPUT_SENTENCE_FIELD,
    OUTPUT_ENGLISH_FIELD,
    OUTPUT_CORE_FIELD,
    OUTPUT_GLOSS_FIELD,
)
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
    :param card_order: Note ids of the kept cards in rank order for a run that
        sets the due values, or None.
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
    card_order: Sequence[int] | None = None
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
    files = _write_ordered(job, run)
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


def _write_ordered(
    job: deck_writer.DeckWrite, run: DeckRun
) -> deck_writer.RunFiles:
    """Write the run, then set the due value of each new card to its rank.

    :param job: Stated write of the run.
    :param run: Selected cards and stated values of the run.
    :returns: The written paths.
    """
    files = deck_writer.write_deck(job)
    if run.card_order is not None:
        rank = {note_id: rank for rank, note_id in enumerate(run.card_order, 1)}
        with AnkiPackage(job.output_path) as package:
            cards = sorted(
                (card for card in package.get_new_cards() if card["nid"] in rank),
                key=lambda card: rank[card["nid"]],
            )
            package.set_card_due(
                {card["id"]: due for due, card in enumerate(cards, 1)}
            )
            package.save()
    return files


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



