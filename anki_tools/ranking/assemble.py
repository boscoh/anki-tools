"""One assembler for the deck, the side files, the manifest, and the report.

The core selection and the cull selection are the two adapters across this
seam. Each adapter states the values, the tables, and the report lines of its
mode. This module holds the write order, the manifest schema, and the report
layout once. The identity module names every artifact.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.reorder import reorder_deck
from anki_tools.sentence_pack.audit import sha256_value

#: Columns of the rank CSV that ``reorder_deck`` reads.
RANK_COLUMNS = ("rank", "note_id", "sentence", "kept")


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
    :param shape: Stated shape of the output package.
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
    shape: OutputShape
    values: Mapping[int, Sequence[str]]
    card_table: SideTable
    coverage_table: SideTable
    manifest: ManifestValues
    report: Sequence[tuple[str | None, str]]
    rank_rows: Sequence[tuple[int, int, str, int]] | None = None
    reserve: Sequence[Any] = ()
    reserve_pack_path: Path | None = None
    reserve_values: Callable[[Any], Sequence[str]] | None = None


@dataclass(frozen=True)
class RunFiles:
    """The four files that the assembler writes.

    :param output_path: Output APKG file.
    :param card_csv: Card CSV file.
    :param coverage_csv: Coverage CSV file.
    :param manifest_path: Manifest JSON file.
    :param rank_csv: Rank CSV file of a run that reorders, else None.
    """

    output_path: Path
    card_csv: Path
    coverage_csv: Path
    manifest_path: Path
    rank_csv: Path | None = None


def write_run(run: DeckRun) -> RunFiles:
    """Write the deck, the side files, the manifest, and the report of one run.

    :param run: Selected cards and stated values of the run.
    :returns: The written paths.
    """
    identity = run.identity
    output = identity.output_path
    card_csv = identity.side(run.card_table.suffix)
    coverage_csv = identity.side(run.coverage_table.suffix)
    manifest_path = identity.side("manifest.json")
    rank_csv = identity.side("rank.csv") if run.rank_rows is not None else None
    package_args: dict[str, Any] = {
        "reserve": run.reserve,
        "reserve_pack_path": run.reserve_pack_path,
        "reserve_values": run.reserve_values,
    }
    if rank_csv is not None:
        write_rank_csv(run.rank_rows or (), rank_csv)
        temporary = output.with_name(output.stem + ".tmp.apkg")
        write_package(run.input_path, temporary, run.shape, run.values, **package_args)
        reorder_deck(str(temporary), str(output), str(rank_csv))
        temporary.unlink(missing_ok=True)
    else:
        write_package(run.input_path, output, run.shape, run.values, **package_args)
    write_csv(run.card_table.columns, run.card_table.rows, card_csv)
    write_csv(run.coverage_table.columns, run.coverage_table.rows, coverage_csv)
    write_manifest(
        manifest_path,
        run,
        files={
            run.card_table.label: sha256_value(card_csv),
            run.coverage_table.label: sha256_value(coverage_csv),
            "input_apkg": sha256_value(run.input_path),
        },
    )
    files = RunFiles(output, card_csv, coverage_csv, manifest_path, rank_csv)
    print_report(run.report, files)
    return files


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


def write_csv(
    columns: Sequence[str], rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    """Write a CSV file with a fixed header.

    :param columns: Column names, in file order.
    :param rows: One row mapping per line.
    :param path: Output CSV file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def manifest_payload(run: DeckRun, files: Mapping[str, str]) -> dict[str, Any]:
    """Give the manifest payload of a run, in its stated key order.

    :param run: Selected cards and stated values of the run.
    :param files: ``sha256:`` values of the output files.
    :returns: The manifest object.
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
        "files": dict(files),
    }
    if values.licenses is not None:
        payload["licenses"] = dict(values.licenses)
    payload["kept_order"] = values.kept_order
    if values.reserve_pack:
        payload["reserve_pack"] = values.reserve_pack
        payload["swadesh_exceptions"] = list(values.swadesh_exceptions)
    return payload


def write_manifest(path: Path, run: DeckRun, files: Mapping[str, str]) -> None:
    """Write the manifest JSON of a run.

    :param path: Output JSON file.
    :param run: Selected cards and stated values of the run.
    :param files: ``sha256:`` values of the output files.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest_payload(run, files), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_report(lines: Sequence[tuple[str | None, str]], files: RunFiles) -> None:
    """Print the summary of a run and the files that it wrote.

    :param lines: Report lines of the mode.
    :param files: The written paths.
    """
    for label, value in lines:
        if label is None:
            print(value)
        else:
            print(f"{label + ':':<21}{value}")
    print()
    print(f"wrote {files.output_path}")
    print(f"wrote {files.card_csv}")
    print(f"wrote {files.coverage_csv}")
    print(f"wrote {files.manifest_path}")
