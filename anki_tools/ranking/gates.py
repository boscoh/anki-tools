"""One gate table for the core and cull modes.

The runner, the report, and the shared gates live here. Each mode supplies a
file reader, so :class:`anki_tools.ranking.coregates.CoreFiles` and
:class:`anki_tools.ranking.cullgates.CullFiles` are the two adapters across the
seam. A gate never writes a file. The gate numbers stay stable.

The report is :func:`anki_tools.sentence_pack.audit.print_audit`, so the two
modes print one table.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Protocol, Sequence

from anki_tools.ranking.assemble import output_fields
from anki_tools.ranking.coreflow import (
    OUTPUT_CORE_FIELD,
    OUTPUT_GLOSS_FIELD,
    OUTPUT_SENTENCE_FIELD,
    coverage_glosses,
)
from anki_tools.sentence_pack.audit import print_audit

#: Items of a failure message before the rest become a count.
SAMPLE = 3


class GateResult(NamedTuple):
    """Result of one gate.

    :param name: Gate label, for example ``K1``.
    :param passed: True when the gate accepts the files.
    :param message: Detail line of the gate table.
    """

    name: str
    passed: bool
    message: str


class GateFiles(Protocol):
    """Contract of the file reader that the gates read.

    A read error is cached and re-raised for each gate that needs the file.
    """

    @property
    def profile(self) -> Any:
        """Give the language hooks of the run."""
        ...

    def rows(self) -> list[dict[str, str]]:
        """Read the card CSV rows."""
        ...

    def coverage(self) -> list[dict[str, str]]:
        """Read the coverage CSV rows."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Read the manifest object."""
        ...

    def params(self) -> dict[str, Any]:
        """Read the run options of the manifest."""
        ...

    def output(self) -> dict[str, Any]:
        """Read the output APKG."""
        ...

    def notes(self) -> dict[int, list[str]]:
        """Read the note fields of the output deck, keyed by note id."""
        ...

    def fields(self) -> list[str]:
        """Read the field names of the only note type."""
        ...


@dataclass(frozen=True)
class Gate:
    """One gate value built from a ``(passed, message)`` check function.

    :param name: Gate label, for example ``K1``.
    :param check: Check that gives ``(passed, message)`` for the files.
    """

    name: str
    check: Callable[[Any], tuple[bool, str]]

    def result(self, files: GateFiles) -> GateResult:
        """Run the check and wrap the result with the gate name.

        :param files: Lazy reader of the output files.
        :returns: The result of the gate.
        """
        passed, message = self.check(files)
        return GateResult(self.name, passed, message)


def run_gates(gates: Sequence[Gate], files: GateFiles) -> list[GateResult]:
    """Run every gate and catch the error of a gate that cannot read a file.

    :param gates: Gate table of the mode.
    :param files: Lazy reader of the output files.
    :returns: One result per gate.
    """
    results: list[GateResult] = []
    for gate in gates:
        try:
            results.append(gate.result(files))
        except Exception as error:  # noqa: BLE001 - a gate reports the error
            results.append(
                GateResult(gate.name, False, f"{type(error).__name__}: {error}")
            )
    return results


def report(results: Sequence[GateResult]) -> int:
    """Print the gate table.

    :param results: Rows from :func:`run_gates`.
    :returns: The exit code of the table, 0 on a pass.
    """
    return print_audit([tuple(result) for result in results])


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings.

    :param path: Input CSV file.
    :returns: One row mapping per line.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file.

    :param path: Input JSON file.
    :raises ValueError: The file is not a JSON object.
    :returns: The JSON object.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def sample(items: Sequence[str], limit: int = SAMPLE) -> str:
    """Join a bounded sample of failure items and count the rest.

    :param items: Failure items.
    :param limit: Number of items to spell out.
    :returns: The sample message.
    """
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f" (+{len(items) - limit} more)"
    return shown


def gate_identity(files: GateFiles) -> tuple[bool, str]:
    """K9 and G10: the deck title, the note type, and the manifest agree.

    The values come from the manifest that the same writer wrote, so the gate
    reports the identity of the run and not of the profile.
    """
    params = files.params()
    source = files.manifest().get("source", {})
    output = files.output()
    title = str(params.get("name"))
    model = str(source.get("model"))
    problems: list[str] = []
    if title != model:
        problems.append(f"title {title!r} and model {model!r}")
    if output["decks"] != {title}:
        problems.append(f"decks {sorted(map(str, output['decks']))}")
    if output["note_models"] != {model}:
        problems.append(f"models {sorted(map(str, output['note_models']))}")
    if problems:
        return False, f"issues {sample(problems)}"
    return True, f"deck {title!r} and model {model!r}"


def output_model_problems(files: GateFiles) -> list[str]:
    """Check the note type shape of an output package.

    :param files: Reader of the output files.
    :returns: One problem line per fault.
    """
    output = files.output()
    model = str(files.manifest().get("source", {}).get("model"))
    problems: list[str] = []
    if output["note_models"] != {model}:
        problems.append(f"note types {sorted(map(str, output['note_models']))}")
    expected = list(output_fields(files.profile))
    if output["fields"] != expected:
        problems.append(f"fields {output['fields']}")
    templates = output["templates"].get(model, ())
    if len(templates) != 1:
        problems.append(f"{len(templates)} card template(s)")
    ords = list(output["ords"])
    if len(ords) != output["notes"] or any(ords):
        problems.append(f"{len(ords)} cards, ordinals {sorted(set(ords))}")
    return problems


def gate_output_model(files: GateFiles) -> tuple[bool, str]:
    """K21 and G16: one note type, the output fields, and one card per note."""
    problems = output_model_problems(files)
    return not problems, f"issues {sample(problems)}"


def core_word_problems(
    notes: Mapping[int, Sequence[str]],
    fields: Sequence[str],
    glosses: Mapping[str, str],
) -> tuple[list[str], int]:
    """Check the core word, the gloss, and the bold text of every note.

    :param notes: Note id to field values.
    :param fields: Field names of the output note type.
    :param glosses: Word to its coverage gloss.
    :returns: The problems, and the count of notes with a core word.
    """
    missing = [
        name
        for name in (OUTPUT_CORE_FIELD, OUTPUT_GLOSS_FIELD)
        if name not in fields
    ]
    if missing:
        return [f"the note type has no {sample(missing)} field"], 0
    word_index = fields.index(OUTPUT_CORE_FIELD)
    gloss_index = fields.index(OUTPUT_GLOSS_FIELD)
    sentence_index = (
        fields.index(OUTPUT_SENTENCE_FIELD)
        if OUTPUT_SENTENCE_FIELD in fields
        else -1
    )
    problems: list[str] = []
    filled = 0
    for note_id, values in notes.items():
        word = values[word_index].strip() if word_index < len(values) else ""
        gloss = values[gloss_index].strip() if gloss_index < len(values) else ""
        if not word:
            problems.append(f"{note_id} has no core word")
            continue
        filled += 1
        sentence = (
            values[sentence_index] if 0 <= sentence_index < len(values) else ""
        )
        if "<b>" not in sentence:
            problems.append(f"{note_id} has no bold")
        if not gloss:
            problems.append(f"{note_id} has no gloss")
        elif "," in gloss:
            problems.append(f"{note_id} holds a comma gloss")
        elif word in glosses and gloss != glosses[word]:
            problems.append(
                f"{note_id} gloss {gloss!r} is not the coverage gloss of {word}"
            )
    return problems, filled


def gate_core_word(files: GateFiles) -> tuple[bool, str]:
    """K19 and G17: every note holds a core word and its coverage gloss."""
    problems, filled = core_word_problems(
        files.notes(), files.fields(), coverage_glosses(files.coverage())
    )
    return not problems, f"{filled} core word(s), {len(problems)} bad: {sample(problems)}"
