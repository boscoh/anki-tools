"""One gate table for the core and cull modes.

The gate value, the runner, and the file reader contract live in
:mod:`anki_tools.gates`. The report and the shared checks live here. Each mode
supplies a file reader, so :class:`anki_tools.ranking.coregates.CoreFiles` and
:class:`anki_tools.ranking.cullgates.CullFiles` are the two adapters across the
seam. A gate never writes a file. The gate numbers stay stable.

The report is :func:`anki_tools.sentence_pack.audit.print_audit`, so the two
modes print one table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Sequence

from anki_tools.gates import Gate, GateFiles, GateResult, run_gates  # noqa: F401
from anki_tools.ranking.assemble import output_fields
from anki_tools.ranking.coreflow import (
    OUTPUT_CORE_FIELD,
    OUTPUT_GLOSS_FIELD,
    OUTPUT_SENTENCE_FIELD,
    coverage_glosses,
)
from anki_tools.ranking.duplicates import at_limit
from anki_tools.sentence_pack.audit import print_audit, sha256_value

#: Items of a failure message before the rest become a count.
SAMPLE = 3


def report(results: Sequence[GateResult], title: str = "") -> int:
    """Print the gate table.

    :param results: Rows from :func:`run_gates`.
    :param title: Flow name of the table.
    :returns: The exit code of the table, 0 on a pass.
    """
    return print_audit(results, title)


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
    if output["kind"] != 0:
        problems.append(f"note type kind {output['kind']}")
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
    """K21 and G16: one standard note type, the output fields, one card per note."""
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


def gate_counts(files: GateFiles, check_media: bool = False) -> tuple[bool, str]:
    """G8 and K8: the note count matches the CSV and the manifest.

    :param files: Reader of the output files.
    :param check_media: Also require the media set to equal the card audio.
    """
    output = files.output()
    cards = int(files.manifest().get("counts", {}).get("cards", 0))
    kept = [row for row in files.rows() if row.get("kept") == "1"]
    notes = int(output["notes"])
    ok = notes == len(kept) == cards
    message = f"{notes} notes, {len(kept)} kept rows, manifest {cards}"
    if not check_media:
        return ok, message
    audio = {card.audio for card in output["cards"] if card.audio}
    media = set(output["media"])
    ok = ok and media == audio
    return ok, f"{message}, media {len(media)} of {len(audio)}"


def gate_manifest_hashes(
    files: GateFiles, file_keys: Mapping[str, Path]
) -> tuple[bool, str]:
    """G13 and K12: the manifest hashes match the side files.

    :param files: Reader of the output files.
    :param file_keys: Manifest key to side file, in the check order.
    """
    recorded = files.manifest().get("files", {})
    problems: list[str] = []
    for key, path in file_keys.items():
        actual = sha256_value(path)
        if recorded.get(key) != actual:
            problems.append(f"{key} manifest {recorded.get(key)} != {actual}")
    if problems:
        return False, f"{len(problems)} hash(es) differ: {sample(problems)}"
    return True, f"{len(file_keys)} file hashes match"


def gate_similarity(
    files: GateFiles, sets: Sequence[AbstractSet[str]], limit: float
) -> tuple[bool, str]:
    """G6 and K6: no kept card pair reaches the lemma Jaccard limit.

    :param files: Reader of the output files.
    :param sets: Content lemmas of every card, in card order.
    :param limit: Lemma Jaccard limit of a duplicate pair.
    """
    cards = files.output()["cards"]
    seen: dict[str, list[int]] = {}
    problems: list[str] = []
    for index, lemmas in enumerate(sets):
        others = {other for lemma in lemmas for other in seen.get(lemma, ())}
        if any(at_limit(lemmas, sets[other], limit) for other in others):
            problems.append(cards[index].sentence[:40])
        for lemma in lemmas:
            seen.setdefault(lemma, []).append(index)
    if problems:
        return False, f"{len(problems)} duplicate card(s): {sample(problems)}"
    return True, f"{len(sets)} kept cards below the Jaccard limit {limit}"
