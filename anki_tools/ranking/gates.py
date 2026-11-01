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
from typing import AbstractSet, Any, Sequence

from anki_tools.gates import Gate, GateFiles, GateResult, run_gates  # noqa: F401
from anki_tools.ranking.duplicates import at_limit
from anki_tools.sentence_pack.audit import print_audit

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
