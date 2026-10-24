"""Shared Swadesh word-table readers, checks, and CSV writers.

The per-language profiles read and validate their own reviewed word CSVs. The
readers and the small check helpers of that job are identical across every
language, so they live here. A language profile keeps the parts that differ:
the :class:`WordForm` shape, the per-row parsing, and the language-specific
checks such as pinyin, jyutping, script, or form kind.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

OPEN_STATUS = "open"


class TableForm(Protocol):
    """A reviewed word-table form that the shared checks read."""

    swadesh_index: int
    english: str
    variant: str


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into stripped row mappings.

    :param csv_path: CSV file to read.
    :returns: One mapping per data row, with stripped values.
    """
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_swadesh_labels(swadesh_path: Path) -> list[str]:
    """Read the Swadesh English labels in index order.

    :param swadesh_path: File with one English label per line.
    :returns: Label list, index 1 at position 0.
    """
    text = Path(swadesh_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def read_word_review(review_path: Path) -> list[dict[str, str]]:
    """Read a word review CSV.

    :param review_path: Review CSV file.
    :returns: One mapping per row, empty without a file.
    """
    if not Path(review_path).exists():
        return []
    return read_csv_rows(review_path)


def check_review_file(review_path: Path) -> list[str]:
    """Check that a word review file exists and has no open row.

    :param review_path: Review CSV file.
    :returns: Failure strings. An empty list means the file passes.
    """
    if not Path(review_path).exists():
        return [f"review file missing: {review_path}"]
    open_rows = [
        row.get("index", "")
        for row in read_csv_rows(review_path)
        if row.get("status", "").lower() == OPEN_STATUS
    ]
    if not open_rows:
        return []
    return [f"review file has {len(open_rows)} open row(s): {open_rows}"]


def check_row_count(
    table: Sequence[TableForm], *, swadesh_rows: int
) -> list[str]:
    """Check that the table covers one Swadesh index set.

    :param table: Reviewed word table.
    :param swadesh_rows: Expected number of Swadesh labels.
    :returns: Failure strings. An empty list means the table passes.
    """
    indices = sorted({form.swadesh_index for form in table})
    if len(indices) != swadesh_rows:
        return [f"row count: expected {swadesh_rows}, found {len(indices)}"]
    return []


def check_form_count(
    table: Sequence[TableForm],
    *,
    min_forms: int,
    max_forms: int,
    label: str = "form count",
) -> list[str]:
    """Check that the form count stays in a window.

    :param table: Reviewed word table.
    :param min_forms: Minimum form count.
    :param max_forms: Maximum form count.
    :param label: Label of the check in the failure text.
    :returns: Failure strings. An empty list means the table passes.
    """
    if not min_forms <= len(table) <= max_forms:
        return [f"{label}: expected {min_forms} to {max_forms}, found {len(table)}"]
    return []


def check_missing_indices(
    table: Sequence[TableForm], *, swadesh_rows: int
) -> list[str]:
    """Check that every Swadesh index has a form.

    :param table: Reviewed word table.
    :param swadesh_rows: Expected number of Swadesh labels.
    :returns: Failure strings. An empty list means the table passes.
    """
    indices = {form.swadesh_index for form in table}
    missing = sorted(set(range(1, swadesh_rows + 1)) - indices)
    if not missing:
        return []
    return [f"swadesh index: missing {missing}"]


def check_form_labels(
    table: Sequence[TableForm], labels: Sequence[str]
) -> list[str]:
    """Check each form index and Swadesh English label.

    The check runs one pass per form, so the failures keep the table order.

    :param table: Reviewed word table.
    :param labels: Swadesh labels in index order.
    :returns: Failure strings. An empty list means the table passes.
    """
    failures: list[str] = []
    for form in table:
        if not 1 <= form.swadesh_index <= len(labels):
            failures.append(f"index out of range: {form.swadesh_index}")
            continue
        label = labels[form.swadesh_index - 1]
        if form.english != label:
            failures.append(
                f"label {form.swadesh_index}: expected {label!r}, "
                f"found {form.english!r}"
            )
    return failures


def check_english_labels(
    table: Sequence[TableForm], labels: Sequence[str]
) -> list[str]:
    """Check that every form keeps its Swadesh English label.

    :param table: Reviewed word table.
    :param labels: Swadesh labels in index order.
    :returns: Failure strings. An empty list means the table passes.
    """
    failures: list[str] = []
    for form in table:
        if not 1 <= form.swadesh_index <= len(labels):
            continue
        label = labels[form.swadesh_index - 1]
        if form.english != label:
            failures.append(
                f"label {form.swadesh_index}: expected {label!r}, "
                f"found {form.english!r}"
            )
    return failures


def duplicate_forms(
    table: Sequence[TableForm], key: Callable[[TableForm], str]
) -> list[str]:
    """Give the repeated form values of a table.

    :param table: Reviewed word table.
    :param key: Function that gives the unique form string of one form.
    :returns: Sorted repeated values. An empty list means no duplicates.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for form in table:
        value = key(form)
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def form_positions(
    table: Sequence[TableForm], key: Callable[[TableForm], str]
) -> dict[str, tuple[int, int]]:
    """Give each form its Swadesh index and variant rank.

    :param table: Reviewed word table.
    :param key: Function that gives the unique form string of one form.
    :returns: Mapping from form to ``(swadesh_index, variant_rank)``.
    """
    return {
        key(form): (form.swadesh_index, 0 if form.variant == "primary" else 1)
        for form in table
    }


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]
) -> Path:
    """Write row mappings to a CSV file.

    :param path: Output CSV file.
    :param fieldnames: Column names of the file.
    :param rows: One mapping per data row.
    :returns: The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
