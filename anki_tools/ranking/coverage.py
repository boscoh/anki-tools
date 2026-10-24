"""CSV writer that the coverage tables share."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    """Write CSV rows with a fixed column order and a newline after each row.

    :param path: Output CSV file.
    :param columns: Column names in file order.
    :param rows: Rows that hold one value for each column.
    :raises RuntimeError: The file does not open for writing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = target.open("w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {target}.") from exc
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
