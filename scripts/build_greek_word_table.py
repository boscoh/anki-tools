#!/usr/bin/env python
"""Build the reviewed Greek word table and its review file (stage S1).

The build never runs this script. It reads the two committed CSV files. Run this
script only when the reviewed Greek forms change.

Usage:
    uv run python scripts/build_greek_word_table.py
"""

from __future__ import annotations

import csv
import io
import subprocess

from anki_tools.sentence_pack.languages.greek import (
    MIN_ZIPF,
    SWADESH_ROWS,
    WORD_CSV_PATH,
    WORD_REVIEW_FIELDS,
    WORD_REVIEW_PATH,
    clean_form,
    load_word_table,
    module_terms,
    normalized,
    read_csv_rows,
    read_swadesh_labels,
    strip_note,
)
from wordfreq import zipf_frequency

CHOSEN = {
    2: "εσύ",
    3: "αυτός",
    7: "αυτό",
    11: "ποιος",
    36: "γυναίκα",
    37: "άνδρας",
    40: "σύζυγος",
    41: "άντρας",
    70: "πούπουλο",
    80: "πόδι",
    81: "πόδι",
    84: "φτερό",
}
SHARED_INDEX = 81


def read_raw_rows() -> list[dict[str, str]]:
    """Read the original reviewed CSV from the last commit.

    The script rewrites the tracked CSV, so it applies the review decisions to the
    committed form of the file. A missing git history falls back to the file.

    :returns: One mapping per data row of the original file.
    """
    try:
        text = subprocess.run(
            ["git", "show", f"HEAD:{WORD_CSV_PATH}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return read_csv_rows(WORD_CSV_PATH)
    return [
        {key: (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(text))
    ]


def choose_form(index: int, raw: str) -> tuple[str, str]:
    """Choose the reviewed form and gender of one CSV cell.

    :param index: Swadesh index of the row.
    :param raw: Raw cell of the current CSV.
    :returns: Reviewed form and gender.
    """
    if index in CHOSEN:
        form = CHOSEN[index]
        gender = clean_form(raw)[1] if " " in normalized(raw) else ""
        if index in (36, 37, 40, 41, 70, 80, 81, 84):
            gender = clean_form(raw)[1]
        return form, gender
    return clean_form(raw)


def module_row_forms(index: int) -> set[str]:
    """Give the cleaned module terms of one index.

    :param index: Swadesh index.
    :returns: Normalized module forms.
    """
    terms = module_terms().get(index, ())
    forms = {normalized(strip_note(term)) for term in terms}
    forms |= {clean_form(term)[0] for term in terms}
    return forms


def main() -> None:
    """Write the reviewed word CSV and the word review CSV."""
    rows = read_raw_rows()
    labels = read_swadesh_labels()
    if len(rows) != SWADESH_ROWS:
        raise SystemExit(f"expected {SWADESH_ROWS} rows, found {len(rows)}")

    table: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        raw = row.get("greek", "")
        form, gender = choose_form(index, raw)
        table.append({"english": labels[index - 1], "greek": form, "gender": gender})

    with WORD_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["english", "greek", "gender"])
        writer.writeheader()
        writer.writerows(table)

    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        raw = row.get("greek", "")
        form, gender = choose_form(index, raw)
        issues: list[tuple[str, str, str]] = []
        if "(" in raw:
            issues.append(("note", raw, f"strip the note: {strip_note(raw)}"))
        if len(normalized(raw).split()) > 1 and normalized(raw).split()[0].casefold() in {
            "ο",
            "η",
            "το",
            "οι",
            "τα",
            "τον",
            "την",
            "τη",
            "τους",
            "τις",
            "των",
        }:
            issues.append(("article", raw, f"strip the article: {form}"))
        if "/" in raw:
            issues.append(("slash", raw, f"one form per concept: {form}"))
        if form in seen and index != SHARED_INDEX:
            issues.append(("duplicate", raw, f"duplicate of index {seen[form]}: {form}"))
        if index == SHARED_INDEX:
            issues.append(("shared", raw, f"shared form with index 80: {form}"))
        seen.setdefault(form, index)
        if form not in module_row_forms(index):
            terms = "|".join(module_terms().get(index, ()))
            issues.append(("module_diff", raw, f"file form kept; module has {terms}"))
        zipf = zipf_frequency(form, "el")
        if zipf < MIN_ZIPF:
            issues.append(("zipf", raw, f"rare form zipf {zipf:.2f}; keep if it has candidates"))
        if not issues:
            continue
        for issue, current, fix in issues:
            status = "shared" if issue == "shared" else ("kept" if issue in ("module_diff", "zipf") else "fixed")
            review.append(
                {
                    "index": str(index),
                    "english": labels[index - 1],
                    "current": current,
                    "module_terms": "|".join(module_terms().get(index, ())),
                    "chosen": form,
                    "issue": issue,
                    "fix": fix,
                    "status": status,
                }
            )

    with WORD_REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORD_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)

    table_forms = load_word_table()
    print(f"rows={len(table_forms)} unique={len({form.greek for form in table_forms})}")
    print(f"review rows={len(review)}")
    print(f"wrote {WORD_CSV_PATH}")
    print(f"wrote {WORD_REVIEW_PATH}")


if __name__ == "__main__":
    main()
