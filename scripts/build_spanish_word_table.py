#!/usr/bin/env python
"""Build the reviewed Spanish word table and its review file (stage S1).

The build never runs this script. It reads the two committed CSV files. Run this
script only when the reviewed Spanish forms change.

The script regenerates the table from ``Module:Swadesh/data/es``. The review
decisions are the ``CHOSEN`` forms and the ``GENDER`` values of the noun rows.
``candidate_count`` and ``tier`` are placeholders, because stage S2 fills them in
memory and the build must not write this file.

Usage:
    uv run python scripts/build_spanish_word_table.py
"""

from __future__ import annotations

import csv

from wordfreq import zipf_frequency

from anki_tools.languages.french.profile import NOUN_INDICES
from anki_tools.languages.spanish.profile import (
    MIN_ZIPF,
    SWADESH_ROWS,
    WORD_CSV_PATH,
    WORD_REVIEW_FIELDS,
    WORD_REVIEW_PATH,
    clean_term,
    form_kind,
    module_terms,
    read_swadesh_labels,
)

CHOSEN = {
    37: "hombre",
    38: "persona",
    41: "marido",
    85: "barriga",
    115: "dividir",
    117: "rascar",
    123: "acostarse",
    125: "estar de pie",
    129: "sostener",
    136: "lanzar",
    145: "congelar",
    180: "cálido",
    191: "afilado",
    192: "romo",
    201: "a",
}
REVIEWED = {
    37: ("two_terms", "hombre is the common word for man; varón is rare", "fixed"),
    38: ("shared_form", "hombre stays with index 37; persona is one token", "fixed"),
    40: ("shared_form", "mujer stays with index 36", "kept"),
    41: ("two_terms", "marido is more frequent than esposo", "fixed"),
    115: ("sense", "partir mostly means leave; dividir means split", "fixed"),
    117: ("two_terms", "rascar is more frequent than arañar", "kept"),
    123: ("sense", "acostarse is the plain lie down action", "fixed"),
    125: ("sense", "estar de pie is the stand sense; levantarse is get up", "kept"),
    129: ("sense", "tener is have; sostener is the hold sense", "fixed"),
    136: ("shared_form", "tirar stays as the base of tirar de at index 134", "fixed"),
    145: ("two_terms", "congelar without the clitic; the matcher adds congelarse", "fixed"),
    180: ("sense", "cálido is warm; caliente is hot", "kept"),
    191: ("two_terms", "afilado is more frequent than agudo for a blade", "kept"),
    192: ("two_terms", "romo is the shortest word for dull; desafilado is rarer", "kept"),
    201: ("shared_form", "en stays with index 202", "fixed"),
}
ZIPF_REVIEW = {192}
GENDER = {
    36: "f", 37: "m", 38: "f", 39: "m", 40: "f", 41: "m", 42: "f", 43: "m",
    44: "m", 45: "m", 46: "m", 47: "m", 48: "m", 49: "f", 50: "m", 51: "m",
    52: "m", 53: "m", 54: "f", 55: "f", 56: "f", 57: "f", 58: "f", 59: "f",
    60: "f", 61: "f", 62: "f", 63: "f", 64: "f", 65: "m", 66: "f", 67: "m",
    68: "m", 69: "f", 70: "f", 71: "m", 72: "f", 73: "f", 74: "m", 75: "f",
    76: "f", 77: "m", 78: "f", 79: "f", 80: "m", 81: "f", 82: "f", 83: "f",
    84: "f", 85: "f", 86: "f", 87: "m", 88: "f", 89: "m", 90: "m", 91: "m",
    147: "m", 148: "f", 149: "f", 150: "f", 151: "f", 152: "m", 153: "m",
    154: "m", 155: "f", 156: "f", 157: "f", 158: "m", 159: "f", 160: "f",
    161: "f", 162: "m", 163: "m", 164: "f", 165: "m", 166: "m", 167: "m",
    168: "f", 170: "m", 171: "f", 177: "f", 178: "m", 179: "m", 207: "m",
}  # fmt: skip
TABLE_FIELDS = [
    "swadesh_index",
    "english",
    "spanish",
    "gender",
    "variant",
    "form_kind",
    "zipf",
    "candidate_count",
    "tier",
]


def default_form(index: int) -> str:
    """Give the first cleaned term of one module row.

    :param index: Swadesh index.
    :raises SystemExit: The module row holds no term.
    :returns: The first clean module form.
    """
    terms = module_terms().get(index, ())
    if not terms:
        raise SystemExit(f"module row {index} has no term")
    return clean_term(terms[0])


def review_row(
    index: int, label: str, current: str, chosen: str, issue: str, fix: str, status: str
) -> dict[str, str]:
    """Build one review CSV row.

    :param index: Swadesh index.
    :param label: English label.
    :param current: First module form.
    :param chosen: Chosen form.
    :param issue: Issue key.
    :param fix: Fix text.
    :param status: Review status.
    :returns: Row mapping.
    """
    return {
        "index": str(index),
        "english": label,
        "current": current,
        "module_terms": "|".join(module_terms().get(index, ())),
        "chosen": chosen,
        "issue": issue,
        "fix": fix,
        "status": status,
    }


def main() -> None:
    """Write the reviewed word CSV and the word review CSV."""
    labels = read_swadesh_labels()
    if len(labels) != SWADESH_ROWS:
        raise SystemExit(f"expected {SWADESH_ROWS} labels, found {len(labels)}")
    if set(GENDER) != set(NOUN_INDICES):
        raise SystemExit("GENDER and NOUN_INDICES hold different indices")

    table: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    for index in range(1, SWADESH_ROWS + 1):
        label = labels[index - 1]
        current = default_form(index)
        chosen = CHOSEN.get(index, current)
        gender = GENDER.get(index, "")
        table.append(
            {
                "swadesh_index": str(index),
                "english": label,
                "spanish": chosen,
                "gender": gender,
                "variant": "primary",
                "form_kind": form_kind(chosen),
                "zipf": f"{zipf_frequency(chosen, 'es'):.2f}",
                "candidate_count": "0",
                "tier": "core",
            }
        )
        if index in REVIEWED:
            issue, fix, status = REVIEWED[index]
            review.append(review_row(index, label, current, chosen, issue, fix, status))
        if index in ZIPF_REVIEW:
            zipf = zipf_frequency(chosen, "es")
            review.append(
                review_row(
                    index,
                    label,
                    current,
                    chosen,
                    "zipf",
                    f"rare form, zipf {zipf:.2f}; keep because it has candidates",
                    "kept",
                )
            )
        if gender:
            review.append(
                review_row(index, label, current, chosen, "gender", gender, "fixed")
            )

    forms = [row["spanish"] for row in table]
    if len(forms) != len(set(forms)):
        duplicates = sorted({form for form in forms if forms.count(form) > 1})
        raise SystemExit(f"duplicate forms: {duplicates}")

    with WORD_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(table)

    with WORD_REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORD_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)

    low = sorted(
        (float(row["zipf"]), row["spanish"])
        for row in table
        if float(row["zipf"]) < MIN_ZIPF
    )
    print(f"rows={len(table)} unique={len(set(forms))}")
    print(f"nouns={len(NOUN_INDICES)} with_gender={sum(1 for r in table if r['gender'])}")
    print(f"review rows={len(review)}")
    print(f"zipf below {MIN_ZIPF}: {low}")


if __name__ == "__main__":
    main()
