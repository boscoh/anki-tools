#!/usr/bin/env python
"""Build the reviewed French word table and its review file (stage S1).

The build never runs this script. It reads the two committed CSV files. Run this
script only when the reviewed French forms change.

The script regenerates the table from ``Module:Swadesh/data/fr``. The review
decisions are the ``CHOSEN`` forms of spec section 6 and the ``GENDER`` values of
the noun rows. ``candidate_count`` and ``tier`` are placeholders, because stage
S2 fills them in memory and the build must not write this file.

Usage:
    uv run python scripts/build_french_word_table.py
"""

from __future__ import annotations

import csv

from wordfreq import zipf_frequency

from anki_tools.sentence_pack.languages.french import (
    MIN_ZIPF,
    NOUN_INDICES,
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
    2: "tu",
    38: "être humain",
    40: "épouse",
    41: "mari",
    86: "intestins",
    179: "année",
    183: "nouveau",
}
REVIEWED = {
    2: ("shared_form", "vous stays with index 5", "kept"),
    36: ("shared_form", "femme stays with index 36", "kept"),
    37: ("shared_form", "homme stays with index 37", "kept"),
    38: ("duplicate_form", "être humain avoids a duplicate of index 37", "fixed"),
    40: ("shared_form", "femme stays with index 36", "fixed"),
    41: ("two_terms", "mari is more frequent than époux", "kept"),
    86: ("two_terms", "entrailles has 0 candidates", "fixed"),
    179: ("two_terms", "année is the general sense of year", "fixed"),
    183: ("two_terms", "neuf means brand new", "kept"),
}
GENDER = {
    36: "f",
    37: "m",
    38: "m",
    39: "m",
    40: "f",
    41: "m",
    42: "f",
    43: "m",
    44: "m",
    45: "m",
    46: "m",
    47: "m",
    48: "m",
    49: "m",
    50: "m",
    51: "m",
    52: "f",
    53: "m",
    54: "m",
    55: "f",
    56: "f",
    57: "f",
    58: "f",
    59: "f",
    60: "f",
    61: "f",
    62: "f",
    63: "f",
    64: "m",
    65: "m",
    66: "f",
    67: "m",
    68: "f",
    69: "f",
    70: "f",
    71: "m",
    72: "f",
    73: "f",
    74: "m",
    75: "m",
    76: "f",
    77: "f",
    78: "f",
    79: "m",
    80: "m",
    81: "f",
    82: "m",
    83: "f",
    84: "f",
    85: "m",
    86: "m",
    87: "m",
    88: "m",
    89: "m",
    90: "m",
    91: "m",
    147: "m",
    148: "f",
    149: "f",
    150: "f",
    151: "f",
    152: "f",
    153: "m",
    154: "f",
    155: "m",
    156: "f",
    157: "m",
    158: "f",
    159: "f",
    160: "m",
    161: "m",
    162: "m",
    163: "m",
    164: "f",
    165: "f",
    166: "f",
    167: "m",
    168: "f",
    170: "f",
    171: "f",
    177: "f",
    178: "m",
    179: "f",
    207: "m",
}
ZIPF_REVIEW = {192}
TABLE_FIELDS = [
    "swadesh_index",
    "english",
    "french",
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
        current = default_form(index)
        chosen = CHOSEN.get(index, current)
        gender = GENDER.get(index, "")
        table.append(
            {
                "swadesh_index": str(index),
                "english": labels[index - 1],
                "french": chosen,
                "gender": gender,
                "variant": "primary",
                "form_kind": form_kind(chosen),
                "zipf": f"{zipf_frequency(chosen, 'fr'):.2f}",
                "candidate_count": "0",
                "tier": "core",
            }
        )
        terms = "|".join(module_terms().get(index, ()))
        if index in REVIEWED:
            issue, fix, status = REVIEWED[index]
            review.append(
                {
                    "index": str(index),
                    "english": labels[index - 1],
                    "current": current,
                    "module_terms": terms,
                    "chosen": chosen,
                    "issue": issue,
                    "fix": fix,
                    "status": status,
                }
            )
        if index in ZIPF_REVIEW:
            review.append(
                {
                    "index": str(index),
                    "english": labels[index - 1],
                    "current": current,
                    "module_terms": terms,
                    "chosen": chosen,
                    "issue": "zipf",
                    "fix": (
                        f"rare form, zipf {zipf_frequency(chosen, 'fr'):.2f}; "
                        "keep because it has candidates"
                    ),
                    "status": "kept",
                }
            )
        if gender:
            review.append(
                {
                    "index": str(index),
                    "english": labels[index - 1],
                    "current": current,
                    "module_terms": terms,
                    "chosen": chosen,
                    "issue": "gender",
                    "fix": gender,
                    "status": "fixed",
                }
            )

    forms = [row["french"] for row in table]
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
        (float(row["zipf"]), row["french"]) for row in table if float(row["zipf"]) < MIN_ZIPF
    )
    print(f"rows={len(table)} unique={len(set(forms))}")
    print(f"nouns={len(NOUN_INDICES)} with_gender={sum(1 for row in table if row['gender'])}")
    print(f"review rows={len(review)}")
    print(f"zipf below {MIN_ZIPF}: {low}")
    print(f"wrote {WORD_CSV_PATH}")
    print(f"wrote {WORD_REVIEW_PATH}")


if __name__ == "__main__":
    main()
