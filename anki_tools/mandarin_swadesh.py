"""Word table for the Mandarin Swadesh sentence pack.

Stage S1 of the Mandarin Swadesh sentence pack reads the reviewed
``vocab/english-mandarin.csv`` file, expands each slash-separated row into one
form, and checks the result against ``vocab/swadesh.txt``. The module reads the
review files and never writes them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import opencc
from wordfreq import zipf_frequency

DEFAULT_CSV_PATH = Path("vocab/english-mandarin.csv")
DEFAULT_SWADESH_PATH = Path("vocab/swadesh.txt")
DEFAULT_OVERRIDES_PATH = Path("vocab/mandarin_pinyin_overrides.csv")
DEFAULT_REVIEW_PATH = Path("vocab/swadesh_review.csv")

SWADESH_ROWS = 207
MIN_FORMS = 205
MAX_FORMS = 220
MIN_ZIPF = 3.0
OPEN_STATUS = "open"

REVIEW_FIELDS = [
    "lang",
    "index",
    "english",
    "current",
    "canonical",
    "issue",
    "fix",
    "status",
]
OVERRIDE_FIELDS = ["mandarin", "pinyin", "reason"]

_converter: opencc.OpenCC | None = None


@dataclass
class WordForm:
    """One reviewed Mandarin form of a Swadesh concept.

    :param swadesh_index: CSV row order, 1-based.
    :param english: Swadesh English label of the row.
    :param mandarin: Simplified form, the unique key of the row.
    :param pinyin: Tone-marked pinyin of the form.
    :param traditional: Traditional form from OpenCC ``s2t``.
    :param variant: ``primary`` for the first form of a row, else ``alternate``.
    :param zipf: ``wordfreq`` Zipf value of the form for ``zh``.
    :param candidate_count: Sentence count after the S2 filters.
    :param tier: ``core`` or ``scaffold``.
    """

    swadesh_index: int
    english: str
    mandarin: str
    pinyin: str
    traditional: str
    variant: str
    zipf: float
    candidate_count: int = 0
    tier: str = "core"


def _traditional_converter() -> opencc.OpenCC:
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC("s2t")
    return _converter


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of stripped row mappings.

    :param csv_path: CSV file to read.
    :returns: One mapping per data row, with stripped values.
    """
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_swadesh_labels(swadesh_path: Path = DEFAULT_SWADESH_PATH) -> list[str]:
    """Read the Swadesh English labels in index order.

    :param swadesh_path: File with one English label per line.
    :returns: Label list, index 1 at position 0.
    """
    text = swadesh_path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def read_pinyin_overrides(overrides_path: Path = DEFAULT_OVERRIDES_PATH) -> dict[str, str]:
    """Read the manual pinyin overrides.

    :param overrides_path: CSV file with columns ``mandarin,pinyin,reason``.
    :returns: Mapping from form to its reviewed pinyin. Missing file gives an
        empty mapping.
    """
    if not overrides_path.exists():
        return {}
    overrides: dict[str, str] = {}
    for row in _read_csv_rows(overrides_path):
        form = row.get("mandarin", "")
        pinyin = row.get("pinyin", "")
        if form and pinyin:
            overrides[form] = pinyin
    return overrides


def split_forms(mandarin: str, index: int) -> list[str]:
    """Split a slash-separated ``mandarin`` cell into single forms.

    :param mandarin: Raw cell value, for example ``老/旧``.
    :param index: Swadesh index, used in the error message.
    :raises ValueError: A form is empty, or the cell is empty.
    :returns: One form per slash-separated part.
    """
    forms = [part.strip() for part in mandarin.split("/")]
    if not forms or any(not form for form in forms):
        raise ValueError(f"row {index}: bad form cell {mandarin!r}")
    return forms


def align_pinyin(forms: list[str], transliteration: str, index: int) -> list[str]:
    """Assign the syllables of a transliteration cell to the forms.

    Each form takes as many syllables as it has Han characters, in order.

    :param forms: Forms of the row, in file order.
    :param transliteration: Space-separated pinyin of the whole row.
    :param index: Swadesh index, used in the error message.
    :raises ValueError: The syllable count does not equal the Han count.
    :returns: One pinyin string per form.
    """
    syllables = transliteration.split()
    han_count = sum(len(form) for form in forms)
    if han_count != len(syllables):
        raise ValueError(
            f"row {index}: {han_count} Han characters but {len(syllables)} "
            f"syllables in {transliteration!r}"
        )
    aligned: list[str] = []
    start = 0
    for form in forms:
        stop = start + len(form)
        aligned.append(" ".join(syllables[start:stop]))
        start = stop
    return aligned


def _row_forms(
    rows: list[dict[str, str]], overrides: dict[str, str]
) -> list[tuple[int, str, str, str, str]]:
    """Expand CSV rows into ``(index, english, form, pinyin, variant)`` tuples.

    :param rows: CSV rows in file order.
    :param overrides: Mapping from form to a manual pinyin value.
    :raises ValueError: A row does not align.
    :returns: One tuple per form.
    """
    expanded: list[tuple[int, str, str, str, str]] = []
    for index, row in enumerate(rows, start=1):
        english = row.get("english", "")
        forms = split_forms(row.get("mandarin", ""), index)
        pinyins = align_pinyin(forms, row.get("transliteration", ""), index)
        for position, (form, pinyin) in enumerate(zip(forms, pinyins)):
            variant = "primary" if position == 0 else "alternate"
            expanded.append(
                (index, english, form, overrides.get(form, pinyin), variant)
            )
    return expanded


def load_word_table(
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> list[WordForm]:
    """Load the Swadesh word table from the reviewed CSV.

    :param csv_path: Reviewed word CSV with columns
        ``english,mandarin,transliteration``.
    :param swadesh_path: File with one English label per line.
    :param overrides_path: CSV file with manual pinyin overrides.
    :raises ValueError: The CSV row count differs from the label count, or a
        row does not align.
    :returns: One :class:`WordForm` per expanded form.
    """
    rows = _read_csv_rows(csv_path)
    labels = read_swadesh_labels(swadesh_path)
    if len(rows) != len(labels):
        raise ValueError(
            f"{csv_path}: {len(rows)} rows but {len(labels)} Swadesh labels"
        )
    converter = _traditional_converter()
    table: list[WordForm] = []
    for index, english, form, pinyin, variant in _row_forms(
        rows, read_pinyin_overrides(overrides_path)
    ):
        table.append(
            WordForm(
                swadesh_index=index,
                english=english,
                mandarin=form,
                pinyin=pinyin,
                traditional=converter.convert(form),
                variant=variant,
                zipf=zipf_frequency(form, "zh"),
            )
        )
    return table


def _check_size(table: list[WordForm]) -> list[str]:
    failures: list[str] = []
    indices = sorted({form.swadesh_index for form in table})
    if len(indices) != SWADESH_ROWS:
        failures.append(
            f"row count: expected {SWADESH_ROWS}, found {len(indices)}"
        )
    if not MIN_FORMS <= len(table) <= MAX_FORMS:
        failures.append(
            f"form count: expected {MIN_FORMS} to {MAX_FORMS}, "
            f"found {len(table)}"
        )
    return failures


def _check_indices(table: list[WordForm]) -> list[str]:
    indices = {form.swadesh_index for form in table}
    expected = set(range(1, SWADESH_ROWS + 1))
    missing = sorted(expected - indices)
    extra = sorted(indices - expected)
    if not missing and not extra:
        return []
    return [f"swadesh index: missing {missing}, extra {extra}"]


def _check_unique(table: list[WordForm]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for form in table:
        if form.mandarin in seen:
            duplicates.add(form.mandarin)
        seen.add(form.mandarin)
    if not duplicates:
        return []
    return [f"duplicate form: {sorted(duplicates)}"]


def _check_labels(table: list[WordForm], swadesh_path: Path) -> list[str]:
    labels = read_swadesh_labels(swadesh_path)
    failures: list[str] = []
    if len(labels) != SWADESH_ROWS:
        failures.append(
            f"swadesh labels: expected {SWADESH_ROWS}, found {len(labels)}"
        )
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


def _check_pinyin(table: list[WordForm], csv_path: Path, overrides_path: Path) -> list[str]:
    try:
        rows = _read_csv_rows(csv_path)
        expected = {
            form: pinyin
            for _, _, form, pinyin, _ in _row_forms(rows, {})
        }
    except (OSError, ValueError) as error:
        return [f"pinyin source: {error}"]
    overrides = read_pinyin_overrides(overrides_path)
    failures: list[str] = []
    for form in table:
        source = expected.get(form.mandarin)
        if source is None:
            failures.append(f"pinyin {form.mandarin}: form is not in {csv_path}")
            continue
        if _normalize(form.pinyin) == source:
            continue
        if overrides.get(form.mandarin) == _normalize(form.pinyin):
            continue
        failures.append(
            f"pinyin {form.mandarin}: expected {source!r}, "
            f"found {form.pinyin!r}"
        )
    return failures


def _check_review_file(review_path: Path) -> list[str]:
    if not review_path.exists():
        return [f"review file missing: {review_path}"]
    open_rows = [
        row
        for row in _read_csv_rows(review_path)
        if row.get("status", "").lower() == OPEN_STATUS
    ]
    if not open_rows:
        return []
    indices = [row.get("index", "") for row in open_rows]
    return [f"review file has {len(open_rows)} open row(s): {indices}"]


def check_word_table(
    table: list[WordForm],
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
) -> list[str]:
    """Check the word table against the reviewed files.

    :param table: Word table from :func:`load_word_table`.
    :param csv_path: Reviewed word CSV, the pinyin source.
    :param swadesh_path: File with one English label per line.
    :param overrides_path: CSV file with manual pinyin overrides.
    :param review_path: Review file that must exist and have no open row.
    :returns: Failure strings. An empty list means every check passes.
    """
    failures = _check_size(table)
    failures.extend(_check_indices(table))
    failures.extend(_check_unique(table))
    failures.extend(_check_labels(table, swadesh_path))
    failures.extend(_check_pinyin(table, csv_path, overrides_path))
    failures.extend(_check_review_file(review_path))
    return failures


def word_table_report(table: list[WordForm]) -> dict[str, Any]:
    """Summarize the word table.

    :param table: Word table from :func:`load_word_table`.
    :returns: Counts of words and forms, the low Zipf forms, and the forms
        with no candidate sentence. ``uncovered`` stays empty until S2.
    """
    forms = [form.mandarin for form in table]
    low = sorted(
        ((form.mandarin, round(form.zipf, 2)) for form in table if form.zipf < MIN_ZIPF),
        key=lambda item: item[1],
    )
    return {
        "words": len({form.swadesh_index for form in table}),
        "forms": len(table),
        "unique_forms": len(set(forms)),
        "primary_forms": sum(1 for form in table if form.variant == "primary"),
        "alternate_forms": sum(1 for form in table if form.variant == "alternate"),
        "core_forms": sum(1 for form in table if form.tier == "core"),
        "zipf_below_3": len(low),
        "low_zipf_forms": low,
        "uncovered": [],
    }


def print_word_table_report(report: dict[str, Any]) -> None:
    """Print a word table report.

    :param report: Mapping from :func:`word_table_report`.
    """
    print(
        f"words={report['words']} forms={report['forms']} "
        f"unique={report['unique_forms']} "
        f"primary={report['primary_forms']} "
        f"alternate={report['alternate_forms']}"
    )
    print(
        f"zipf<{MIN_ZIPF}: {report['zipf_below_3']} "
        f"uncovered={len(report['uncovered'])}"
    )
    low_forms = report.get("low_zipf_forms", [])
    if low_forms:
        print("low zipf forms: " + ", ".join(f"{form} {score}" for form, score in low_forms))
