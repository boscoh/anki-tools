"""Mandarin word frequency cache from the wordfreq ``zh`` list.

wordfreq holds one tokenized list for Mandarin with the language code ``zh``
and the script ``Hans``. The build takes the top 60,000 entries, keeps the rows
that are Han only, and cuts the list to 50,000 rows. The fallback source is the
FrequencyWords ``zh_cn`` and ``zh_tw`` pair, which the build reads only on
demand.

Chinese has no inflection and the package holds no Mandarin lemmatizer, so the
``lemma`` column repeats the ``word`` column and the rank is the 1-based rank of
that form. Several rows never share a lemma, unlike the Spanish cache. The ranks
are unique and contiguous from 1 to 50,000.

wordfreq stores a log frequency per word and no corpus size, so the ``count``
column is 0 in the primary build and ``pm`` carries the frequency. The fallback
build writes real counts. wordfreq holds no part-of-speech data, so ``pos`` is
empty.

The rank follows the order of the wordfreq list, not the order of ``pm``. The
two orders agree over the first 4,750 rows and differ in the tail, where the
quantized values repeat and jump. Use ``rank`` as the order of the list.

See ``specs/core-deck-all-languages.md`` sections 8.1 and 8.2.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from wordfreq import top_n_list, word_frequency

from anki_tools.languages.mandarin.characters import is_chinese_char

FREQUENCY_PATH = Path("data/mandarin_frequency.tsv")

WORDFREQ_LANG = "zh"
SOURCE_NAME = "wordfreq zh"
SOURCE_VERSION = "wordfreq 3.1.1"
SOURCE_SHA256 = "e079d7aaff1cfb92874850cb3ffbc7b611af939d9d671c1cbda932d968467cdd"
LICENSE = "Apache-2.0"
SCRIPT = "Hans"
MAX_ROWS = 50_000
SOURCE_ROWS = 60_000


FUNCTION_WORDS = ("的", "是", "在", "我", "你")
COMMON_CONTENT_WORDS = ("我们", "朋友", "今天")
GREETING_WORDS = ("你好", "谢谢")

_COLUMNS = ("word", "lemma", "count", "pm", "rank", "pos", "source", "license")


@dataclass(frozen=True)
class MandarinFrequencyRow:
    """One row of the cached Mandarin frequency list.

    :param word: Form, as it appears in the deck.
    :param lemma: Lemma of the form. Equal to the word, because Chinese has no
        inflection and the package holds no Mandarin lemmatizer.
    :param count: Token count of the form, 0 when the source holds no count.
    :param pm: Frequency per million tokens.
    :param rank: 1-based rank of the lemma.
    :param pos: Part of speech, empty when the source holds none.
    :param source: Short source name.
    :param license: SPDX license identifier.
    """

    word: str
    lemma: str
    count: int
    pm: float
    rank: int
    pos: str
    source: str
    license: str


def load_frequency_zh(
    path: Path = FREQUENCY_PATH,
) -> list[MandarinFrequencyRow]:
    """Read the cached Mandarin frequency list.

    :param path: TSV file with the columns ``word``, ``lemma``, ``count``,
        ``pm``, ``rank``, ``pos``, ``source``, and ``license``.
    :raises RuntimeError: The file cannot be read.
    :returns: Rows in file order.
    """
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read the Mandarin frequency list {path}.") from exc
    rows: list[MandarinFrequencyRow] = []
    with handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            word = (record.get("word") or "").strip()
            if not word:
                continue
            rows.append(
                MandarinFrequencyRow(
                    word=word,
                    lemma=(record.get("lemma") or word).strip(),
                    count=_int_cell(record.get("count")),
                    pm=_float_cell(record.get("pm")),
                    rank=_int_cell(record.get("rank")),
                    pos=(record.get("pos") or "").strip(),
                    source=(record.get("source") or "").strip(),
                    license=(record.get("license") or "").strip(),
                )
            )
    return rows


def frequency_ranks_zh(
    rows: Sequence[MandarinFrequencyRow],
) -> dict[str, int]:
    """Map each lemma to its 1-based rank.

    :param rows: Frequency rows.
    :returns: Lemma to rank. The most frequent lemma has rank 1. Rows above
        :data:`MAX_ROWS` keep no rank.
    """
    return {row.lemma: row.rank for row in rows if row.rank <= MAX_ROWS}


def is_han_only(word: str) -> bool:
    """Check if every character of a word is Han.

    :param word: Candidate form.
    :returns: True when the word is not empty and holds Han characters only.
    """
    return bool(word) and all(is_chinese_char(char) for char in word)


def build_frequency_zh(
    *,
    rows: int = MAX_ROWS,
    counts: Mapping[str, int] | None = None,
    source: str = SOURCE_NAME,
    license: str = LICENSE,
) -> list[MandarinFrequencyRow]:
    """Build the frequency list from the wordfreq ``zh`` list.

    :param rows: Number of most frequent forms to keep.
    :param counts: Optional token counts per form. A form with no count takes 0,
        because wordfreq holds proportions and no token count.
    :param source: Value of the ``source`` cell.
    :param license: Value of the ``license`` cell.
    :returns: Rows in rank order.
    """
    forms = [
        word for word in top_n_list(WORDFREQ_LANG, SOURCE_ROWS) if is_han_only(word)
    ][:rows]
    return [
        MandarinFrequencyRow(
            word=word,
            lemma=word,
            count=(counts or {}).get(word, 0),
            pm=round(word_frequency(word, WORDFREQ_LANG) * 1_000_000, 6),
            rank=rank,
            pos="",
            source=source,
            license=license,
        )
        for rank, word in enumerate(forms, 1)
    ]




def write_frequency_tsv(rows: Iterable[MandarinFrequencyRow], path: Path) -> None:
    """Write the frequency list as a TSV file sorted by rank.

    :param rows: Frequency rows.
    :param path: Output file.
    :raises RuntimeError: The file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write the frequency list {path}.") from exc
    with handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for row in sorted(rows, key=lambda item: item.rank):
            writer.writerow(
                [
                    row.word,
                    row.lemma,
                    row.count,
                    f"{row.pm:.6f}",
                    row.rank,
                    row.pos,
                    row.source,
                    row.license,
                ]
            )


def check_frequency(
    rows: Sequence[MandarinFrequencyRow],
) -> list[tuple[str, bool, str]]:
    """Run gates Z1, Z2, and Z3 of the spec on the list.

    :param rows: Frequency rows in rank order.
    :returns: A list of gate name, pass flag, and message.
    """
    ranks = frequency_ranks_zh(rows)
    results: list[tuple[str, bool, str]] = []
    contiguous = [row.rank for row in rows] == list(range(1, len(rows) + 1))
    results.append(
        (
            "Z1",
            len(rows) == MAX_ROWS and contiguous,
            f"{len(rows)} rows, ranks contiguous: {contiguous}",
        )
    )
    for gate, words, limit in (
        ("Z2", FUNCTION_WORDS, 20),
        ("Z3", COMMON_CONTENT_WORDS, 1000),
        ("Z3", GREETING_WORDS, 3000),
    ):
        found = {word: ranks.get(word) for word in words}
        ok = all(rank is not None and rank <= limit for rank in found.values())
        results.append((gate, ok, f"ranks within {limit}: {found}"))
    return results


def _int_cell(text: str | None) -> int:
    try:
        return int((text or "").strip() or 0)
    except ValueError:
        return 0


def _float_cell(text: str | None) -> float:
    try:
        return float((text or "").strip() or 0.0)
    except ValueError:
        return 0.0


__all__ = [
    "FREQUENCY_PATH",
    "LICENSE",
    "MAX_ROWS",
    "SOURCE_NAME",
    "SOURCE_ROWS",
    "SOURCE_SHA256",
    "SOURCE_VERSION",
    "WORDFREQ_LANG",
    "MandarinFrequencyRow",
    "build_frequency_zh",
    "check_frequency",
    "frequency_ranks_zh",
    "is_han_only",
    "load_frequency_zh",
    "write_frequency_tsv",
]
