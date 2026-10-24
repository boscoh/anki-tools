"""Spanish word frequency cache from the FrequencyWords ``es_full`` list.

The cache has one row per surface form. The lemma column comes from the spaCy
``es_core_news_sm`` model. The rank is the rank of the lemma after the counts
of its surface forms are summed. Only the top 50,000 rows are kept.

See ``specs/core-deck-all-languages.md`` sections 8.1, 8.2, and 11 step S3.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FREQUENCY_PATH = Path("data/spanish_frequency.tsv")

ES_FULL_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "525f9b560de45753a5ea01069454e72e9aa541c6/content/2018/es/es_full.txt"
)
ES_FULL_SHA256 = "b4794c425f3500c21a0ff9367175e36795a978c928a36670e9ad6ae206fb7728"
SOURCE_VERSION = "525f9b560de45753a5ea01069454e72e9aa541c6"
SOURCE_NAME = "FrequencyWords es_full"
LICENSE = "CC BY-SA 4.0"
TOKENIZER = "es_core_news_sm"
MAX_ROWS = 50_000

_COLUMNS = ("word", "lemma", "count", "pm", "rank", "pos", "source", "license")


@dataclass(frozen=True)
class SpanishFrequencyRow:
    """One row of the cached Spanish frequency list.

    :param word: Surface form, as it appears in the deck.
    :param lemma: Lemma of the form from spaCy.
    :param count: Token count of the surface form.
    :param pm: Count per million tokens.
    :param rank: 1-based rank of the lemma.
    :param pos: spaCy part of speech of the surface form.
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


def load_frequency_es(
    path: Path = FREQUENCY_PATH,
) -> list[SpanishFrequencyRow]:
    """Read the cached Spanish frequency list.

    :param path: TSV file with the columns ``word``, ``lemma``, ``count``,
        ``pm``, ``rank``, ``pos``, ``source``, and ``license``.
    :returns: Rows in file order.
    """
    rows: list[SpanishFrequencyRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                SpanishFrequencyRow(
                    word=record["word"],
                    lemma=record["lemma"],
                    count=int(record["count"]),
                    pm=float(record["pm"]),
                    rank=int(record["rank"]),
                    pos=record["pos"],
                    source=record["source"],
                    license=record["license"],
                )
            )
    return rows


def frequency_ranks_es(
    rows: Sequence[SpanishFrequencyRow],
) -> dict[str, int]:
    """Map each lemma to its 1-based rank.

    :param rows: Frequency rows.
    :returns: Lemma to rank. The most frequent lemma has rank 1.
    """
    return {row.lemma: row.rank for row in rows}


__all__ = [
    "ES_FULL_SHA256",
    "ES_FULL_URL",
    "FREQUENCY_PATH",
    "LICENSE",
    "MAX_ROWS",
    "SOURCE_NAME",
    "SOURCE_VERSION",
    "TOKENIZER",
    "SpanishFrequencyRow",
    "frequency_ranks_es",
    "load_frequency_es",
]
