"""Tests for the Spanish frequency cache (spec 8.2 and 11 step S3).

The tests read ``data/spanish_frequency.tsv``, which is a local cache file and
is not tracked in git.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from anki_tools.languages.spanish.frequency import (
    ES_FULL_SHA256,
    FREQUENCY_PATH,
    LICENSE,
    SOURCE_NAME,
    SpanishFrequencyRow,
    frequency_ranks_es,
    load_frequency_es,
)

PINNED_SHA256 = "ad19aa90a468dc140d05e98eaeba47da3c23adc69453123257489708396cc278"
SOURCE_SHA256 = "b4794c425f3500c21a0ff9367175e36795a978c928a36670e9ad6ae206fb7728"

COLUMNS = ("word", "lemma", "count", "pm", "rank", "pos", "source", "license")

pytestmark = pytest.mark.skipif(
    not FREQUENCY_PATH.exists(),
    reason="the Spanish frequency cache is absent",
)


def _sha256(path: Path) -> str:
    """Give the sha256 digest of a file.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_tsv() -> None:
    """The cache has the eight columns and 50,000 data rows."""
    with FREQUENCY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0].keys()) == COLUMNS
    assert len(rows) == 50_000


def test_known_lemma_ranks() -> None:
    """Common lemmas hold the expected aggregated ranks."""
    ranks = frequency_ranks_es(load_frequency_es())
    assert ranks["el"] == 1
    assert ranks["de"] == 2
    assert ranks["que"] == 3
    assert ranks["ser"] == 4
    assert ranks["tener"] == 16
    assert ranks["ir"] == 21
    assert ranks["hacer"] == 25


def test_top_100_spot_check() -> None:
    """The top 100 ranks are contiguous and hold common content lemmas."""
    ranks = frequency_ranks_es(load_frequency_es())
    top = {lemma: rank for lemma, rank in ranks.items() if rank <= 100}
    assert len(top) == 100
    assert sorted(top.values()) == list(range(1, 101))
    for lemma, expected in (
        ("hombre", 72),
        ("casa", 74),
        ("tiempo", 79),
        ("ver", 42),
        ("vida", 96),
    ):
        assert ranks[lemma] == expected


def test_loader_fields() -> None:
    """The loader returns the new fields for every row."""
    rows = load_frequency_es()
    assert len(rows) == 50_000
    assert rows[0] == SpanishFrequencyRow(
        word="la",
        lemma="el",
        count=9125471,
        pm=21558.390418,
        rank=1,
        pos="DET",
        source=SOURCE_NAME,
        license=LICENSE,
    )
    assert all(row.source == SOURCE_NAME for row in rows)
    assert all(row.license == LICENSE for row in rows)


def test_pin_sha256() -> None:
    """The derived cache and the source file keep their pinned digests."""
    assert _sha256(FREQUENCY_PATH) == PINNED_SHA256
    assert ES_FULL_SHA256 == SOURCE_SHA256
