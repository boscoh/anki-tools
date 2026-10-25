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
    _count_tatoeba,
    cull_frequency_rows,
    frequency_ranks_es,
    load_frequency_es,
)
from anki_tools.ranking.cullflow import blend_pm, spoken_mass

PINNED_SHA256 = "117b36a109abd2d3fed610835c534847e43b878a12424f9604f7def17522a107"
SOURCE_SHA256 = "b4794c425f3500c21a0ff9367175e36795a978c928a36670e9ad6ae206fb7728"

COLUMNS = (
    "word",
    "lemma",
    "count",
    "pm",
    "rank",
    "pos",
    "source",
    "license",
    "tatoeba_count",
    "tatoeba_pm",
    "written_pm",
    "zipf",
)

needs_cache = pytest.mark.skipif(
    not FREQUENCY_PATH.exists(),
    reason="the Spanish frequency cache is absent",
)


def _sha256(path: Path) -> str:
    """Give the sha256 digest of a file.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_cache
def test_read_tsv() -> None:
    """The cache has the eight columns and 50,000 data rows."""
    with FREQUENCY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0].keys()) == COLUMNS
    assert len(rows) == 50_000


@needs_cache
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


@needs_cache
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


@needs_cache
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
        tatoeba_count=229646,
        tatoeba_pm=150100.624401,
        written_pm=28183.829313,
        zipf=7.45,
    )
    assert all(row.source == SOURCE_NAME for row in rows)
    assert all(row.license == LICENSE for row in rows)


@needs_cache
def test_pin_sha256() -> None:
    """The derived cache and the source file keep their pinned digests."""
    assert _sha256(FREQUENCY_PATH) == PINNED_SHA256
    assert ES_FULL_SHA256 == SOURCE_SHA256


def test_loader_builds_the_cache_when_the_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader builds and writes the list once, then reads it back."""
    from anki_tools.languages.spanish import core

    target = tmp_path / "spanish_frequency.tsv"
    built = [
        SpanishFrequencyRow(
            word="la",
            lemma="el",
            count=9,
            pm=1.0,
            rank=1,
            pos="DET",
            source=SOURCE_NAME,
            license=LICENSE,
            tatoeba_count=0,
            tatoeba_pm=0.0,
            written_pm=0.0,
            zipf=0.0,
        )
    ]
    monkeypatch.setattr(core, "FREQUENCY_PATH", target)
    monkeypatch.setattr(core, "build_frequency_es", lambda **_: list(built))

    assert core.load_frequency() == built
    assert target.exists()

    def again(**_: object) -> list[SpanishFrequencyRow]:
        raise AssertionError("the loader rebuilt a file that exists")

    monkeypatch.setattr(core, "build_frequency_es", again)
    assert core.load_frequency() == built


def test_count_tatoeba_counts_lemmas(tmp_path: Path) -> None:
    """The Tatoeba pass counts the lemmas and the content tokens."""
    path = tmp_path / "spa_sentences.tsv"
    path.write_text(
        "1\tspa\tLa casa es grande.\n"
        "2\tspa\tMi casa y el tiempo.\n"
        "3\tspa\tEl tiempo pasa.\n"
        "4\tspa\n",
        encoding="utf-8",
    )
    counts, content_tokens = _count_tatoeba(path)
    assert counts["casa"] == 2
    assert counts["tiempo"] == 2
    assert counts["el"] == 3
    assert counts["y"] == 1
    assert content_tokens > 0


@needs_cache
def test_blend_pm_uses_the_spoken_term() -> None:
    """A lemma with a Tatoeba count blends to a different value."""
    rows = [
        row
        for row in cull_frequency_rows(load_frequency_es())
        if row.tatoeba_count >= 5
    ]
    assert rows
    row = rows[0]
    assert blend_pm(row, 0.0) != blend_pm(row, 1.0)


@needs_cache
def test_spoken_mass_is_positive() -> None:
    """A small lemma set covers a positive share of the content tokens."""
    rows = cull_frequency_rows(load_frequency_es())
    assert spoken_mass(rows, {"casa", "tiempo"}) > 0
