"""Tests for the Mandarin frequency cache (spec 8.1 and 8.2).

The tests build the cache from wordfreq, write it to a temporary file, and read
it back. The test of the real cache reads ``data/mandarin_frequency.tsv``, which
is a local cache file and is not tracked in git.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from anki_tools.languages.mandarin.frequency import (
    FREQUENCY_PATH,
    LICENSE,
    MAX_ROWS,
    SOURCE_NAME,
    SOURCE_SHA256,
    MandarinFrequencyRow,
    build_frequency_zh,
    check_frequency,
    frequency_ranks_zh,
    load_frequency_zh,
    write_frequency_tsv,
)

PINNED_SHA256 = "11956288a22bfda3e74dc12eac7cde0954521a56bed93213aea6ec76598eb9a3"
WORDFREQ_SHA256 = "e079d7aaff1cfb92874850cb3ffbc7b611af939d9d671c1cbda932d968467cdd"

COLUMNS = ("word", "lemma", "count", "pm", "rank", "pos", "source", "license")


def _sha256(path: Path) -> str:
    """Give the sha256 digest of a file.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built_rows() -> list[MandarinFrequencyRow]:
    """Build the frequency list from wordfreq once for the module.

    :returns: Rows in rank order.
    """
    return build_frequency_zh()


def test_write_and_read_roundtrip(
    built_rows: list[MandarinFrequencyRow],
    tmp_path: Path,
) -> None:
    """The writer and the reader agree, and the file keeps its rank order."""
    path = tmp_path / "mandarin_frequency.tsv"
    write_frequency_tsv(built_rows, path)
    loaded = load_frequency_zh(path)
    assert len(loaded) == MAX_ROWS
    assert loaded == built_rows
    assert [row.rank for row in loaded] == list(range(1, MAX_ROWS + 1))


def test_write_columns(tmp_path: Path) -> None:
    """The file has the eight columns of spec 8.2 in order."""
    rows = [
        MandarinFrequencyRow(
            word="的",
            lemma="的",
            count=0,
            pm=61700.0,
            rank=1,
            pos="",
            source=SOURCE_NAME,
            license=LICENSE,
        )
    ]
    path = tmp_path / "one.tsv"
    write_frequency_tsv(rows, path)
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(records[0].keys()) == COLUMNS
    assert records[0]["pm"] == "61700.000000"


def test_rank_order(built_rows: list[MandarinFrequencyRow]) -> None:
    """Ranks are contiguous, and known forms hold known ranks."""
    ranks = frequency_ranks_zh(built_rows)
    assert ranks["的"] == 1
    assert ranks["是"] == 2
    assert ranks["在"] == 3
    assert ranks["我"] == 5
    assert ranks["我们"] == 25
    assert sorted(ranks.values()) == list(range(1, MAX_ROWS + 1))
    head = built_rows[:1000]
    assert all(earlier.pm >= later.pm for earlier, later in zip(head, head[1:]))


def test_row_fields(built_rows: list[MandarinFrequencyRow]) -> None:
    """Every row repeats its form as the lemma and names the source and license."""
    assert all(row.lemma == row.word for row in built_rows)
    assert all(row.count == 0 for row in built_rows)
    assert all(row.pos == "" for row in built_rows)
    assert all(row.source == SOURCE_NAME for row in built_rows)
    assert all(row.license == LICENSE for row in built_rows)


def test_check_frequency(built_rows: list[MandarinFrequencyRow]) -> None:
    """The gates Z1, Z2, and Z3 pass on the wordfreq build."""
    assert all(passed for _, passed, _ in check_frequency(built_rows))


def test_source_sha256() -> None:
    """The wordfreq data file keeps its pinned digest."""
    assert SOURCE_SHA256 == WORDFREQ_SHA256


@pytest.mark.skipif(
    not FREQUENCY_PATH.exists(),
    reason="the Mandarin frequency cache is absent",
)
def test_cached_file() -> None:
    """The cached file has 50,000 rows and the pinned digest."""
    with FREQUENCY_PATH.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(records[0].keys()) == COLUMNS
    assert len(records) == MAX_ROWS
    assert _sha256(FREQUENCY_PATH) == PINNED_SHA256


def test_loader_builds_the_cache_when_the_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader builds, gates, and writes the list once, then reads it back."""
    from anki_tools.languages.mandarin import core

    target = tmp_path / "mandarin_frequency.tsv"
    built = [
        MandarinFrequencyRow(
            word="的",
            lemma="的",
            count=0,
            pm=1.0,
            rank=1,
            pos="",
            source=SOURCE_NAME,
            license=LICENSE,
        )
    ]
    monkeypatch.setattr(core, "FREQUENCY_PATH", target)
    monkeypatch.setattr(core, "check_frequency", lambda rows: [("Z1", True, "ok")])
    monkeypatch.setattr(core, "build_frequency_zh", lambda **_: list(built))

    assert core.load_frequency() == built
    assert target.exists()

    def again(**_: object) -> list[MandarinFrequencyRow]:
        raise AssertionError("the loader rebuilt a file that exists")

    monkeypatch.setattr(core, "build_frequency_zh", again)
    assert core.load_frequency() == built
