"""Tests for the Tatoeba fetch cache."""

import bz2
import json
from pathlib import Path

import pytest

from anki_tools.languages.mandarin import stages as fetch

SAMPLE_TSV = b"1\tcmn\t\xe4\xbd\xa0\xe5\xa5\xbd\n2\tcmn\t\xe5\x86\x8d\xe8\xa7\x81\n"
SAMPLE_URL = "https://example.invalid/cmn_sentences.tsv.bz2"


class Downloader:
    """Record download calls and write a small bz2 file."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        dest.write_bytes(bz2.compress(SAMPLE_TSV))


@pytest.fixture
def downloader(monkeypatch: pytest.MonkeyPatch) -> Downloader:
    fake = Downloader()
    monkeypatch.setattr(fetch, "_download", fake)
    monkeypatch.setattr(
        fetch, "TATOEBA_FILES", {"cmn_sentences": SAMPLE_URL}
    )
    return fake


def test_first_run_writes_tsv_and_checksums(
    tmp_path: Path, downloader: Downloader
) -> None:
    dest = tmp_path / "tatoeba"

    results = fetch.fetch_tatoeba(dest)

    assert results["cmn_sentences"] == {"rows": 2, "source": "network"}
    assert downloader.urls == [SAMPLE_URL]
    assert (dest / "cmn_sentences.tsv").read_bytes() == SAMPLE_TSV

    checksums = json.loads((dest / "checksums.json").read_text(encoding="utf-8"))
    record = checksums["cmn_sentences"]
    assert record["url"] == SAMPLE_URL
    assert record["rows"] == 2
    assert record["bytes"] == len(bz2.compress(SAMPLE_TSV))
    assert record["sha256"] == fetch._sha256_file(dest / "cmn_sentences.tsv")
    assert record["downloaded_at"]
    assert "Z" in record["downloaded_at"] or "+00:00" in record["downloaded_at"]


def test_second_run_uses_cache(tmp_path: Path, downloader: Downloader) -> None:
    dest = tmp_path / "tatoeba"
    fetch.fetch_tatoeba(dest)

    results = fetch.fetch_tatoeba(dest)

    assert results["cmn_sentences"] == {"rows": 2, "source": "cache"}
    assert len(downloader.urls) == 1


def test_refresh_downloads_again(tmp_path: Path, downloader: Downloader) -> None:
    dest = tmp_path / "tatoeba"
    fetch.fetch_tatoeba(dest)

    results = fetch.fetch_tatoeba(dest, refresh=True)

    assert results["cmn_sentences"]["source"] == "network"
    assert downloader.urls == [SAMPLE_URL, SAMPLE_URL]


def test_bad_sha256_triggers_download(tmp_path: Path, downloader: Downloader) -> None:
    dest = tmp_path / "tatoeba"
    fetch.fetch_tatoeba(dest)

    checksums_path = dest / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["cmn_sentences"]["sha256"] = "0" * 64
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    results = fetch.fetch_tatoeba(dest)

    assert results["cmn_sentences"]["source"] == "network"
    assert len(downloader.urls) == 2
