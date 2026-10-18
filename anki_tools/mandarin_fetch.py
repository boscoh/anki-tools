"""Fetch and cache the Tatoeba exports used by the Mandarin Swadesh pack.

Stage S0 of the Mandarin Swadesh sentence pack downloads five Tatoeba
exports, decompresses each one to ``data/tatoeba/<name>.tsv``, and records
a checksum file so that later runs skip the network.

The sha256 value in ``checksums.json`` is the hash of the decompressed
TSV file. ``bytes`` is the size of the compressed download.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"

TATOEBA_FILES: dict[str, str] = {
    "cmn_sentences": f"{TATOEBA_BASE}/cmn/cmn_sentences.tsv.bz2",
    "cmn_sentences_detailed": f"{TATOEBA_BASE}/cmn/cmn_sentences_detailed.tsv.bz2",
    "cmn-eng_links": f"{TATOEBA_BASE}/cmn/cmn-eng_links.tsv.bz2",
    "cmn_sentences_with_audio": (
        f"{TATOEBA_BASE}/cmn/cmn_sentences_with_audio.tsv.bz2"
    ),
    "eng_sentences": f"{TATOEBA_BASE}/eng/eng_sentences.tsv.bz2",
}

CHECKSUMS_NAME = "checksums.json"
CHUNK_SIZE = 1 << 20
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = 120


def _sha256_file(path: Path) -> str:
    """Hash a file with sha256.

    :param path: File to hash.
    :returns: Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download a URL to a file, with a short retry loop.

    Tests monkeypatch this function to avoid the network.

    :param url: Source URL.
    :param dest: Destination path for the compressed bytes.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "anki-tools"})
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                with dest.open("wb") as handle:
                    while chunk := response.read(CHUNK_SIZE):
                        handle.write(chunk)
            return
        except OSError as error:
            last_error = error
            if attempt + 1 < DOWNLOAD_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"download failed: {url}") from last_error


def _decompress(src: Path, dest: Path) -> int:
    """Decompress a bz2 file and count the data rows.

    :param src: Compressed bz2 file.
    :param dest: Destination path for the TSV file.
    :returns: Number of newline-terminated rows in the TSV file.
    """
    rows = 0
    tail = b""
    with bz2.open(src, "rb") as reader, dest.open("wb") as writer:
        while chunk := reader.read(CHUNK_SIZE):
            writer.write(chunk)
            rows += chunk.count(b"\n")
            tail = chunk[-1:]
    if tail and tail != b"\n":
        rows += 1
    return rows


def _load_checksums(path: Path) -> dict[str, Any]:
    """Read the checksum file.

    :param path: Path to ``checksums.json``.
    :returns: The stored mapping, or an empty mapping.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_checksums(path: Path, checksums: dict[str, Any]) -> None:
    """Write the checksum file through a temporary file and ``os.replace``.

    :param path: Path to ``checksums.json``.
    :param checksums: Mapping from file name to its record.
    """
    text = json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _is_cached(tsv_path: Path, record: dict[str, Any] | None) -> bool:
    """Report if a TSV file matches its recorded sha256.

    :param tsv_path: Path to the decompressed TSV file.
    :param record: Cached record for the file.
    :returns: True when the file exists and the sha256 matches.
    """
    if record is None or not tsv_path.exists():
        return False
    return record.get("sha256") == _sha256_file(tsv_path)


def _fetch_one(name: str, url: str, dest: Path) -> dict[str, Any]:
    """Download and decompress one Tatoeba file.

    :param name: Export name, used for the TSV file name.
    :param url: Source URL of the bz2 file.
    :param dest: Cache directory.
    :returns: A fresh cache record with ``source`` set to ``network``.
    """
    tsv_path = dest / f"{name}.tsv"
    with tempfile.NamedTemporaryFile(
        "wb", dir=dest, delete=False, suffix=".bz2"
    ) as handle:
        compressed_path = Path(handle.name)
    with tempfile.NamedTemporaryFile(
        "wb", dir=dest, delete=False, suffix=".tsv"
    ) as handle:
        temp_tsv_path = Path(handle.name)
    try:
        _download(url, compressed_path)
        rows = _decompress(compressed_path, temp_tsv_path)
        sha256 = _sha256_file(temp_tsv_path)
        os.replace(temp_tsv_path, tsv_path)
        return {
            "url": url,
            "bytes": compressed_path.stat().st_size,
            "sha256": sha256,
            "rows": rows,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "network",
        }
    finally:
        compressed_path.unlink(missing_ok=True)
        temp_tsv_path.unlink(missing_ok=True)


def fetch_tatoeba(
    dest: Path = Path("data/tatoeba"), refresh: bool = False
) -> dict[str, dict[str, Any]]:
    """Fetch the Tatoeba exports for the Mandarin Swadesh sentence pack.

    A file is skipped when its TSV file exists and its sha256 matches the
    value in ``checksums.json``. The checksum file is written after each
    file, so an interrupted run keeps its progress.

    :param dest: Cache directory for the TSV files and ``checksums.json``.
    :param refresh: Ignore the cache and download every file again.
    :returns: Per-file record with ``rows`` and ``source`` (``cache`` or
        ``network``).
    """
    dest.mkdir(parents=True, exist_ok=True)
    checksums_path = dest / CHECKSUMS_NAME
    checksums = _load_checksums(checksums_path)
    results: dict[str, dict[str, Any]] = {}

    for name, url in TATOEBA_FILES.items():
        tsv_path = dest / f"{name}.tsv"
        record = checksums.get(name)
        rows = record.get("rows") if isinstance(record, dict) else None
        if (
            not refresh
            and isinstance(record, dict)
            and isinstance(rows, int)
            and _is_cached(tsv_path, record)
        ):
            results[name] = {"rows": rows, "source": "cache"}
            continue
        fresh = _fetch_one(name, url, dest)
        checksums[name] = {
            key: value for key, value in fresh.items() if key != "source"
        }
        _write_checksums(checksums_path, checksums)
        results[name] = {"rows": fresh["rows"], "source": fresh["source"]}

    return results
