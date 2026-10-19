"""Tatoeba export readers and the fetch cache (stage S0).

The cache writes ``data/tatoeba/<name>.tsv`` and a ``checksums.json`` file that
holds the sha256 value of each decompressed TSV. A later run skips a file when
its sha256 still matches. The readers stream the dumps, because the English
export has about 2 million rows.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"

CHECKSUMS_NAME = "checksums.json"
CHUNK_SIZE = 1 << 20
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = 120


def export_url(lang: str, export: str) -> str:
    """Build the download URL of one Tatoeba per-language export.

    :param lang: Tatoeba language code, for example ``ell``.
    :param export: Export suffix, for example ``sentences``.
    :returns: The download URL.
    """
    return f"{TATOEBA_BASE}/{lang}/{lang}_{export}.tsv.bz2"


def link_url(lang: str, other_lang: str = "eng") -> str:
    """Build the download URL of a translation-link export.

    :param lang: Tatoeba language code, for example ``ell``.
    :param other_lang: Code of the other side of the links.
    :returns: The download URL.
    """
    return f"{TATOEBA_BASE}/{lang}/{lang}-{other_lang}_links.tsv.bz2"


def export_files(
    lang: str, *, native_audio: bool = True, other_lang: str = "eng"
) -> dict[str, str]:
    """Build the export map of one language pair.

    :param lang: Tatoeba language code.
    :param native_audio: Include the ``sentences_with_audio`` export.
    :param other_lang: Code of the translation language.
    :returns: Mapping from cache file name to download URL.
    """
    files = {
        f"{lang}_sentences": export_url(lang, "sentences"),
        f"{lang}_sentences_detailed": export_url(lang, "sentences_detailed"),
        f"{lang}-{other_lang}_links": link_url(lang, other_lang),
    }
    if native_audio:
        files[f"{lang}_sentences_with_audio"] = export_url(
            lang, "sentences_with_audio"
        )
    files[f"{other_lang}_sentences"] = export_url(other_lang, "sentences")
    return files


def sha256_file(path: Path) -> str:
    """Hash a file with sha256.

    :param path: File to hash.
    :returns: Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download a URL to a file, with a short retry loop.

    :param url: Source URL.
    :param dest: Destination path for the compressed bytes.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "anki-tools"})
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                with Path(dest).open("wb") as handle:
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
    with bz2.open(src, "rb") as reader, Path(dest).open("wb") as writer:
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
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_checksums(path: Path, checksums: dict[str, Any]) -> None:
    """Write the checksum file through a temporary file and ``os.replace``.

    :param path: Path to ``checksums.json``.
    :param checksums: Mapping from file name to its record.
    """
    text = json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    path = Path(path)
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
    if record is None or not Path(tsv_path).exists():
        return False
    return record.get("sha256") == sha256_file(tsv_path)


def _fetch_one(
    name: str, url: str, dest: Path, download_file: Callable[[str, Path], None]
) -> dict[str, Any]:
    """Download and decompress one Tatoeba file.

    :param name: Export name, used for the TSV file name.
    :param url: Source URL of the bz2 file.
    :param dest: Cache directory.
    :param download_file: Callable that writes a URL to a local file.
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
        download_file(url, compressed_path)
        rows = _decompress(compressed_path, temp_tsv_path)
        digest = sha256_file(temp_tsv_path)
        os.replace(temp_tsv_path, tsv_path)
        return {
            "url": url,
            "bytes": compressed_path.stat().st_size,
            "sha256": digest,
            "rows": rows,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "network",
        }
    finally:
        compressed_path.unlink(missing_ok=True)
        temp_tsv_path.unlink(missing_ok=True)


def fetch_tatoeba(
    files: dict[str, str],
    dest: Path = Path("data/tatoeba"),
    refresh: bool = False,
    download_file: Callable[[str, Path], None] = download,
) -> dict[str, dict[str, Any]]:
    """Fetch and cache a set of Tatoeba exports.

    A file is skipped when its TSV file exists and its sha256 matches the value
    in ``checksums.json``. The checksum file is written after each file, so an
    interrupted run keeps its progress.

    :param files: Mapping from cache file name to download URL.
    :param dest: Cache directory for the TSV files and ``checksums.json``.
    :param refresh: Ignore the cache and download every file again.
    :param download_file: Callable that writes a URL to a local file.
    :returns: Per-file record with ``rows`` and ``source`` (``cache`` or
        ``network``).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    checksums_path = dest / CHECKSUMS_NAME
    checksums = _load_checksums(checksums_path)
    results: dict[str, dict[str, Any]] = {}

    for name, url in files.items():
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
        fresh = _fetch_one(name, url, dest, download_file)
        checksums[name] = {key: value for key, value in fresh.items() if key != "source"}
        _write_checksums(checksums_path, checksums)
        results[name] = {"rows": fresh["rows"], "source": fresh["source"]}

    return results


def _iter_tsv(path: Path) -> Iterable[list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line in handle:
            yield line.rstrip("\n").split("\t")


def to_int(value: str) -> int:
    """Read an integer cell, with 0 for a bad value.

    :param value: Text of the cell.
    :returns: The integer value, or 0.
    """
    try:
        return int(value)
    except ValueError:
        return 0


def read_links(links_path: Path) -> dict[int, list[int]]:
    """Read the source to English translation links.

    :param links_path: TSV file with columns ``source_id,eng_id`` and no header.
    :returns: Mapping from source sentence id to its English sentence ids.
    """
    links: dict[int, list[int]] = defaultdict(list)
    for parts in _iter_tsv(links_path):
        if len(parts) < 2:
            continue
        links[to_int(parts[0])].append(to_int(parts[1]))
    return dict(links)


def read_sentences(sentences_path: Path) -> dict[int, str]:
    """Read the sentence texts of one language.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :returns: Mapping from sentence id to the raw sentence text.
    """
    sentences: dict[int, str] = {}
    for parts in _iter_tsv(sentences_path):
        if len(parts) < 3:
            continue
        sentences[to_int(parts[0])] = parts[2]
    return sentences


def read_authors(detailed_path: Path) -> dict[int, str]:
    """Read the author of each sentence.

    :param detailed_path: TSV file with columns ``id,lang,text,author,...``.
    :returns: Mapping from sentence id to the author name.
    """
    authors: dict[int, str] = {}
    for parts in _iter_tsv(detailed_path):
        if len(parts) < 4:
            continue
        authors[to_int(parts[0])] = "" if parts[3] == "\\N" else parts[3]
    return authors


def read_audio(detailed_path: Path) -> dict[int, tuple[int, str]]:
    """Read the native audio id and license of each sentence.

    :param detailed_path: TSV file with columns
        ``sentence_id,audio_id,author,license,attribution``.
    :returns: Mapping from sentence id to ``(audio_id, license)``.
    """
    audio: dict[int, tuple[int, str]] = {}
    for parts in _iter_tsv(detailed_path):
        if len(parts) < 2:
            continue
        license_name = parts[3] if len(parts) > 3 else ""
        audio[to_int(parts[0])] = (to_int(parts[1]), license_name)
    return audio


def read_english_sentences(
    sentences_path: Path, wanted_ids: set[int]
) -> dict[int, str]:
    """Stream the English dump and keep the wanted rows.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :param wanted_ids: English sentence ids to keep.
    :returns: Mapping from English sentence id to its text.
    """
    if not wanted_ids:
        return {}
    found: dict[int, str] = {}
    with Path(sentences_path).open(encoding="utf-8", newline="") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            sentence_id = to_int(parts[0])
            if sentence_id not in wanted_ids:
                continue
            found[sentence_id] = parts[2]
            if len(found) == len(wanted_ids):
                break
    return found
