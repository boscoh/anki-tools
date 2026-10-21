"""Mandarin caller of the shared Tatoeba fetch cache (stage S0).

The shared module holds the download, decompress, and checksum logic. This module
names the Mandarin and English exports and keeps the module-level patch points
that the tests use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anki_tools.sentence_pack import tatoeba

TATOEBA_BASE = tatoeba.TATOEBA_BASE
TATOEBA_FILES: dict[str, str] = tatoeba.export_files("cmn")
CHECKSUMS_NAME = tatoeba.CHECKSUMS_NAME
CHUNK_SIZE = tatoeba.CHUNK_SIZE
DOWNLOAD_ATTEMPTS = tatoeba.DOWNLOAD_ATTEMPTS
DOWNLOAD_TIMEOUT = tatoeba.DOWNLOAD_TIMEOUT

_download = tatoeba.download
_sha256_file = tatoeba.sha256_file
_decompress = tatoeba._decompress
_load_checksums = tatoeba._load_checksums
_write_checksums = tatoeba._write_checksums
_is_cached = tatoeba._is_cached
_fetch_one = tatoeba._fetch_one


def fetch_tatoeba(
    dest: Path = Path("data/tatoeba"), refresh: bool = False
) -> dict[str, dict[str, Any]]:
    """Fetch the Tatoeba exports for the Mandarin Swadesh sentence pack.

    A file is skipped when its TSV file exists and its sha256 matches the value
    in ``checksums.json``. The checksum file is written after each file, so an
    interrupted run keeps its progress.

    :param dest: Cache directory for the TSV files and ``checksums.json``.
    :param refresh: Ignore the cache and download every file again.
    :returns: Per-file record with ``rows`` and ``source`` (``cache`` or
        ``network``).
    """
    return tatoeba.fetch_tatoeba(
        TATOEBA_FILES, Path(dest), refresh=refresh, download_file=_download
    )
