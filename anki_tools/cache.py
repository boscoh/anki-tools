"""Download a file into the cache directory.

The download writes a temporary file first and then renames it, so an
interrupted download never leaves a short cache file.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

DOWNLOAD_TIMEOUT = 120


def download_cached_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download a file into the cache.

    :param url: Source URL.
    :param dest: Cache destination.
    :param force: Download again when the file already exists.
    :returns: Path to the cached file.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    print(f"Downloading {url}")
    try:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot download {url}.") from exc
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return dest


__all__ = ["download_cached_file"]
