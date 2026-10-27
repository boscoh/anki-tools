"""Read and write the media map of an Anki package.

The media map is the archive entry ``media``. It gives one archive file name
for each media file name. The format follows the package format:

- ``anki2`` and ``anki21``: a JSON object, for example ``{"0": "a.mp3"}``.
- ``anki21b``: a zstd frame of a protobuf ``MediaEntries`` message.

The anki21b protobuf holds no key. The entry order is the file id, so
:func:`write` renumbers the kept media files to ``0..n-1`` in the archive and
writes the protobuf entries in the same order.

The module holds the map of an open package in ``package.media_map``. A call to
:func:`add` or :func:`delete` changes that map only, so a run that adds many
files writes the map one time. :meth:`AnkiPackage.save` writes the map.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import zstandard
from pathlib import Path
from typing import Protocol

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class Package(Protocol):
    """The package attributes that this module reads."""

    temp_dir: str | None
    db_format: str | None
    media_map: dict[str, str] | None

    def touch(self) -> None:
        """Mark the package as changed."""


def read(package: Package) -> dict[str, str]:
    """Give the media map of a package.

    :param package: Open package.
    :returns: Archive file name to media file name mapping.
    """
    if package.media_map is None:
        package.media_map = _read_disk(package)
    return dict(package.media_map)


def write(package: Package, mapping: dict[str, str]) -> None:
    """Write the media map in the format of the package.

    For ``anki21b``, the kept media files take the archive names ``0..n-1``
    and their entries follow in that order. The archive holds no other media
    file.

    :param package: Open package.
    :param mapping: Archive file name to media file name mapping.
    """
    root = _root(package)
    if package.db_format == "anki21b":
        kept = _write_anki21b(root, mapping)
    else:
        _remove_unmapped(root, set(mapping))
        with open(os.path.join(root, "media"), "w", encoding="utf-8") as handle:
            json.dump(mapping, handle, ensure_ascii=False)
        kept = dict(mapping)
    package.media_map = kept
    package.touch()


def add(package: Package, path: str | Path, filename: str | None = None) -> str:
    """Copy one media file into the package under a free archive name.

    :param package: Open package.
    :param path: Path of the media file.
    :param filename: Media file name. The base name of ``path`` is the default.
    :raises FileNotFoundError: The media file is absent.
    :returns: The media file name that a note field references.
    """
    root = _root(package)
    name = filename or Path(path).name
    mapping = read(package)
    taken = {int(key) for key in mapping if key.isdigit()}
    taken |= {int(entry) for entry in os.listdir(root) if entry.isdigit()}
    file_id = max(taken, default=-1) + 1
    shutil.copy2(path, os.path.join(root, str(file_id)))
    mapping[str(file_id)] = name
    package.media_map = mapping
    package.touch()
    return name


def delete(package: Package, filename: str) -> bool:
    """Remove one media file from the package.

    :param package: Open package.
    :param filename: Media file name.
    :returns: True when the map held the name and the file went.
    """
    mapping = read(package)
    file_id = next((key for key, name in mapping.items() if name == filename), None)
    if file_id is None:
        return False
    del mapping[file_id]
    Path(_root(package), file_id).unlink(missing_ok=True)
    package.media_map = mapping
    package.touch()
    return True


def prune(package: Package, used: set[str]) -> None:
    """Delete every media file whose name is not in ``used``.

    The call reads the archive one time, removes the unused files, and writes
    the map one time.

    :param package: Open package.
    :param used: Media file names that a note still references.
    """
    mapping = read(package)
    write(package, {key: name for key, name in mapping.items() if name in used})


def _root(package: Package) -> str:
    if package.temp_dir is None:
        raise RuntimeError("The package is not open.")
    return package.temp_dir


def _read_disk(package: Package) -> dict[str, str]:
    path = os.path.join(_root(package), "media")
    if not os.path.exists(path):
        return {}
    data = Path(path).read_bytes()
    if data[:4] == ZSTD_MAGIC:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            return _parse_entries(reader.read())
    try:
        mapping = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _parse_entries(data)
    return {str(key): str(value) for key, value in mapping.items()}


def _remove_unmapped(root: str, keep: set[str]) -> None:
    for entry in os.listdir(root):
        if entry.isdigit() and entry not in keep:
            os.remove(os.path.join(root, entry))


def _write_anki21b(root: str, mapping: dict[str, str]) -> dict[str, str]:
    _remove_unmapped(root, set(mapping))
    kept: dict[str, str] = {}
    entries = b""
    for position, file_id in enumerate(sorted(mapping, key=int)):
        source = os.path.join(root, file_id)
        target = os.path.join(root, str(position))
        if source != target:
            os.replace(source, target)
        name = mapping[file_id]
        data = Path(target).read_bytes()
        entry = (
            _bytes_field(1, name.encode("utf-8"))
            + _varint_field(2, len(data))
            + _bytes_field(3, hashlib.sha1(data).digest())
        )
        entries += _bytes_field(1, entry)
        kept[str(position)] = name
    with open(os.path.join(root, "media"), "wb") as handle:
        handle.write(zstandard.ZstdCompressor().compress(entries))
    return kept


def _parse_entries(data: bytes) -> dict[str, str]:
    mapping: dict[str, str] = {}
    position = 0
    order = 0
    while position < len(data):
        tag, position = _read_varint(data, position)
        field = tag >> 3
        wire = tag & 0x07
        if wire == 2:
            length, position = _read_varint(data, position)
            payload = data[position : position + length]
            position += length
            name = _filename(payload) if field == 1 else None
            if name is not None:
                mapping[str(order)] = name
                order += 1
        elif wire == 0:
            _, position = _read_varint(data, position)
        else:
            break
    return mapping


def _filename(payload: bytes) -> str | None:
    position = 0
    while position < len(payload):
        tag, position = _read_varint(payload, position)
        field = tag >> 3
        wire = tag & 0x07
        if wire == 2:
            length, position = _read_varint(payload, position)
            value = payload[position : position + length]
            position += length
            if field == 1:
                return value.decode("utf-8")
        elif wire == 0:
            _, position = _read_varint(payload, position)
        else:
            return None
    return None


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return value, position


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return bytes([field << 3 | 2]) + _write_varint(len(payload)) + payload


def _varint_field(field: int, value: int) -> bytes:
    return bytes([field << 3]) + _write_varint(value)
