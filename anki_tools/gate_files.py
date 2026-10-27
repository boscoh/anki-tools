"""One lazy reader for the files that the gates read.

The gate table reads its files through
:class:`anki_tools.ranking.gates.GateFiles`, which documents each accessor. This
module holds the one implementation, so the cached error lives in one place.
Each mode supplies a loader table, which maps an accessor name to the callable
that reads the file.

The module sits above ``ranking`` and ``sentence_pack``, because ``ranking``
imports ``sentence_pack.audit`` and that import would cycle from
``ranking.gates``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from anki_tools.apkg.package import AnkiPackage


def read_notes(path: Path) -> dict[int, list[str]]:
    """Read the note fields of a deck, keyed by note id.

    :param path: APKG file.
    :returns: One list of field values per note.
    """
    with AnkiPackage(path) as package:
        return {
            int(row["id"]): str(row["flds"]).split("\x1f")
            for row in package.get_notes()
        }


class GateFileReader:
    """Read each gate file once and cache the value or the error.

    The accessors are the ones of
    :class:`anki_tools.ranking.gates.GateFiles`. Each mode sets ``profile`` to
    its language hooks and supplies the loader table.

    :param loaders: Table from accessor name to the callable that reads it.
    """

    profile: Any

    def __init__(self, loaders: Mapping[str, Callable[[], Any]]) -> None:
        self._loaders = dict(loaders)
        self._cache: dict[str, tuple[bool, Any]] = {}

    def _load(self, name: str) -> Any:
        """Give the value of one reader, or raise its cached error.

        :param name: Accessor name in the loader table.
        :raises Exception: The error of the first read of that name.
        :returns: The value that the loader gave.
        """
        if name not in self._cache:
            try:
                self._cache[name] = (True, self._loaders[name]())
            except Exception as error:  # noqa: BLE001 - cached for every gate
                self._cache[name] = (False, error)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    def rows(self) -> list[dict[str, str]]:
        return self._load("rows")

    def coverage(self) -> list[dict[str, str]]:
        return self._load("coverage")

    def manifest(self) -> dict[str, Any]:
        return self._load("manifest")

    def params(self) -> dict[str, Any]:
        params = self.manifest().get("params", {})
        return dict(params)

    def output(self) -> dict[str, Any]:
        return self._load("output")

    def notes(self) -> dict[int, list[str]]:
        return self._load("notes")

    def fields(self) -> list[str]:
        return self._load("fields")

    def tokens(self) -> list[list[tuple[str, str]]]:
        return self._load("tokens")

    def media(self) -> set[str]:
        return self._load("media")

    def rank_rows(self) -> list[dict[str, str]]:
        return self._load("rank")
