"""Contract tests for the media map of an APKG package.

The map is JSON for ``anki2`` and a zstd protobuf for ``anki21b``. The tests
write a package, read it again, and check the archive names, so a writer that
the reader cannot read fails here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from anki_tools.apkg.package import AnkiPackage
from tests.deck_fixture import (
    ANKI21B_NOTE_TYPE,
    MP3_STUB,
    build_anki21b_deck,
    build_contract_deck,
    write_mp3s,
)

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
AUDIO_NAMES = ["a0.mp3", "a1.mp3", "a2.mp3"]


def _media_bytes(apkg: Path) -> bytes:
    """Give the raw bytes of the ``media`` entry of a package."""
    with zipfile.ZipFile(apkg) as archive:
        return archive.read("media")


def _archive_names(apkg: Path) -> set[str]:
    """Give the archive entry names of a package."""
    with zipfile.ZipFile(apkg) as archive:
        return set(archive.namelist())


def _anki2_deck(tmp_path: Path) -> Path:
    directory = tmp_path / "deck"
    directory.mkdir()
    return build_contract_deck(directory)


def _anki21b_audio_deck(tmp_path: Path) -> Path:
    """Write an anki21b package with three media files and two note sounds."""
    deck = build_anki21b_deck(tmp_path / "audio21b.apkg")
    with AnkiPackage(deck) as package:
        for path in write_mp3s(tmp_path, AUDIO_NAMES):
            package.add_media_file(path)
        package.conn.executemany(
            "UPDATE notes SET flds = ? WHERE id = ?",
            [
                ("Hola.\x1fEnglish.\x1f[sound:a0.mp3]", 1),
                ("Adios.\x1fEnglish.\x1f[sound:a1.mp3]", 2),
            ],
        )
        package.conn.commit()
        package.touch()
        package.save()
    return deck


def test_anki21b_round_trip_writes_a_zstd_protobuf(tmp_path: Path) -> None:
    deck = build_anki21b_deck(tmp_path / "deck.apkg")
    first, second = write_mp3s(tmp_path, AUDIO_NAMES[:2])

    with AnkiPackage(deck) as package:
        assert package.add_media_file(first) == "a0.mp3"
        assert package.add_media_file(second) == "a1.mp3"
        package.save()

    with AnkiPackage(deck) as package:
        mapping = package.get_media_mapping()

    assert mapping == {"0": "a0.mp3", "1": "a1.mp3"}
    assert _media_bytes(deck)[:4] == ZSTD_MAGIC
    assert {name for name in _archive_names(deck) if name.isdigit()} == {"0", "1"}


def test_anki2_round_trip_stays_json(tmp_path: Path) -> None:
    deck = _anki2_deck(tmp_path)
    (extra,) = write_mp3s(tmp_path, ["extra.mp3"])

    with AnkiPackage(deck) as package:
        assert package.db_format == "anki2"
        before = package.get_media_mapping()
        assert package.add_media_file(extra, "extra.mp3") == "extra.mp3"
        package.save()

    with AnkiPackage(deck) as package:
        mapping = package.get_media_mapping()

    payload = _media_bytes(deck)
    assert payload[:1] == b"{"
    assert json.loads(payload) == mapping
    assert mapping[str(len(before))] == "extra.mp3"
    assert {key: mapping[key] for key in before} == before


def test_prune_keeps_one_file_and_renumbers_it(tmp_path: Path) -> None:
    deck = build_anki21b_deck(tmp_path / "deck.apkg")

    with AnkiPackage(deck) as package:
        for path in write_mp3s(tmp_path, AUDIO_NAMES):
            package.add_media_file(path)
        package.prune_media({"a1.mp3"})
        package.save()

    with AnkiPackage(deck) as package:
        mapping = package.get_media_mapping()
    with zipfile.ZipFile(deck) as archive:
        names = {name for name in archive.namelist() if name.isdigit()}
        kept = archive.read("0")

    assert mapping == {"0": "a1.mp3"}
    assert names == {"0"}
    assert kept == MP3_STUB


def test_the_deck_flow_prunes_and_adds_on_anki21b(tmp_path: Path) -> None:
    """The cull order: delete the cards, add the reserve media, prune one time."""
    deck = _anki21b_audio_deck(tmp_path)
    (reserve,) = write_mp3s(tmp_path, ["reserve.mp3"])

    with AnkiPackage(deck) as package:
        doomed = [int(row["id"]) for row in package.get_cards() if int(row["nid"]) == 2]
        package.delete_cards(doomed, cleanup_audio=False)
        package.add_media_file(reserve, "reserve.mp3")
        package.add_note_and_card(
            ["Hola.", "English.", "[sound:reserve.mp3]"],
            ANKI21B_NOTE_TYPE,
            1,
            " shared ",
            "reserve-guid",
            0,
            "Hola.",
        )
        package.prune_media(package.used_media())
        package.save()

    with AnkiPackage(deck) as package:
        mapping = package.get_media_mapping()
        used = package.used_media()

    assert used == {"a0.mp3", "reserve.mp3"}
    assert mapping == {"0": "a0.mp3", "1": "reserve.mp3"}
    assert _media_bytes(deck)[:4] == ZSTD_MAGIC
    assert {name for name in _archive_names(deck) if name.isdigit()} == {"0", "1"}
