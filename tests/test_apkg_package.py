"""Contract tests for the APKG package API on a real deck.

The fixture is one Chinese deck of 576 notes and 538 media files. The
module-scoped fixture opens the deck one time, so the tests share the unzip.
Each test that changes the deck copies it first, so the tests stay independent.

The deck references 4 audio files that it does not hold. That gap comes from
the source deck, so the tests here check the mapping and not the tags.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterator

import pytest

from anki_tools.apkg.package import AnkiPackage

APKG_PATH = Path("tests/A_Course_in_Contemporary_Chinese_1.apkg")
SOUND = re.compile(r"\[sound:([^\]]+)\]")
AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg")


def _note_audio(package: AnkiPackage) -> dict[int, set[str]]:
    """Give the audio names of each note, by note id."""
    return {
        int(row["id"]): set(SOUND.findall(row["flds"]))
        for row in package.get_notes()
    }


def _audio_users(audio: dict[int, set[str]]) -> Counter[str]:
    """Count the notes that use each audio file."""
    users: Counter[str] = Counter()
    for names in audio.values():
        users.update(names)
    return users


def _card_of(package: AnkiPackage, note_id: int) -> int:
    """Give the card id of one note."""
    return next(
        int(row["id"]) for row in package.get_cards() if int(row["nid"]) == note_id
    )


@pytest.fixture(scope="module")
def package() -> Iterator[AnkiPackage]:
    with AnkiPackage(APKG_PATH) as open_package:
        yield open_package


@pytest.fixture
def editable(tmp_path: Path) -> Iterator[AnkiPackage]:
    target = tmp_path / "deck.apkg"
    shutil.copy(APKG_PATH, target)
    with AnkiPackage(target) as open_package:
        yield open_package


def test_every_media_file_is_referenced_by_a_note(package: AnkiPackage) -> None:
    """A media file with no note is dead weight in the package."""
    media = set(package.get_media_mapping().values())
    used = set().union(*_note_audio(package).values())

    assert media - used == set()


def test_the_extraction_writes_every_mapped_audio_file(
    package: AnkiPackage, tmp_path: Path
) -> None:
    extracted = package.extract_audio_files(tmp_path, audio_only=True)
    media = set(package.get_media_mapping().values())

    assert set(extracted) == {name for name in media if name.endswith(AUDIO_SUFFIXES)}
    assert extracted
    for path in extracted.values():
        assert Path(path).stat().st_size > 0


def test_delete_card_removes_the_note_and_its_exclusive_audio(
    editable: AnkiPackage,
) -> None:
    audio = _note_audio(editable)
    users = _audio_users(audio)
    note_id = next(
        note_id
        for note_id, names in audio.items()
        if names and all(users[name] == 1 for name in names)
    )
    names = audio[note_id]

    result = editable.delete_card(_card_of(editable, note_id))

    assert result["card_deleted"] is True
    assert result["note_deleted"] is True
    assert set(result["audio_files_deleted"]) == names
    assert all(not editable.is_audio_file_used(name) for name in names)


def test_delete_card_keeps_a_shared_audio_file(editable: AnkiPackage) -> None:
    audio = _note_audio(editable)
    users = _audio_users(audio)
    note_id = next(
        note_id
        for note_id, names in audio.items()
        if any(users[name] > 1 for name in names)
    )
    shared = {name for name in audio[note_id] if users[name] > 1}

    result = editable.delete_card(_card_of(editable, note_id))

    assert result["note_deleted"] is True
    assert set(result["audio_files_deleted"]).isdisjoint(shared)
    assert all(editable.is_audio_file_used(name) for name in shared)


def test_delete_cards_removes_the_unused_media(editable: AnkiPackage) -> None:
    """The cull uses the bulk call, so its counts and its cleanup must hold."""
    audio = _note_audio(editable)
    users = _audio_users(audio)
    exclusive = [
        note_id
        for note_id, names in audio.items()
        if names and all(users[name] == 1 for name in names)
    ][:2]
    cards = [_card_of(editable, note_id) for note_id in exclusive]
    cards_before = len(editable.get_cards())

    result = editable.delete_cards(cards)

    assert result["cards_deleted"] == 2
    assert result["notes_deleted"] == 2
    assert len(editable.get_cards()) == cards_before - 2
    assert result["audio_files_deleted"]
