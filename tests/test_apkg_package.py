"""Contract tests for the APKG package API.

The deck comes from :mod:`tests.deck_fixture`, so the tests state the contract
and do not depend on the quirks of one real deck. The module-scoped fixture
builds the deck one time. Each test that changes the deck copies it first.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterator

import pytest

from anki_tools.apkg.package import AnkiPackage
from tests.deck_fixture import MP3_STUB, build_contract_deck

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


def _note_cards(package: AnkiPackage) -> Counter[int]:
    """Count the cards of each note."""
    return Counter(int(row["nid"]) for row in package.get_cards())


def _card_of(package: AnkiPackage, note_id: int) -> int:
    """Give the card id of one note."""
    return next(
        int(row["id"]) for row in package.get_cards() if int(row["nid"]) == note_id
    )


@pytest.fixture(scope="module")
def deck_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_contract_deck(tmp_path_factory.mktemp("deck"))


@pytest.fixture(scope="module")
def package(deck_path: Path) -> Iterator[AnkiPackage]:
    with AnkiPackage(deck_path) as open_package:
        yield open_package


@pytest.fixture
def editable(deck_path: Path, tmp_path: Path) -> Iterator[AnkiPackage]:
    target = tmp_path / "deck.apkg"
    shutil.copy(deck_path, target)
    with AnkiPackage(target) as open_package:
        yield open_package


def test_the_deck_references_every_media_file(package: AnkiPackage) -> None:
    """A media file with no note is dead weight in the package."""
    media = set(package.get_media_mapping().values())
    used = set().union(*_note_audio(package).values())

    assert media == used


def test_the_extraction_writes_every_mapped_audio_file(
    package: AnkiPackage, tmp_path: Path
) -> None:
    extracted = package.extract_audio_files(tmp_path, audio_only=True)
    media = set(package.get_media_mapping().values())

    assert set(extracted) == {name for name in media if name.endswith(AUDIO_SUFFIXES)}
    assert extracted
    for path in extracted.values():
        assert Path(path).stat().st_size == len(MP3_STUB)


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
    assert names.isdisjoint(set(editable.get_media_mapping().values()))


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
    assert shared <= set(editable.get_media_mapping().values())


def test_delete_card_keeps_the_note_of_a_two_card_note(editable: AnkiPackage) -> None:
    note_id = next(note_id for note_id, count in _note_cards(editable).items() if count > 1)
    audio = _note_audio(editable)[note_id]

    result = editable.delete_card(_card_of(editable, note_id))

    assert result["card_deleted"] is True
    assert result["note_deleted"] is False
    assert result["audio_files_deleted"] == []
    assert editable.is_audio_file_used(next(iter(audio)))


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
