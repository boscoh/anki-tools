"""Contract tests for the APKG package API.

The deck comes from :mod:`tests.deck_fixture`, so the tests state the contract
and do not depend on the quirks of one real deck. The module-scoped fixture
builds the deck one time. Each test that changes the deck copies it first.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import pytest

from anki_tools.apkg.package import AnkiPackage, field_checksum
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


ANKI21B_NOTE_TYPE = "1700000000001"
ANKI21B_NOTE_TYPE_NAME = "Shared sentences"
ANKI21B_FIELDS = ("Sentence", "English", "Audio")
ANKI21B_SCHEMA = """
CREATE TABLE col (id integer primary key, crt integer, decks text);
CREATE TABLE notetypes (
    id integer primary key, name text, mtime_secs integer, usn integer, config blob
);
CREATE TABLE fields (ntid integer, ord integer, name text, config blob);
CREATE TABLE templates (
    ntid integer, ord integer, name text, mtime_secs integer, usn integer,
    config blob
);
CREATE TABLE notes (
    id integer primary key, mid integer, flds text, tags text,
    mod integer, usn integer, sfld text, csum integer
);
CREATE TABLE cards (id integer primary key, nid integer, did integer, ord integer);
CREATE TABLE decks (id integer primary key, name text);
CREATE TABLE revlog (id integer primary key);
CREATE TABLE graves (usn integer, oid integer, type integer);
"""


def _anki21b_database(path: Path) -> None:
    """Fill a SQLite file with the tables of the anki21b schema."""
    connection = sqlite3.connect(path)
    connection.executescript(ANKI21B_SCHEMA)
    connection.execute("INSERT INTO col VALUES (1, 0, '{}')")
    connection.execute(
        "INSERT INTO notetypes VALUES (?, 'Shared sentences', 0, -1, '')",
        (int(ANKI21B_NOTE_TYPE),),
    )
    connection.executemany(
        "INSERT INTO fields VALUES (?, ?, ?, '')",
        [
            (int(ANKI21B_NOTE_TYPE), ord_, name)
            for ord_, name in enumerate(ANKI21B_FIELDS)
        ],
    )
    connection.execute(
        "INSERT INTO templates VALUES (?, 0, 'Card 1', 0, -1, '')",
        (int(ANKI21B_NOTE_TYPE),),
    )
    for index, sentence in enumerate(("Hola.", "Adios.")):
        connection.execute(
            "INSERT INTO notes VALUES (?, ?, ?, '', 0, 0, '', 0)",
            (
                index + 1,
                int(ANKI21B_NOTE_TYPE),
                "\x1f".join([sentence, "English.", f"a{index}.mp3"]),
            ),
        )
        connection.execute(
            "INSERT INTO cards VALUES (?, ?, 1, 0)", (10 + index, index + 1)
        )
    connection.execute("INSERT INTO decks VALUES (1, 'Source')")
    connection.commit()
    connection.close()


def build_anki21b_deck(path: Path) -> Path:
    """Write a minimal APKG file in the anki21b database format.

    :param path: Path of the new APKG file.
    :raises ImportError: The ``zstandard`` package is absent.
    :returns: The same path, after the write.
    """
    zstandard = pytest.importorskip("zstandard")
    database = path.with_name("collection.db")
    _anki21b_database(database)
    compressed = path.with_name("collection.anki21b")
    with database.open("rb") as source, compressed.open("wb") as target:
        zstandard.ZstdCompressor().copy_stream(source, target)
    database.unlink()
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(compressed, "collection.anki21b")
    compressed.unlink()
    return path


@pytest.fixture
def anki21b(tmp_path: Path) -> Iterator[AnkiPackage]:
    deck = build_anki21b_deck(tmp_path / "deck.apkg")
    with AnkiPackage(deck) as open_package:
        yield open_package
        temp_dir = open_package.temp_dir
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


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


CLONE_FIELDS = ("Audio", "Sentence", "English", "Core")
CLONE_TEMPLATES = (("{{Audio}}", "{{Sentence}}"),)
CLONE_CSS = ".card { color: black; }"


def test_clone_model_copies_the_shape_and_takes_a_new_id(
    anki21b: AnkiPackage,
) -> None:
    """The clone holds the new shape, and the source keeps its own fields."""
    source = anki21b.get_models(include_templates=True)[ANKI21B_NOTE_TYPE]

    model_id = anki21b.clone_model(
        ANKI21B_NOTE_TYPE,
        "Output Sentences",
        list(CLONE_FIELDS),
        list(CLONE_TEMPLATES),
        CLONE_CSS,
    )

    clone = anki21b.get_models(include_templates=True)[model_id]
    assert model_id != ANKI21B_NOTE_TYPE
    assert clone["name"] == "Output Sentences"
    assert [field["name"] for field in clone["flds"]] == list(CLONE_FIELDS)
    assert len(clone["tmpls"]) == 1
    assert clone["css"] == CLONE_CSS
    assert [field["name"] for field in source["flds"]] == list(ANKI21B_FIELDS)


def test_clone_model_copies_an_anki2_model(editable: AnkiPackage) -> None:
    """An anki2 clone holds the new fields and one template."""
    source = next(iter(editable.get_models(include_templates=True).values()))
    source_fields = [field["name"] for field in source["flds"]]

    model_id = editable.clone_model(
        next(iter(editable.get_models())),
        "Output Sentences",
        list(CLONE_FIELDS),
        list(CLONE_TEMPLATES),
        CLONE_CSS,
    )

    clone = editable.get_models(include_templates=True)[model_id]
    assert editable.db_format == "anki2"
    assert clone["name"] == "Output Sentences"
    assert [field["name"] for field in clone["flds"]] == list(CLONE_FIELDS)
    assert len(clone["tmpls"]) == 1
    assert clone["css"] == CLONE_CSS
    assert [field["name"] for field in source["flds"]] == source_fields


def test_delete_model_refuses_a_note_that_still_uses_it(
    anki21b: AnkiPackage,
) -> None:
    """A note type that a note holds stays in the package."""
    with pytest.raises(ValueError):
        anki21b.delete_model(ANKI21B_NOTE_TYPE)

    assert anki21b.get_models()[ANKI21B_NOTE_TYPE]["name"] == ANKI21B_NOTE_TYPE_NAME


def test_update_note_writes_the_fields_and_the_model(editable: AnkiPackage) -> None:
    """One call sets the fields, the note type, the sort field, and the checksum."""
    model_id = next(iter(editable.get_models()))
    note_id = int(editable.get_notes()[0]["id"])
    values = ["Hola.", "Hello.", "[sound:a.mp3]"]

    editable.update_note(note_id, values, model_id, 1)

    row = editable.conn.execute(
        "SELECT mid, flds, sfld, csum FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    assert str(row["mid"]) == model_id
    assert row["flds"] == "\x1f".join(values)
    assert row["sfld"] == "Hello."
    assert row["csum"] == field_checksum("Hello.")


def test_update_note_refuses_a_wrong_value_count(editable: AnkiPackage) -> None:
    """A value count that differs from the field count raises."""
    note_id = int(editable.get_notes()[0]["id"])

    with pytest.raises(ValueError):
        editable.update_note(note_id, ["Hola."])


def test_prune_extra_cards_keeps_one_card_per_note(editable: AnkiPackage) -> None:
    """The card with the lowest ord stays, and its ord becomes 0."""
    model_id = next(iter(editable.get_models()))
    note_id = next(note_id for note_id, count in _note_cards(editable).items() if count > 1)

    deleted = editable.prune_extra_cards(model_id)

    cards = [row for row in editable.get_cards() if int(row["nid"]) == note_id]
    assert deleted >= 1
    assert len(cards) == 1
    assert int(cards[0]["ord"]) == 0
    assert _note_cards(editable)[note_id] == 1


# Leading bytes of the note type config of the note type "Spanish sentences
# cloze+" in apkg/Spanish 9000 sentences with native audio.apkg. The kind is
# 1, the sort field index 5, and the CSS starts at byte 7.
CLOZE_CONFIG = bytes.fromhex("080110051ac5012e")

# Leading bytes of the note type config of the note type "Mandarin (Chinese)
# Basic" in apkg/Mandarin (Chinese) (25-02-22).apkg. The kind field is absent
# and the CSS starts at byte 3.
STANDARD_CONFIG = bytes.fromhex("1af9022e63617264")


def test_the_parser_reads_the_kind_and_the_sort_index(anki21b: AnkiPackage) -> None:
    """The field numbers come from the Anki proto message, not from the writer."""
    cloze = anki21b._parse_notetype_config(CLOZE_CONFIG)
    standard = anki21b._parse_notetype_config(STANDARD_CONFIG)

    assert cloze["kind"] == 1
    assert cloze["sort_field_idx"] == 5
    assert standard["kind"] == 0
    assert standard["sort_field_idx"] is None
