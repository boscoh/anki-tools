"""Small APKG decks that the tests build themselves.

The repository holds no deck binary. A fixture deck adds megabytes to the
clone, and a real course deck also carries a third-party license. These
builders write the deck into a temporary directory instead.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage

FIELDS = ["Hanzi", "English", "Audio"]
MP3_STUB = b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40)
DECK_NAME = "Test Deck"
MODEL_NAME = "Test Sentence"


def write_mp3s(directory: Path, names: list[str]) -> list[Path]:
    """Write one tiny MP3 file for each name."""
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(MP3_STUB)
        paths.append(path)
    return paths


def _create(path: Path, cards: list[dict[str, str]], media: list[Path]) -> None:
    """Write a one-template deck with the test fields."""
    AnkiPackage.create(
        path,
        DECK_NAME,
        fields=FIELDS,
        cards=cards,
        media_files=[str(item) for item in media],
        model_name=MODEL_NAME,
        question_format="{{Audio}}",
        answer_format="{{Audio}}<hr id=answer>{{Hanzi}}<br>{{English}}",
    )


def _add_second_template(path: Path, note_index: int) -> None:
    """Add a second card template and give one note a card of that template."""
    with AnkiPackage(path) as package:
        conn = package.conn
        assert conn is not None
        models = json.loads(conn.execute("SELECT models FROM col").fetchone()[0])
        model_id = next(iter(models))
        templates = models[model_id]["tmpls"]
        extra = dict(templates[0])
        extra.update({"name": "Card 2", "ord": len(templates)})
        templates.append(extra)
        conn.execute(
            "UPDATE col SET models = ? WHERE id = 1", (json.dumps(models),)
        )
        note_id = int(
            conn.execute("SELECT id FROM notes ORDER BY id").fetchall()[note_index][0]
        )
        card = conn.execute("SELECT * FROM cards WHERE nid = ?", (note_id,)).fetchone()
        columns = [row[1] for row in conn.execute("PRAGMA table_info(cards)")]
        next_id = int(conn.execute("SELECT MAX(id) FROM cards").fetchone()[0]) + 1
        record = {name: card[name] for name in card.keys()}
        record.update(id=next_id, ord=1)
        conn.execute(
            f"INSERT INTO cards ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            [record[column] for column in columns],
        )
        conn.commit()
        package._modified = True
        package.save()


def build_contract_deck(directory: Path) -> Path:
    """Write a 6-note deck for the package API tests.

    The deck has one audio file per note, except that note 5 shares the audio
    of note 1. The last note has two cards.

    :param directory: Directory for the deck and its media files.
    :returns: The APKG path.
    """
    names = [f"note{index}.mp3" for index in range(1, 6)]
    audio = ["note1.mp3", "note2.mp3", "note3.mp3", "note4.mp3", "note1.mp3", "note5.mp3"]
    cards = [
        {
            "Hanzi": f"第{index}句话",
            "English": f"Line {index}",
            "Audio": f"[sound:{name}]",
        }
        for index, name in enumerate(audio, start=1)
    ]
    path = directory / "contract.apkg"
    _create(path, cards, write_mp3s(directory, names))
    _add_second_template(path, len(cards) - 1)
    return path


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
