"""Small APKG decks that the tests build themselves.

The repository holds no deck binary. A fixture deck adds megabytes to the
clone, and a real course deck also carries a third-party license. These
builders write the deck into a temporary directory instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from anki_tools.apkg.package import AnkiPackage

FIELDS = ["Hanzi", "English", "Audio"]
MP3_STUB = b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40)
RANKING_NOTES = 120
DECK_NAME = "Test Deck"
MODEL_NAME = "Test Sentence"


def _write_mp3s(directory: Path, names: list[str]) -> list[Path]:
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
    _create(path, cards, _write_mp3s(directory, names))
    _add_second_template(path, len(cards) - 1)
    return path


def build_ranking_deck(directory: Path, count: int = RANKING_NOTES) -> Path:
    """Write a deck of ``count`` notes with one distinct sentence each.

    :param directory: Directory for the deck and its media files.
    :param count: Note count. The rank tests reduce 120 notes to 100.
    :returns: The APKG path.
    """
    names = [f"line{index}.mp3" for index in range(count)]
    cards = [
        {
            "Hanzi": f"第{index}句话",
            "English": f"Line {index}",
            "Audio": f"[sound:line{index}.mp3]",
        }
        for index in range(count)
    ]
    path = directory / "ranking.apkg"
    _create(path, cards, _write_mp3s(directory, names))
    return path
