"""The Mandarin pack data that the audit and the build tests share."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from anki_tools.languages.mandarin.profile import WordForm
from tests.deck_fixture import write_mp3s

VOICE = "gtts:zh-CN"
INDEX_FIELDS = [
    "filename",
    "text",
    "voice",
    "source",
    "audio_id",
    "license",
    "bytes",
    "sha1",
    "duration_s",
]

TABLE = [
    WordForm(1, "water", "水", "shuǐ", "水", "primary", 5.0),
    WordForm(2, "fire", "火", "huǒ", "火", "primary", 5.0),
    WordForm(3, "mountain", "山", "shān", "山", "primary", 5.0),
    WordForm(4, "dog", "狗", "gǒu", "狗", "primary", 5.0),
    WordForm(5, "bird", "鸟", "niǎo", "鸟", "primary", 5.0),
]


def pack_row(
    target: str,
    sentence: str,
    pinyin: str,
    english: str,
    tatoeba_id: str,
    author: str,
) -> dict[str, str]:
    return {
        "target": target,
        "sentence": sentence,
        "pinyin": pinyin,
        "english": english,
        "tatoeba_id": tatoeba_id,
        "tatoeba_author": author,
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "tier": "core",
        "audio_missing": "false",
    }


ROWS = [
    pack_row("水", "我要喝水。", "wǒ yào hē shuǐ", "I want to drink water.", "200", "bob"),
    pack_row("火", "这场火很大。", "zhè chǎng huǒ hěn dà", "This fire is big.", "300", "amy"),
    pack_row("山", "山上有雪。", "shān shàng yǒu xuě", "There is snow on the mountain.", "400", "bob"),
    pack_row("狗", "那只狗很黑。", "nà zhī gǒu hěn hēi", "That dog is black.", "500", "amy"),
]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audio_entries(rows: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    """Name one sentence file and one word file for each distinct text."""
    sentences = dict.fromkeys(row["sentence"] for row in rows)
    targets = dict.fromkeys(row["target"] for row in rows)
    return [(f"tts_{number:04d}.mp3", text) for number, text in enumerate(sentences)] + [
        (f"word_{number:04d}.mp3", text) for number, text in enumerate(targets)
    ]


def write_audio(
    audio_dir: Path,
    entries: Sequence[tuple[str, str]],
    no_file: frozenset[str] = frozenset(),
) -> None:
    """Write the index of ``entries`` and one stub MP3 each, but none for ``no_file`` texts."""
    audio_dir.mkdir(exist_ok=True)
    write_mp3s(audio_dir, [name for name, text in entries if text not in no_file])
    write_csv(
        audio_dir / "index.csv",
        INDEX_FIELDS,
        [
            dict(zip(INDEX_FIELDS, [name, text, VOICE, "tts", "0", "", "1", "x", "1.5"]))
            for name, text in entries
        ],
    )
