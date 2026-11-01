"""A tiny Tatoeba data directory that the pack tests share."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class PoolRow(NamedTuple):
    id: int
    text: str
    english_id: int
    english: str | None
    author: str = "alice"
    audio: tuple[int, str, str] | None = None


def write_tsv(path: Path, rows: list[tuple[object, ...]]) -> None:
    text = "\n".join("\t".join(str(cell) for cell in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def write_tatoeba_dir(data_dir: Path, lang: str, rows: list[PoolRow]) -> Path:
    """Write the five Tatoeba files of one language; a row without English has a link only."""
    data_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(data_dir / f"{lang}_sentences.tsv", [(r.id, lang, r.text) for r in rows])
    write_tsv(data_dir / f"{lang}-eng_links.tsv", [(r.id, r.english_id) for r in rows])
    write_tsv(
        data_dir / "eng_sentences.tsv",
        [(r.english_id, "eng", r.english) for r in rows if r.english],
    )
    write_tsv(
        data_dir / f"{lang}_sentences_detailed.tsv",
        [(r.id, lang, r.text, r.author, "\\N", "0000-00-00") for r in rows],
    )
    write_tsv(
        data_dir / f"{lang}_sentences_with_audio.tsv",
        [(r.id, r.audio[0], r.audio[1], r.audio[2], "") for r in rows if r.audio],
    )
    return data_dir
