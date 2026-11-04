"""The pack environment that the audit tests share."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.mandarin.profile import FIELDS, SENTENCE_FIELDS, WordForm
from anki_tools.ranking.identity import DeckIdentity, default_output
from tests.deck_fixture import write_mp3s
from tests.mandarin_fixture import ROWS, TABLE, audio_entries, write_audio, write_csv

PACK_PATH = default_output("mandarin", "pack")
MODEL_NAME = DeckIdentity(PACK_PATH).model_name
REVIEW_FIELDS = ["lang", "index", "english", "current", "canonical", "issue", "fix", "status"]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    sentence_fields: Sequence[str],
    word_fields: Sequence[str],
    word_rows: list[dict[str, str]],
    review_status: str = "fixed",
    no_file: frozenset[str] = frozenset(),
    no_index: frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """Write the audio directory and the three CSV files that a pack reads.

    :param tmp_path: Test directory.
    :param rows: Card rows of the sentence CSV.
    :param sentence_fields: Columns of the sentence CSV.
    :param word_fields: Columns of the word CSV.
    :param word_rows: Rows of the word CSV.
    :param review_status: Status value of the review row.
    :param no_file: Texts with an index row but no MP3 file.
    :param no_index: Texts with neither an index row nor a file.
    :returns: The paths ``audio_dir``, ``sentences_csv``, ``word_csv``, ``review_csv``.
    """
    audio_dir = tmp_path / "audio"
    entries = [entry for entry in audio_entries(rows) if entry[1] not in no_index]
    write_audio(audio_dir, entries, no_file)
    sentences_csv = tmp_path / "sentences.csv"
    write_csv(sentences_csv, sentence_fields, rows)
    word_csv = tmp_path / "words.csv"
    write_csv(word_csv, word_fields, word_rows)
    review_csv = tmp_path / "swadesh_review.csv"
    write_csv(
        review_csv,
        REVIEW_FIELDS,
        [
            {
                "lang": "cmn",
                "index": "3",
                "english": "he/she/it/they (singular)",
                "current": "他/她",
                "canonical": "他/她/它",
                "issue": "label",
                "fix": "他/她/它",
                "status": review_status,
            }
        ],
    )
    return {
        "audio_dir": audio_dir,
        "sentences_csv": sentences_csv,
        "word_csv": word_csv,
        "review_csv": review_csv,
    }


def write_apkg(
    env: dict[str, Path],
    notes: int = len(ROWS),
    fields: Sequence[str] = FIELDS,
    model_name: str = MODEL_NAME,
    drop: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> None:
    """Write the pack APKG with the MP3 files of the audio directory, less ``drop``, plus ``extra``."""
    audio_dir = env["audio_dir"]
    write_mp3s(audio_dir, list(extra))
    media = [
        str(path)
        for path in sorted(audio_dir.glob("*.mp3"))
        if path.name not in drop
    ]
    AnkiPackage.create(
        env["apkg_path"],
        DeckIdentity(PACK_PATH).title,
        fields=list(fields),
        cards=[{field: "" for field in fields} for _ in range(notes)],
        media_files=media,
        model_name=model_name,
    )


def build_env(
    tmp_path: Path,
    *,
    table: list[WordForm] = TABLE,
    rows: list[dict[str, str]] = ROWS,
    review_status: str = "fixed",
    no_file: frozenset[str] = frozenset(),
    no_index: frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """Write one small Mandarin pack that passes every gate but P1.

    :param tmp_path: Test directory.
    :param table: Word table that the audit reads.
    :param rows: Card rows of the sentence CSV.
    :param review_status: Status value of the review row.
    :param no_file: Texts with an index row but no MP3 file.
    :param no_index: Texts with neither an index row nor a file.
    :returns: The keyword paths of ``run_pack_audit``.
    """
    env = write_inputs(
        tmp_path,
        rows,
        sentence_fields=SENTENCE_FIELDS,
        word_fields=["english", "mandarin", "transliteration"],
        word_rows=[
            {"english": form.english, "mandarin": form.mandarin, "transliteration": form.pinyin}
            for form in table
        ],
        review_status=review_status,
        no_file=no_file,
        no_index=no_index,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "counts": {
                    "words": len(table),
                    "cards": len(rows),
                    "scaffold": sum(row["tier"] == "scaffold" for row in rows),
                    "native_audio": sum(row["audio_source"] == "native" for row in rows),
                    "tts_audio": sum(row["audio_source"] == "tts" for row in rows),
                },
                "files": {
                    "source_csv": sha256(env["word_csv"]),
                    "sentences_csv": sha256(env["sentences_csv"]),
                },
            }
        ),
        encoding="utf-8",
    )
    attribution_path = tmp_path / "ATTRIBUTION.md"
    attribution_path.write_text("# Attribution\n\nSentences: Tatoeba.\n", encoding="utf-8")
    env |= {
        "apkg_path": tmp_path / PACK_PATH.name,
        "manifest_path": manifest_path,
        "attribution_path": attribution_path,
    }
    write_apkg(env, notes=len(rows))
    return env
