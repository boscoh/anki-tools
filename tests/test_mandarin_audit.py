"""Tests for the Mandarin Swadesh quality gates (G1 to G16)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.gates import GateResult
from anki_tools.languages.mandarin.profile import FIELDS, SENTENCE_FIELDS, WordForm
from anki_tools.languages.mandarin.stages import GATES, MANDARIN, print_audit
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack.audio import MAX_DURATION_S, MIN_DURATION_S
from anki_tools.sentence_pack.audit import run_pack_audit
from tests.deck_fixture import write_mp3s
from tests.mandarin_fixture import (
    ROWS,
    TABLE,
    audio_entries,
    card,
    write_audio,
    write_csv,
)

PACK_PATH = default_output("mandarin", "pack")
MODEL_NAME = DeckIdentity(PACK_PATH).model_name
REVIEW_FIELDS = ["lang", "index", "english", "current", "canonical", "issue", "fix", "status"]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(duration: float):
    return lambda _path: duration


def _unreadable(_path: Path) -> float:
    raise RuntimeError("ffprobe failed")


def _overlap_pair(shared: int) -> tuple[str, str]:
    """Give two 100-character sentences whose character Jaccard is ``shared / 100``."""
    chars = [chr(0x5000 + index) for index in range(100)]
    only_first = chars[shared : shared + 16]
    return "".join(chars[:shared] + only_first), "".join(chars[:shared] + chars[shared + 16 :])


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
    """Write one small pack that passes every gate but G1.

    :param tmp_path: Test directory.
    :param table: Word table that the audit reads.
    :param rows: Card rows of the sentence CSV.
    :param review_status: Status value of the review row.
    :param no_file: Texts with an index row but no MP3 file.
    :param no_index: Texts with neither an index row nor a file.
    :returns: The keyword paths of ``run_pack_audit``.
    """
    audio_dir = tmp_path / "audio"
    entries = [entry for entry in audio_entries(rows) if entry[1] not in no_index]
    write_audio(audio_dir, entries, no_file)

    sentences_csv = tmp_path / "sentences.csv"
    write_csv(sentences_csv, SENTENCE_FIELDS, rows)
    word_csv = tmp_path / "words.csv"
    write_csv(
        word_csv,
        ["english", "mandarin", "transliteration"],
        [
            {"english": form.english, "mandarin": form.mandarin, "transliteration": form.pinyin}
            for form in table
        ],
    )
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
                    "source_csv": _sha256(word_csv),
                    "sentences_csv": _sha256(sentences_csv),
                },
            }
        ),
        encoding="utf-8",
    )
    attribution_path = tmp_path / "ATTRIBUTION.md"
    attribution_path.write_text("# Attribution\n\nSentences: Tatoeba.\n", encoding="utf-8")
    env = {
        "apkg_path": tmp_path / PACK_PATH.name,
        "sentences_csv": sentences_csv,
        "word_csv": word_csv,
        "review_csv": review_csv,
        "audio_dir": audio_dir,
        "manifest_path": manifest_path,
        "attribution_path": attribution_path,
    }
    write_apkg(env, notes=len(rows))
    return env


def run(
    env: dict[str, Path], table: list[WordForm] = TABLE, probe=_probe(1.5)
) -> list[GateResult]:
    return run_pack_audit(
        MANDARIN, GATES, load_table=lambda **_: list(table), probe=probe, **env
    )


def audit(
    tmp_path: Path,
    *,
    table: list[WordForm] = TABLE,
    probe=_probe(1.5),
    apkg: dict[str, Any] | None = None,
    **build: Any,
) -> list[GateResult]:
    env = build_env(tmp_path, table=table, **build)
    if apkg:
        write_apkg(env, **apkg)
    return run(env, table, probe)


def _gate(results: list[GateResult], gate_id: str) -> tuple[bool, str]:
    return next((passed, message) for gate, passed, message in results if gate == gate_id)


def _card(**changes: str) -> dict[str, str]:
    return ROWS[0] | changes


def test_base_pack_fails_only_the_word_table_gate(tmp_path: Path) -> None:
    results = run(build_env(tmp_path))
    failed = [gate for gate, passed, _ in results if not passed]

    assert [gate for gate, _, _ in results] == [f"G{number}" for number in range(1, 17)]
    assert failed == ["G1"]
    assert print_audit(results) == 1


MOON = card("月", "月亮很大。", "yuè liang hěn dà", "The moon is big.", "900", "bob")
SIMILAR_AT_LIMIT = _overlap_pair(67)
SIMILAR_BELOW_LIMIT = _overlap_pair(66)


@pytest.mark.parametrize(
    ("setup", "gate", "ok", "fragments"),
    [
        pytest.param(
            {"table": TABLE + [WordForm(5, "bird", "水", "shuǐ", "水", "alternate", 5.0)]},
            "G2", False, ("duplicate form", "水"),
            id="G2 a duplicate form",
        ),
        pytest.param(
            {"rows": ROWS + [MOON]},
            "G3", False, ("月 is not in the word table",),
            id="G3 a target outside the table",
        ),
        pytest.param(
            {
                "table": TABLE + [WordForm(6, "moon", "月", "yuè", "月", "shared", 5.0)],
                "rows": ROWS + [MOON],
            },
            "G3", False, ("月 is a shared form",),
            id="G3 a target on a shared form",
        ),
        pytest.param(
            {"rows": ROWS + [_card(sentence="这里没有水。", pinyin="zhè lǐ méi yǒu shuǐ")]},
            "G4", False, ("duplicate Tatoeba id", "200"),
            id="G4 a repeated Tatoeba id",
        ),
        pytest.param(
            {"rows": ROWS + [_card(sentence="水里有水。", pinyin="shuǐ lǐ yǒu shuǐ", tatoeba_id="600")]},
            "G5", False, ("水 x2",),
            id="G5 a target that appears twice",
        ),
        pytest.param(
            {"rows": ROWS + [_card(sentence="我要水。", pinyin="wǒ yào shuǐ", tatoeba_id="700")]},
            "G6", False, ("0.67 limit", "0.75"),
            id="G6 a similar pair",
        ),
        pytest.param(
            {
                "rows": ROWS
                + [
                    _card(sentence=SIMILAR_AT_LIMIT[0], tatoeba_id="701"),
                    _card(target="火", sentence=SIMILAR_AT_LIMIT[1], tatoeba_id="702"),
                ]
            },
            "G6", False, ("= 0.67",),
            id="G6 a pair at the limit",
        ),
        pytest.param(
            {
                "rows": ROWS
                + [
                    _card(sentence=SIMILAR_BELOW_LIMIT[0], tatoeba_id="701"),
                    _card(target="火", sentence=SIMILAR_BELOW_LIMIT[1], tatoeba_id="702"),
                ]
            },
            "G6", True, ("stay below",),
            id="G6 a pair below the limit",
        ),
        pytest.param(
            {"no_file": frozenset({"我要喝水。"})},
            "G7", False, ("tts_0000.mp3", "missing"),
            id="G7 a missing sentence file",
        ),
        pytest.param(
            {"probe": _probe(MIN_DURATION_S - 0.01)},
            "G7", False, ("0.39 s",),
            id="G7 just under the shortest duration",
        ),
        pytest.param(
            {"probe": _probe(MIN_DURATION_S)}, "G7", True, ("sentence files",),
            id="G7 the shortest duration",
        ),
        pytest.param(
            {"probe": _probe(MAX_DURATION_S)}, "G7", True, ("sentence files",),
            id="G7 the longest duration",
        ),
        pytest.param(
            {"probe": _probe(MAX_DURATION_S + 0.01)},
            "G7", False, ("15.01 s",),
            id="G7 just over the longest duration",
        ),
        pytest.param(
            {"probe": _unreadable}, "G7", False, ("is not readable",),
            id="G7 an unreadable file",
        ),
        pytest.param(
            {"apkg": {"extra": ("stray.mp3",)}},
            "G8", False, ("media files, cards use",),
            id="G8 an extra media file",
        ),
        pytest.param(
            {"apkg": {"drop": ("word_0000.mp3",), "extra": ("stray.mp3",)}},
            "G8", False, ("media names differ", "word_0000.mp3", "stray.mp3"),
            id="G8 the same count with other names",
        ),
        pytest.param(
            {
                "table": TABLE
                + [
                    WordForm(index, f"concept{index}", form, "pinyin", form, "alternate", 5.0)
                    for index, form in enumerate(("雨", "雪", "风", "云", "星"), start=6)
                ]
            },
            "G10", False, ("core coverage 4/10 = 0.400", "0.80 minimum"),
            id="G10 low core coverage",
        ),
        pytest.param(
            {"apkg": {"model_name": "Basic"}},
            "G13", False, (MODEL_NAME, "not found"),
            id="G13 a wrong model name",
        ),
        pytest.param(
            {"apkg": {"notes": 5}}, "G13", False, ("5 notes but 4 cards",),
            id="G13 a wrong note count",
        ),
        pytest.param(
            {"apkg": {"fields": FIELDS[:-1]}}, "G13", False, ("model fields",),
            id="G13 a wrong field list",
        ),
        pytest.param(
            {"review_status": "open"}, "G15", False, ("1 open review row",),
            id="G15 an open review row",
        ),
        pytest.param(
            {"no_file": frozenset({"水"})},
            "G16", False, ("word_0000.mp3 for 水 is missing",),
            id="G16 a missing word file",
        ),
        pytest.param(
            {"no_index": frozenset({"水"})},
            "G16", False, ("水 has no word file in the index",),
            id="G16 a word with no index row",
        ),
    ],
)
def test_the_gate_verdict_follows_the_pack(
    tmp_path: Path,
    setup: dict[str, Any],
    gate: str,
    ok: bool,
    fragments: tuple[str, ...],
) -> None:
    passed, message = _gate(audit(tmp_path, **setup), gate)

    assert passed is ok
    for fragment in fragments:
        assert fragment in message


@pytest.mark.parametrize(
    ("section", "key", "value", "fragment"),
    [
        pytest.param("counts", "cards", 999, "counts.cards: manifest 999 != 4", id="a bad count"),
        pytest.param("files", "sentences_csv", "sha256:deadbeef", "files.sentences_csv", id="a bad hash"),
    ],
)
def test_g11_rejects_a_manifest_that_differs_from_the_pack(
    tmp_path: Path, section: str, key: str, value: Any, fragment: str
) -> None:
    env = build_env(tmp_path)
    manifest = json.loads(env["manifest_path"].read_text(encoding="utf-8"))
    manifest[section][key] = value
    env["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    passed, message = _gate(run(env), "G11")

    assert passed is False
    assert fragment in message


def test_g15_rejects_a_missing_review_file(tmp_path: Path) -> None:
    env = build_env(tmp_path)
    env["review_csv"].unlink()

    passed, message = _gate(run(env), "G15")

    assert passed is False
    assert "review file missing" in message
