"""Tests for the Mandarin Swadesh quality gates (G1 to G16)."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from anki_tools.mandarin import audit
from anki_tools.mandarin.audit import print_audit, run_audit
from anki_tools.mandarin.deck import FIELDS, MODEL_NAME
from anki_tools.mandarin.swadesh import SENTENCE_FIELDS, WordForm
from anki_tools.apkg.package import AnkiPackage

FAKE_MP3 = b"ID3\x03\x00\x00\x00" + b"\x00" * 64
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
REVIEW_FIELDS = [
    "lang",
    "index",
    "english",
    "current",
    "canonical",
    "issue",
    "fix",
    "status",
]

TABLE = [
    WordForm(1, "water", "水", "shuǐ", "水", "primary", 5.0),
    WordForm(2, "fire", "火", "huǒ", "火", "primary", 5.0),
    WordForm(3, "mountain", "山", "shān", "山", "primary", 5.0),
    WordForm(4, "dog", "狗", "gǒu", "狗", "primary", 5.0),
    WordForm(5, "bird", "鸟", "niǎo", "鸟", "primary", 5.0),
]

CARDS = [
    {
        "target": "水",
        "sentence": "我要喝水。",
        "pinyin": "wǒ yào hē shuǐ",
        "english": "I want to drink water.",
        "tatoeba_id": "200",
        "tatoeba_author": "bob",
        "audio_source": "tts",
        "tier": "core",
        "audio_missing": "false",
    },
    {
        "target": "火",
        "sentence": "这场火很大。",
        "pinyin": "zhè chǎng huǒ hěn dà",
        "english": "This fire is big.",
        "tatoeba_id": "300",
        "tatoeba_author": "amy",
        "audio_source": "tts",
        "tier": "core",
        "audio_missing": "false",
    },
    {
        "target": "山",
        "sentence": "山上有雪。",
        "pinyin": "shān shàng yǒu xuě",
        "english": "There is snow on the mountain.",
        "tatoeba_id": "400",
        "tatoeba_author": "bob",
        "audio_source": "tts",
        "tier": "core",
        "audio_missing": "false",
    },
    {
        "target": "狗",
        "sentence": "那只狗很黑。",
        "pinyin": "nà zhī gǒu hěn hēi",
        "english": "That dog is black.",
        "tatoeba_id": "500",
        "tatoeba_author": "amy",
        "audio_source": "tts",
        "tier": "core",
        "audio_missing": "false",
    },
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    table: list[WordForm] | None = None,
    cards: list[dict[str, str]] | None = None,
    model_name: str = MODEL_NAME,
    review_status: str = "fixed",
    skip_sentence_audio: frozenset[str] = frozenset(),
    skip_word_audio: frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """Write one small pack and patch the word table and ffprobe.

    :param tmp_path: Test directory.
    :param monkeypatch: Pytest patch helper for the module imports.
    :param table: Word table that the audit reads.
    :param cards: Card rows of the sentence CSV.
    :param model_name: Note model name of the APKG file.
    :param review_status: Status value of the review row.
    :param skip_sentence_audio: Sentences with an index row but no file.
    :param skip_word_audio: Targets with no word audio row and no file.
    :returns: The keyword paths of :func:`run_audit`.
    """
    forms = list(TABLE if table is None else table)
    rows = [dict(card) for card in (CARDS if cards is None else cards)]
    monkeypatch.setattr(audit, "load_word_table", lambda **_: list(forms))
    monkeypatch.setattr(audit, "probe_duration", lambda *_: 1.5)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    index_rows: list[list[str]] = []
    media: list[str] = []
    for number, sentence in enumerate(dict.fromkeys(row["sentence"] for row in rows)):
        name = f"tts_{number:04d}.mp3"
        path = audio_dir / name
        if sentence not in skip_sentence_audio:
            path.write_bytes(FAKE_MP3)
            media.append(str(path))
        index_rows.append(
            [name, sentence, "gtts:zh-CN", "tts", "0", "", "1", "x", "1.5"]
        )
    for number, target in enumerate(dict.fromkeys(row["target"] for row in rows)):
        if target in skip_word_audio:
            continue
        name = f"word_{number:04d}.mp3"
        path = audio_dir / name
        path.write_bytes(FAKE_MP3)
        media.append(str(path))
        index_rows.append([name, target, "gtts:zh-CN", "tts", "0", "", "1", "x", "1.5"])
    with (audio_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_FIELDS)
        writer.writerows(index_rows)

    sentences_csv = tmp_path / "sentences.csv"
    _write_csv(sentences_csv, SENTENCE_FIELDS, rows)

    word_csv = tmp_path / "words.csv"
    _write_csv(
        word_csv,
        ["english", "mandarin", "transliteration"],
        [
            {
                "english": form.english,
                "mandarin": form.mandarin,
                "transliteration": form.pinyin,
            }
            for form in forms
        ],
    )

    review_csv = tmp_path / "swadesh_review.csv"
    _write_csv(
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
    manifest = {
        "counts": {
            "words": len(forms),
            "cards": len(rows),
            "scaffold": sum(1 for row in rows if row.get("tier") == "scaffold"),
            "native_audio": sum(
                1 for row in rows if row.get("audio_source") == "native"
            ),
            "tts_audio": sum(1 for row in rows if row.get("audio_source") == "tts"),
        },
        "files": {
            "source_csv": _sha256(word_csv),
            "sentences_csv": _sha256(sentences_csv),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    attribution_path = tmp_path / "ATTRIBUTION.md"
    attribution_path.write_text(
        "# Attribution\n\nSentences: Tatoeba.\n", encoding="utf-8"
    )

    apkg_path = tmp_path / "pack.apkg"
    AnkiPackage.create(
        apkg_path,
        "Swadesh Mandarin Sentences",
        fields=FIELDS,
        cards=[{field: "" for field in FIELDS} for _ in rows],
        media_files=media,
        model_name=model_name,
    )

    return {
        "apkg_path": apkg_path,
        "sentences_csv": sentences_csv,
        "word_csv": word_csv,
        "review_csv": review_csv,
        "audio_dir": audio_dir,
        "manifest_path": manifest_path,
        "attribution_path": attribution_path,
    }


def _gate(results: list[tuple[str, bool, str]], gate_id: str) -> tuple[bool, str]:
    return next(
        (passed, message) for gate, passed, message in results if gate == gate_id
    )


def _card(**changes: str) -> dict[str, str]:
    card = dict(CARDS[0])
    card.update(changes)
    return card


def test_base_pack_fails_only_the_word_table_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch)

    results = run_audit(**env)
    failed = [gate for gate, passed, _ in results if not passed]

    assert [gate for gate, _, _ in results] == [f"G{number}" for number in range(1, 19)]
    assert failed == ["G1"]
    assert print_audit(results) == 1


def test_print_audit_returns_zero_on_pass(capsys: pytest.CaptureFixture[str]) -> None:
    code = print_audit([("G1", True, "207 concepts"), ("G2", True, "211 unique forms")])

    assert code == 0
    assert "all 2 gates passed" in capsys.readouterr().out


def test_g2_rejects_a_duplicate_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = list(TABLE) + [WordForm(5, "bird", "水", "shuǐ", "水", "alternate", 5.0)]
    env = build_env(tmp_path, monkeypatch, table=table)

    passed, message = _gate(run_audit(**env), "G2")

    assert passed is False
    assert "duplicate form" in message
    assert "水" in message


def test_g4_rejects_a_repeated_tatoeba_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = list(CARDS) + [
        _card(
            target="水",
            sentence="这里没有水。",
            pinyin="zhè lǐ méi yǒu shuǐ",
            tatoeba_id="200",
        )
    ]
    env = build_env(tmp_path, monkeypatch, cards=cards)

    passed, message = _gate(run_audit(**env), "G4")

    assert passed is False
    assert "duplicate Tatoeba id" in message
    assert "200" in message


def test_g5_rejects_a_target_that_appears_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = list(CARDS) + [
        _card(
            target="水",
            sentence="水里有水。",
            pinyin="shuǐ lǐ yǒu shuǐ",
            tatoeba_id="600",
        )
    ]
    env = build_env(tmp_path, monkeypatch, cards=cards)

    passed, message = _gate(run_audit(**env), "G5")

    assert passed is False
    assert "水 x2" in message


def test_g6_rejects_a_similar_card_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = list(CARDS) + [
        _card(target="水", sentence="我要水。", pinyin="wǒ yào shuǐ", tatoeba_id="700")
    ]
    env = build_env(tmp_path, monkeypatch, cards=cards)

    passed, message = _gate(run_audit(**env), "G6")

    assert passed is False
    assert "0.67 limit" in message
    assert "0.75" in message


def test_g7_rejects_a_missing_sentence_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(
        tmp_path, monkeypatch, skip_sentence_audio=frozenset({"我要喝水。"})
    )

    passed, message = _gate(run_audit(**env), "G7")

    assert passed is False
    assert "tts_0000.mp3" in message
    assert "missing" in message


def test_g10_rejects_low_core_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = list(TABLE) + [
        WordForm(index, f"concept{index}", form, "pinyin", form, "alternate", 5.0)
        for index, form in enumerate(("雨", "雪", "风", "云", "星"), start=6)
    ]
    env = build_env(tmp_path, monkeypatch, table=table)

    passed, message = _gate(run_audit(**env), "G10")

    assert passed is False
    assert "core coverage 4/10 = 0.400" in message
    assert "0.80 minimum" in message


def test_g11_rejects_a_bad_manifest_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch)
    manifest = json.loads(env["manifest_path"].read_text(encoding="utf-8"))
    manifest["counts"]["cards"] = 999
    env["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    passed, message = _gate(run_audit(**env), "G11")

    assert passed is False
    assert "counts.cards: manifest 999 != 4" in message


def test_g11_rejects_a_bad_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch)
    manifest = json.loads(env["manifest_path"].read_text(encoding="utf-8"))
    manifest["files"]["sentences_csv"] = "sha256:deadbeef"
    env["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    passed, message = _gate(run_audit(**env), "G11")

    assert passed is False
    assert "files.sentences_csv" in message


def test_g13_rejects_a_wrong_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch, model_name="Basic")

    passed, message = _gate(run_audit(**env), "G13")

    assert passed is False
    assert MODEL_NAME in message
    assert "not found" in message


def test_g13_rejects_a_wrong_note_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch)
    AnkiPackage.create(
        env["apkg_path"],
        "Swadesh Mandarin Sentences",
        fields=FIELDS,
        cards=[{field: "" for field in FIELDS} for _ in range(5)],
        model_name=MODEL_NAME,
    )

    passed, message = _gate(run_audit(**env), "G13")

    assert passed is False
    assert "5 notes but 4 cards" in message


def test_g15_rejects_an_open_review_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch, review_status="open")

    passed, message = _gate(run_audit(**env), "G15")

    assert passed is False
    assert "1 open review row" in message


def test_g15_rejects_a_missing_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch)
    env["review_csv"].unlink()

    passed, message = _gate(run_audit(**env), "G15")

    assert passed is False
    assert "review file missing" in message


def test_g16_rejects_a_missing_word_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build_env(tmp_path, monkeypatch, skip_word_audio=frozenset({"水"}))

    passed, message = _gate(run_audit(**env), "G16")

    assert passed is False
    assert "水 has no word file" in message
