"""Tests for the Mandarin Swadesh sentence pack build (stage S5)."""

import csv
import json
from pathlib import Path

import pytest

from anki_tools.languages.mandarin import stages as deck
from anki_tools.languages.mandarin.profile import SWADESH_ROWS, WordForm
from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import load_char_links

FAKE_MP3 = b"ID3\x03\x00\x00\x00" + b"\x00" * 64

SENTENCE_FIELDS = [
    "target",
    "sentence",
    "pinyin",
    "english",
    "tatoeba_id",
    "tatoeba_author",
    "audio_id",
    "audio_source",
    "audio_license",
    "tier",
    "audio_missing",
]

WORD_ROWS = [
    {
        "target": "水",
        "sentence": "我要喝水。",
        "pinyin": "wǒ yào hē shuǐ",
        "english": "I want to drink water.",
        "tatoeba_id": "200",
        "tatoeba_author": "bob",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "tier": "core",
        "audio_missing": "false",
    },
    {
        "target": "火",
        "sentence": "火很大。",
        "pinyin": "huǒ hěn dà",
        "english": "The fire is big.",
        "tatoeba_id": "100",
        "tatoeba_author": "amy",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "tier": "scaffold",
        "audio_missing": "false",
    },
]

TABLE = [
    WordForm(1, "water", "水", "shuǐ", "水", "primary", 5.0),
    WordForm(2, "fire", "火", "huǒ", "火", "primary", 5.0),
]

INDEX_ROWS = [
    ("tts_aaaa.mp3", "我要喝水。", "gtts:zh-CN"),
    ("tts_bbbb.mp3", "火很大。", "gtts:zh-CN"),
    ("word_aaaa.mp3", "水", "gtts:zh-CN"),
    ("word_bbbb.mp3", "火", "gtts:zh-CN"),
]


def test_slug() -> None:
    assert deck.slug("water") == "water"
    assert deck.slug("I") == "i"
    assert deck.slug("he/she/it/they (singular)") == "he_she_it_they_singular"
    assert deck.slug("  Water!! ") == "water"
    assert deck.slug("") == ""


def test_build_card_fields_and_tags() -> None:
    card = deck.build_card(WORD_ROWS[0], TABLE[0], "tts_aaaa.mp3", "word_aaaa.mp3")

    assert list(card)[:8] == deck.FIELDS
    assert card["Audio"] == "[sound:tts_aaaa.mp3]"
    assert card["Sentence"] == "我要喝<b>水</b>。"
    assert card["Romanization"] == "wǒ yào hē shuǐ"
    assert card["English"] == "I want to drink water."
    assert card["Target"] == "水"
    assert card["Gloss"] == "water · shuǐ"
    assert card["WordAudio"] == "[sound:word_aaaa.mp3]"
    assert card["Source"] == "Tatoeba #200 · bob"
    assert card["tags"] == ["swadesh::water", "audio::tts", "tier::core"]


def test_sentence_target_is_bold_once() -> None:
    card = deck.build_card(WORD_ROWS[1], TABLE[1], "tts_bbbb.mp3", "word_bbbb.mp3")

    assert card["Sentence"].count("<b>火</b>") == 1
    assert card["Sentence"].count("<b>") == 1
    assert card["tags"][1:] == ["audio::tts", "tier::scaffold"]


def test_media_paths_are_absolute_and_unique(tmp_path: Path) -> None:
    paths = deck.media_paths(
        ["b.mp3", "a.mp3", "b.mp3"], audio_dir=tmp_path / "audio"
    )

    assert [Path(path).name for path in paths] == ["a.mp3", "b.mp3"]
    assert all(Path(path).is_absolute() for path in paths)


def test_load_audio_index_splits_word_and_sentence(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(deck_index_fields())
        for filename, text, voice in INDEX_ROWS:
            writer.writerow([filename, text, voice, "tts", "0", "", "1", "x", "1.0"])

    index = deck.load_audio_index(index_path)

    assert index.sentences == {"我要喝水。": "tts_aaaa.mp3", "火很大。": "tts_bbbb.mp3"}
    assert index.words == {"水": "word_aaaa.mp3", "火": "word_bbbb.mp3"}
    assert index.voice == "gtts:zh-CN"


def deck_index_fields() -> list[str]:
    return [
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


@pytest.fixture
def pack_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for filename, _, _ in INDEX_ROWS:
        (audio_dir / filename).write_bytes(FAKE_MP3)
    with (audio_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(deck_index_fields())
        for filename, text, voice in INDEX_ROWS:
            writer.writerow([filename, text, voice, "tts", "0", "", "1", "x", "1.0"])

    csv_path = tmp_path / "sentences.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_FIELDS)
        writer.writeheader()
        writer.writerows(WORD_ROWS)

    word_csv = tmp_path / "words.csv"
    word_csv.write_text("english,mandarin,transliteration\n", encoding="utf-8")

    monkeypatch.setattr(deck, "load_word_table", lambda **_: list(TABLE))
    return {
        "audio_dir": audio_dir,
        "csv_path": csv_path,
        "word_csv": word_csv,
        "output": tmp_path / "pack.apkg",
        "manifest": tmp_path / "manifest.json",
        "attribution": tmp_path / "ATTRIBUTION.md",
    }


def test_pack_manifest_counts(pack_env: dict[str, Path]) -> None:
    result = deck.build_pack(
        csv_path=pack_env["csv_path"],
        word_csv=pack_env["word_csv"],
        audio_dir=pack_env["audio_dir"],
        output=pack_env["output"],
        manifest_path=pack_env["manifest"],
        attribution_path=pack_env["attribution"],
    )
    manifest = json.loads(pack_env["manifest"].read_text(encoding="utf-8"))

    assert result["manifest"] == manifest
    assert result["media_count"] == 4
    assert result["skipped"] == 0
    assert manifest["counts"] == {
        "words": 2,
        "cards": 2,
        "scaffold": 1,
        "native_audio": 0,
        "tts_audio": 2,
        "no_target": 0,
    }
    assert manifest["coverage"] == {"core_with_card": round(2 / SWADESH_ROWS, 4)}
    assert manifest["voice"] == "gtts:zh-CN"
    assert manifest["pack_license"] == "public"
    assert manifest["audio_license_skipped"] == {"empty": 0, "CC BY-NC 4.0": 0}
    source = manifest["audio_license_source"]
    assert set(source) == {"empty", "CC BY-NC 4.0"}
    assert all(count >= 0 for count in source.values())
    assert sum(source.values()) > 0
    assert manifest["files"]["source_csv"].startswith("sha256:")
    assert manifest["files"]["sentences_csv"].startswith("sha256:")
    assert manifest["licenses"] == [
        {"data": "Swadesh word data", "license": "CC BY-SA 4.0"},
        {"data": "Tatoeba sentences", "license": "CC BY 2.0 FR"},
        {"data": "Tatoeba audio", "license": "per recording"},
    ]
    assert manifest["excluded_words"] == []

    attribution = pack_env["attribution"].read_text(encoding="utf-8")
    assert "Sentences come from Tatoeba (<https://tatoeba.org>)" in attribution
    assert "### Word data" in attribution
    assert "## Mandarin pack" in attribution
    assert "gTTS" in attribution


def test_create_passes_tags_and_deck_description(tmp_path: Path) -> None:
    output = tmp_path / "tags.apkg"
    AnkiPackage.create(
        output,
        "Tagged",
        fields=["Front"],
        cards=[{"Front": "one", "tags": ["swadesh::water", "tier::core"]}],
        deck_description="Short description",
    )

    with AnkiPackage(output) as pkg:
        notes = pkg.get_notes()
        decks = pkg.get_decks()

    assert len(notes) == 1
    assert "swadesh::water" in notes[0]["tags"]
    assert "tier::core" in notes[0]["tags"]
    tagged = [info for info in decks.values() if info.get("name") == "Tagged"]
    assert [info.get("desc") for info in tagged] == ["Short description"]


def test_apkg_opens_with_two_cards(pack_env: dict[str, Path]) -> None:
    deck.build_pack(
        csv_path=pack_env["csv_path"],
        word_csv=pack_env["word_csv"],
        audio_dir=pack_env["audio_dir"],
        output=pack_env["output"],
        manifest_path=pack_env["manifest"],
        attribution_path=pack_env["attribution"],
    )

    with AnkiPackage(pack_env["output"]) as pkg:
        models = pkg.get_models()
        decks = pkg.get_decks()
        cards = pkg.get_cards()
        parsed = [pkg.parse_card(card, models, decks) for card in cards]
        audio_stats = pkg.get_audio_statistics()

    assert len(cards) == 2
    assert audio_stats["audio_files"] == 4
    assert [item["fields"]["Target"] for item in parsed] == ["水", "火"]
    assert [item["model"] for item in parsed] == [deck.MODEL_NAME, deck.MODEL_NAME]
    assert [list(item["fields"]) for item in parsed] == [deck.FIELDS, deck.FIELDS]
    assert parsed[0]["fields"]["Sentence"] == "我要喝<b>水</b>。"
    assert "swadesh::water" in parsed[0]["tags"]
    assert "audio::tts" in parsed[0]["tags"]
    assert "tier::scaffold" in parsed[1]["tags"]
    named = [info for info in decks.values() if info.get("name") == deck.DECK_NAME]
    assert [info.get("desc") for info in named] == [deck.DECK_DESCRIPTION]

    model = next(iter(models.values()))
    assert model.get("sortf") == deck.SORT_FIELD_INDEX
    template = model["tmpls"][0]
    assert template["qfmt"] == deck.FRONT_TEMPLATE
    assert template["afmt"] == f"{deck.BACK_TEMPLATE}\n{load_char_links()}"
