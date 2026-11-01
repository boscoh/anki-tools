"""Tests for the Mandarin Swadesh sentence pack build (stage S5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import load_char_links
from anki_tools.languages.mandarin import stages as deck
from anki_tools.languages.mandarin.profile import SENTENCE_FIELDS, SWADESH_ROWS
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.sentence_pack import pack as shared_pack
from tests.mandarin_fixture import (
    ROWS,
    TABLE,
    VOICE,
    audio_entries,
    write_audio,
    write_csv,
)

PACK_ROWS = [ROWS[0], ROWS[1] | {"tier": "scaffold"}]
PACK_TABLE = TABLE[:2]


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("water", "water"),
        ("I", "i"),
        ("he/she/it/they (singular)", "he_she_it_they_singular"),
        ("  Water!! ", "water"),
        ("", ""),
    ],
)
def test_slug_makes_a_tag_component(text: str, tag: str) -> None:
    assert deck.slug(text) == tag


def test_build_card_fields_and_tags() -> None:
    card = deck.mandarin_card(PACK_ROWS[0], PACK_TABLE[0], "tts_0000.mp3", "word_0000.mp3")
    scaffold = deck.mandarin_card(PACK_ROWS[1], PACK_TABLE[1], "tts_0001.mp3", "word_0001.mp3")

    assert list(card)[:8] == deck.FIELDS
    assert card["Audio"] == "[sound:tts_0000.mp3]"
    assert card["Sentence"] == "我要喝<b>水</b>。"
    assert card["Romanization"] == "wǒ yào hē shuǐ"
    assert card["English"] == "I want to drink water."
    assert card["Target"] == "水"
    assert card["Gloss"] == "water · shuǐ"
    assert card["WordAudio"] == "[sound:word_0000.mp3]"
    assert card["Source"] == "Tatoeba #200 · bob"
    assert card["tags"] == ["swadesh::water", "audio::tts", "tier::core"]
    assert scaffold["Sentence"].count("<b>") == 1
    assert scaffold["tags"][1:] == ["audio::tts", "tier::scaffold"]


def test_media_paths_are_absolute_and_unique(tmp_path: Path) -> None:
    paths = deck.media_paths(["b.mp3", "a.mp3", "b.mp3"], audio_dir=tmp_path / "audio")

    assert [Path(path).name for path in paths] == ["a.mp3", "b.mp3"]
    assert all(Path(path).is_absolute() for path in paths)


def test_load_audio_index_splits_word_and_sentence(tmp_path: Path) -> None:
    write_audio(tmp_path, audio_entries(PACK_ROWS))

    index = deck.load_audio_index(tmp_path / "index.csv")

    assert index.sentences == {"我要喝水。": "tts_0000.mp3", "这场火很大。": "tts_0001.mp3"}
    assert index.words == {"水": "word_0000.mp3", "火": "word_0001.mp3"}
    assert index.voice == VOICE


@pytest.fixture
def pack_env(tmp_path: Path) -> dict[str, Path]:
    write_audio(tmp_path / "audio", audio_entries(PACK_ROWS))
    csv_path = tmp_path / "sentences.csv"
    write_csv(csv_path, SENTENCE_FIELDS, PACK_ROWS)
    word_csv = tmp_path / "words.csv"
    word_csv.write_text("english,mandarin,transliteration\n", encoding="utf-8")
    return {
        "audio_dir": tmp_path / "audio",
        "csv_path": csv_path,
        "word_csv": word_csv,
        "output": tmp_path / "pack.apkg",
        "manifest_path": tmp_path / "manifest.json",
        "attribution_path": tmp_path / "ATTRIBUTION.md",
    }


def build(pack_env: dict[str, Path]) -> dict:
    return shared_pack.build_pack(deck.MANDARIN, lambda **_: PACK_TABLE, **pack_env)


def test_pack_manifest_counts(pack_env: dict[str, Path]) -> None:
    result = build(pack_env)
    manifest = json.loads(pack_env["manifest_path"].read_text(encoding="utf-8"))

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
    assert manifest["voice"] == VOICE
    assert manifest["audio_license_skipped"] == {"empty": 0, "CC BY-NC 4.0": 0}
    source = manifest["audio_license_source"]
    assert set(source) == {"empty", "CC BY-NC 4.0"}
    assert sum(source.values()) > 0
    assert manifest["files"]["source_csv"].startswith("sha256:")
    assert manifest["files"]["sentences_csv"].startswith("sha256:")
    assert manifest["licenses"] == [
        {"data": "Swadesh word data", "license": "CC BY-SA 4.0"},
        {"data": "Tatoeba sentences", "license": "CC BY 2.0 FR"},
        {"data": "Tatoeba audio", "license": "per recording"},
    ]
    assert manifest["excluded_words"] == []

    attribution = pack_env["attribution_path"].read_text(encoding="utf-8")
    assert "Sentences come from Tatoeba (<https://tatoeba.org>)" in attribution
    assert "### Word data" in attribution
    assert "## Mandarin pack" in attribution
    assert "gTTS" in attribution


def test_the_pack_skips_a_card_that_has_no_audio_form_or_bold_target(
    pack_env: dict[str, Path],
) -> None:
    unvoiced = ROWS[0] | {"audio_missing": "true"}
    unknown = ROWS[3] | {"target": "月"}
    no_bold = ROWS[0] | {"sentence": "我要喝。"}
    rows = PACK_ROWS + [unvoiced, unknown, no_bold]
    write_csv(pack_env["csv_path"], SENTENCE_FIELDS, rows)
    write_audio(pack_env["audio_dir"], audio_entries(rows))

    result = build(pack_env)

    assert result["skipped"] == 3
    assert result["no_target"] == 1
    assert result["manifest"]["counts"]["cards"] == 2
    assert result["manifest"]["counts"]["no_target"] == 1


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
    build(pack_env)

    with AnkiPackage(pack_env["output"]) as pkg:
        models = pkg.get_models()
        decks = pkg.get_decks()
        cards = pkg.get_cards()
        parsed = [pkg.parse_card(card, models, decks) for card in cards]
        audio_stats = pkg.get_audio_statistics()

    assert len(cards) == 2
    assert audio_stats["audio_files"] == 4
    assert [item["fields"]["Target"] for item in parsed] == ["水", "火"]
    title = DeckIdentity(pack_env["output"]).title
    assert [item["model"] for item in parsed] == [title, title]
    assert [list(item["fields"]) for item in parsed] == [deck.FIELDS, deck.FIELDS]
    assert parsed[0]["fields"]["Sentence"] == "我要喝<b>水</b>。"
    assert "swadesh::water" in parsed[0]["tags"]
    assert "audio::tts" in parsed[0]["tags"]
    assert "tier::scaffold" in parsed[1]["tags"]
    named = [info for info in decks.values() if info.get("name") == title]
    assert [info.get("desc") for info in named] == [deck.DECK_DESCRIPTION]

    model = next(iter(models.values()))
    assert model.get("sortf") == deck.SORT_FIELD_INDEX
    template = model["tmpls"][0]
    assert template["qfmt"] == deck.FRONT_TEMPLATE
    assert template["afmt"] == f"{deck.BACK_TEMPLATE}\n{load_char_links()}"
