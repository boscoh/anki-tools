"""Tests for the package card preview builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.preview_apkg import write_apkg_preview
from tests.deck_fixture import build_contract_deck


@pytest.fixture(scope="module")
def deck_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_contract_deck(tmp_path_factory.mktemp("preview"))


def test_preview_writes_every_card(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    result = write_apkg_preview(deck_path, output, limit=0)

    assert result.cards == 7
    assert result.total == 7
    assert result.sections == 1
    assert result.media == 5
    text = output.read_text(encoding="utf-8")
    assert "Anki card previews" in text
    assert "第1句话" in text
    assert "cards_media/note1.mp3" in text


def test_preview_extracts_the_used_media(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    write_apkg_preview(deck_path, output, limit=0)

    media = tmp_path / "cards_media"
    assert sorted(path.name for path in media.iterdir()) == [
        "note1.mp3",
        "note2.mp3",
        "note3.mp3",
        "note4.mp3",
        "note5.mp3",
    ]


def test_limit_selects_the_first_cards(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    result = write_apkg_preview(deck_path, output, limit=2)

    assert result.cards == 2
    assert result.total == 7
    text = output.read_text(encoding="utf-8")
    assert "Showing 2 of 7 cards" in text


def test_search_selects_one_note(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    result = write_apkg_preview(deck_path, output, search="第3句话")

    assert result.cards == 1
    text = output.read_text(encoding="utf-8")
    assert "第3句话" in text
    assert "第1句话" not in text


def test_no_media_keeps_the_placeholders(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    result = write_apkg_preview(deck_path, output, limit=1, extract_media=False)

    assert result.media == 0
    assert not (tmp_path / "cards_media").exists()
    text = output.read_text(encoding="utf-8")
    assert "note1.mp3" in text
    assert "preview-audio" in text


def test_custom_media_directory_sets_the_url(deck_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cards.html"
    media = tmp_path / "assets"

    result = write_apkg_preview(deck_path, output, limit=1, media_dir=media)

    assert result.media == 1
    assert (media / "note1.mp3").exists()
    assert "assets/note1.mp3" in output.read_text(encoding="utf-8")
