"""Tests for the Swadesh deck build helpers."""

from pathlib import Path

import pytest

from anki_tools import cli
from anki_tools.cli import _content_field_order, _prune_outdated_audio


def test_prune_removes_files_not_in_keep(tmp_path: Path) -> None:
    (tmp_path / "keep.mp3").write_bytes(b"a")
    (tmp_path / "stale.mp3").write_bytes(b"b")
    (tmp_path / "old_form.mp3").write_bytes(b"c")

    removed = _prune_outdated_audio(tmp_path, {"keep.mp3"})

    assert sorted(removed) == ["old_form.mp3", "stale.mp3"]
    assert [p.name for p in tmp_path.iterdir()] == ["keep.mp3"]


def test_prune_keeps_every_listed_file(tmp_path: Path) -> None:
    (tmp_path / "a.mp3").write_bytes(b"a")
    (tmp_path / "b.mp3").write_bytes(b"b")

    removed = _prune_outdated_audio(tmp_path, {"a.mp3", "b.mp3"})

    assert removed == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp3", "b.mp3"]


def test_prune_absent_dir_returns_empty(tmp_path: Path) -> None:
    assert _prune_outdated_audio(tmp_path / "absent", set()) == []


def test_content_field_order_keeps_only_card_content() -> None:
    fields = [
        "swadesh_index",
        "english",
        "french",
        "gender",
        "variant",
        "form_kind",
        "zipf",
        "candidate_count",
        "tier",
    ]

    assert _content_field_order(fields, "french") == ["french", "english"]


def test_content_field_order_keeps_the_transliteration() -> None:
    fields = ["english", "mandarin", "transliteration"]

    assert _content_field_order(fields, "mandarin") == [
        "mandarin",
        "transliteration",
        "english",
    ]


def test_vocab_deck_and_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        cli, "_parse_vocab_entries", lambda _config: [{"english": "I", "french": "je"}]
    )
    monkeypatch.setattr(cli, "_generate_audio_files", lambda *_args: None)
    monkeypatch.setattr(cli, "_build_layout", lambda *_args: object())
    monkeypatch.setattr(
        cli, "_pack_apkg", lambda config, *_args: captured.update(config)
    )

    cli._build_apkg("french")

    assert captured["deck_name"] == "Swadesh French Vocabulary"
    assert captured["model_name"] == "Swadesh French Vocabulary"
