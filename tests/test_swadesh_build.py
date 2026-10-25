"""Tests for the Swadesh deck build helpers."""

from pathlib import Path

import pytest

from anki_tools import cli
from anki_tools.cli import _content_field_order, _prune_outdated_audio
from anki_tools.sentence_pack import tatoeba as sentence_tatoeba


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


def test_core_command_reports_a_language_without_a_core_module(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.core("greek", tmp_path / "absent.apkg")

    out = capsys.readouterr().out
    assert "Unsupported language: greek" in out
    assert "Supported: mandarin, french, spanish, cantonese." in out


def test_fetch_command_resolves_the_language_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    def fake_fetch_tatoeba(files: dict[str, str], **_kwargs: object) -> dict:
        captured.append(dict(files))
        return {}

    monkeypatch.setattr(sentence_tatoeba, "fetch_tatoeba", fake_fetch_tatoeba)

    cli.fetch("cantonese")
    assert any(name.startswith("yue_") for name in captured[0])
    assert not any(name.startswith("cmn_") for name in captured[0])

    cli.fetch()
    assert any(name.startswith("cmn_") for name in captured[1])
