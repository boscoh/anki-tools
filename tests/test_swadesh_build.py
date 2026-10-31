"""Tests for the Swadesh deck build helpers."""

import csv
from pathlib import Path

import pytest

from anki_tools import cli, swadesh_vocab
from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.swadesh_vocab import _content_field_order, _prune_outdated_audio
from anki_tools.languages.cantonese.profile import CANTONESE
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
        swadesh_vocab,
        "_parse_vocab_entries",
        lambda _config: [{"english": "I", "french": "je"}],
    )
    monkeypatch.setattr(swadesh_vocab, "_generate_audio_files", lambda *_args: None)
    monkeypatch.setattr(swadesh_vocab, "_build_layout", lambda *_args: object())
    monkeypatch.setattr(
        swadesh_vocab,
        "_pack_apkg",
        lambda config, *_args: captured.update(config),
    )

    swadesh_vocab.build_vocab_deck("french")

    identity = captured["identity"]
    assert identity.title == "Swadesh Vocab French"
    assert identity.model_name == identity.title


def test_the_swadesh_help_lists_the_languages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.app(["swadesh", "vocab", "--help"])
    vocab_help = capsys.readouterr().out
    for name in cli.VOCAB_LANGUAGES:
        assert name in vocab_help

    with pytest.raises(SystemExit):
        cli.app(["swadesh", "pack", "--help"])
    pack_help = capsys.readouterr().out
    for language in cli.profile_languages():
        assert language.name in pack_help


def test_fetch_language_uses_the_profile_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    def fake_fetch_tatoeba(files: dict[str, str], **_kwargs: object) -> dict:
        captured.append(dict(files))
        return {}

    monkeypatch.setattr(sentence_tatoeba, "fetch_tatoeba", fake_fetch_tatoeba)

    cli._fetch_language(CANTONESE, refresh=False)
    assert any(name.startswith("yue_") for name in captured[0])
    assert any(name.startswith("eng_") for name in captured[0])
    assert not any(name.startswith("cmn_") for name in captured[0])


def test_the_vocab_deck_holds_a_card_and_audio_for_every_csv_row(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "english-french.csv"
    csv_path.write_text(
        "english,french\ncat,chat\ndog,chien\n", encoding="utf-8"
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ("chat.mp3", "chien.mp3"):
        (audio_dir / name).write_bytes(b"")
    config = {
        "language": "French",
        "vocab_file": str(csv_path),
        "audio_dir": str(audio_dir),
        "identity": DeckIdentity(tmp_path / "vocab.apkg"),
        "audio_field_name": "french",
    }

    entries = swadesh_vocab._parse_vocab_entries(config)
    layout = swadesh_vocab._build_layout(config, entries)
    files = swadesh_vocab._pack_apkg(config, entries, layout, prune=True)

    assert files is not None
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with AnkiPackage(files.output_path) as package:
        notes = package.get_notes()
        media = set(package.get_media_mapping().values())

    assert len(notes) == len(rows)
    assert media == {entry["audio_filename"] for entry in entries}
