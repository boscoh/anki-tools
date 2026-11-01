"""Tests for the Swadesh deck build helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools import cli, swadesh_vocab
from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.cantonese.profile import CANTONESE
from anki_tools.sentence_pack import tatoeba as sentence_tatoeba
from anki_tools.swadesh_vocab import _content_field_order
from tests.deck_fixture import MP3_STUB

DECK = "Swadesh Vocab German"


@pytest.fixture
def vocab_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run ``build_vocab_deck`` in a temporary root with a two-row German list."""
    (tmp_path / "vocab").mkdir()
    (tmp_path / "vocab" / "english-german.csv").write_text(
        "english,german\ncat,Katze\ndog,Hund\n", encoding="utf-8"
    )
    audio_dir = tmp_path / "audio" / "audio_german"
    audio_dir.mkdir(parents=True)
    monkeypatch.setattr(swadesh_vocab, "ROOT", tmp_path)
    monkeypatch.setattr(swadesh_vocab.time, "sleep", lambda _seconds: None)
    monkeypatch.chdir(tmp_path)
    return audio_dir


@pytest.fixture
def gtts(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the gTTS network call. Set ``fail`` to make every call raise."""
    state = SimpleNamespace(spoken=[], fail=False)

    class FakeGtts:
        def __init__(self, text: str, lang: str, slow: bool) -> None:
            if state.fail:
                raise RuntimeError("tts down")
            state.spoken.append(text)

        def save(self, dest: str) -> None:
            Path(dest).write_bytes(MP3_STUB)

    monkeypatch.setattr(swadesh_vocab, "gTTS", FakeGtts)
    return state


def read_deck(path: Path) -> tuple[int, set[str], list[str]]:
    with AnkiPackage(path) as package:
        names = [deck["name"] for deck in package.get_decks().values()]
        return (
            len(package.get_notes()),
            set(package.get_media_mapping().values()),
            names,
        )


def test_the_vocab_deck_holds_a_card_and_audio_for_every_csv_row(
    vocab_dir: Path, gtts: SimpleNamespace
) -> None:
    for name in ("Katze.mp3", "Hund.mp3", "stale.mp3"):
        (vocab_dir / name).write_bytes(MP3_STUB)

    files = swadesh_vocab.build_vocab_deck("german")

    assert files is not None
    notes, media, decks = read_deck(files.output_path)
    assert (notes, media) == (2, {"Katze.mp3", "Hund.mp3"})
    assert DECK in decks
    assert gtts.spoken == []
    assert sorted(path.name for path in vocab_dir.iterdir()) == ["Hund.mp3", "Katze.mp3"]


def test_a_form_without_cached_audio_is_spoken_by_gtts(
    vocab_dir: Path, gtts: SimpleNamespace
) -> None:
    (vocab_dir / "Katze.mp3").write_bytes(MP3_STUB)

    files = swadesh_vocab.build_vocab_deck("german")

    assert files is not None
    assert read_deck(files.output_path)[:2] == (2, {"Katze.mp3", "Hund.mp3"})
    assert gtts.spoken == ["Hund"]


def test_a_form_without_audio_stops_the_build(
    vocab_dir: Path, gtts: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    gtts.fail = True

    assert swadesh_vocab.build_vocab_deck("german") is None

    assert "2 forms have no audio: cat, dog" in capsys.readouterr().out
    assert not Path("apkg").exists()


@pytest.mark.parametrize(
    ("fields", "audio_field", "expected"),
    [
        pytest.param(
            ["swadesh_index", "english", "french", "gender", "form_kind", "zipf", "tier"],
            "french",
            ["french", "english"],
            id="review columns stay out",
        ),
        pytest.param(
            ["english", "mandarin", "transliteration"],
            "mandarin",
            ["mandarin", "transliteration", "english"],
            id="transliteration stays",
        ),
    ],
)
def test_content_field_order_keeps_only_card_content(
    fields: list[str], audio_field: str, expected: list[str]
) -> None:
    assert _content_field_order(fields, audio_field) == expected


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
