"""Tests for the shared CC-Canto character-link script."""

from anki_tools.sentence_pack.languages.cantonese import CANTONESE
from anki_tools.sentence_pack.languages.mandarin import MANDARIN
from anki_tools.apkg.templates import CHAR_LINKS_PATH, load_char_links


def test_script_links_to_cc_canto() -> None:
    script = load_char_links()
    assert "cantonese.org" in script
    assert ".char-links" in script
    assert "<script>" in script


def test_missing_file_gives_empty_script(tmp_path) -> None:
    assert load_char_links(tmp_path / "absent.html") == ""


def test_mandarin_profile_uses_the_script() -> None:
    assert MANDARIN.char_links is True
    assert "char-links" in MANDARIN.back_template
    assert CHAR_LINKS_PATH.exists()


def test_cantonese_profile_uses_the_script() -> None:
    assert CANTONESE.char_links is True
    assert "char-links" in CANTONESE.back_template
