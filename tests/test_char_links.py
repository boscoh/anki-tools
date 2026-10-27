"""Tests for the shared CC-Canto character-link script."""

import pytest

from anki_tools.languages.cantonese.profile import CANTONESE
from anki_tools.languages.mandarin.profile import MANDARIN
from anki_tools.apkg.templates import CHAR_LINKS_PATH, load_char_links


def test_script_links_to_cc_canto() -> None:
    script = load_char_links()
    assert "cantonese.org" in script
    assert ".char-links" in script
    assert "<script>" in script


def test_missing_file_gives_empty_script(tmp_path) -> None:
    assert load_char_links(tmp_path / "absent.html") == ""


@pytest.mark.parametrize("profile", [MANDARIN, CANTONESE])
def test_the_profile_uses_the_script(profile) -> None:
    assert profile.char_links is True
    assert "char-links" in profile.back_template
    assert CHAR_LINKS_PATH.exists()
