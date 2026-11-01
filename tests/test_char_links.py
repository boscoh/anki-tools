"""Tests for the shared CC-Canto character-link script."""

import pytest

from anki_tools.apkg.templates import load_char_links
from anki_tools.languages.cantonese.core import CANTONESE_CORE_PROFILE
from anki_tools.languages.mandarin.core import MANDARIN_CORE_PROFILE
from anki_tools.ranking.assemble import deck_templates


def test_missing_file_gives_empty_script(tmp_path) -> None:
    assert load_char_links(tmp_path / "absent.html") == ""


@pytest.mark.parametrize(
    "profile", [MANDARIN_CORE_PROFILE, CANTONESE_CORE_PROFILE], ids=["mandarin", "cantonese"]
)
def test_the_core_deck_templates_hold_the_script(profile) -> None:
    script = load_char_links()

    assert script
    for _front, back in deck_templates(profile):
        assert back.endswith(script)
