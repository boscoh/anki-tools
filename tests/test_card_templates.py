"""Tests that the deck builders read their card templates from files."""

from anki_tools.apkg.templates import DEFAULT_TEMPLATE_DIR, TEMPLATES_DIR, load_text
from anki_tools.languages import PROFILES
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.sentence_pack import pack as sentence_pack

PACK_PROFILES = tuple(PROFILES[name] for name in sorted(PROFILES))


def test_the_default_style_files_exist() -> None:
    for filename in ("card.css", "front.html", "back.html"):
        assert (DEFAULT_TEMPLATE_DIR / filename).is_file(), filename


def test_the_default_paths_point_at_the_default_dir() -> None:
    assert sentence_pack.CSS_PATH == DEFAULT_TEMPLATE_DIR / "card.css"
    assert french_cull.CARD_CSS_PATH == DEFAULT_TEMPLATE_DIR / "card.css"


def test_each_pack_has_the_three_template_files() -> None:
    for profile in PACK_PROFILES:
        directory = TEMPLATES_DIR / "sentence_pack" / profile.name
        for filename in ("front.html", "back.html", "card.css"):
            assert (directory / filename).is_file(), f"{profile.name}/{filename}"


def test_each_pack_profile_reads_the_template_files() -> None:
    for profile in PACK_PROFILES:
        directory = TEMPLATES_DIR / "sentence_pack" / profile.name
        assert profile.front_template == load_text(directory / "front.html")
        assert profile.back_template == load_text(directory / "back.html")
        assert profile.extra_css == load_text(directory / "card.css")


def test_the_cull_templates_come_from_files() -> None:
    directory = TEMPLATES_DIR / "french_cull"
    assert french_cull.FRONT_TEMPLATE == load_text(directory / "front.html")
    assert french_cull.BACK_TEMPLATE == load_text(directory / "back.html")


def test_the_core_templates_come_from_files() -> None:
    directory = TEMPLATES_DIR / "french_core"
    assert french_core.CORE_FRONT_TEMPLATE == load_text(directory / "front.html")
    assert french_core.CORE_BACK_TEMPLATE == load_text(directory / "back.html")


def test_the_vocab_templates_hold_the_field_placeholder() -> None:
    directory = TEMPLATES_DIR / "vocab"
    assert "{{audio}}" in load_text(directory / "front.html")
    assert "{{fields}}" in load_text(directory / "back.html")
