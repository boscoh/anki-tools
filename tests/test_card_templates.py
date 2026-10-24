"""Tests that the deck builders read their card templates from files."""

import re

from anki_tools.apkg.templates import (
    DEFAULT_TEMPLATE_DIR,
    SHARED_CSS_PATH,
    TEMPLATES_DIR,
    load_css_layers,
    load_text,
)
from anki_tools.languages import PROFILES
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.sentence_pack import pack as sentence_pack

PACK_PROFILES = tuple(PROFILES[name] for name in sorted(PROFILES))
CLASS_ATTRIBUTE = re.compile(r'class="([^"]+)"')
CSS_SELECTOR = re.compile(r"([^{}]+)\x7b")
CSS_CLASS = re.compile(r"\.([A-Za-z][A-Za-z0-9_-]*)")
SCRIPT_HOOK_CLASSES = {"char-links"}
SHARED_CLASSES = {
    "card",
    "front-audio",
    "hint",
    "sentence",
    "english",
    "target",
    "source",
    "romanization",
}


def test_the_shared_files_exist() -> None:
    assert SHARED_CSS_PATH.is_file()
    for filename in ("front.html", "back.html"):
        assert (DEFAULT_TEMPLATE_DIR / filename).is_file(), filename


def test_the_default_css_layers_start_with_the_shared_sheet() -> None:
    assert sentence_pack.CSS_PATH == SHARED_CSS_PATH
    assert french_cull.CARD_CSS_PATH == SHARED_CSS_PATH


def _template_classes() -> set[str]:
    used: set[str] = set()
    for path in TEMPLATES_DIR.rglob("*.html"):
        for names in CLASS_ATTRIBUTE.findall(path.read_text(encoding="utf-8")):
            used.update(names.split())
    return used


def _css_classes() -> set[str]:
    defined: set[str] = set()
    for path in TEMPLATES_DIR.rglob("*.css"):
        text = path.read_text(encoding="utf-8")
        for selector in CSS_SELECTOR.findall(text):
            defined.update(CSS_CLASS.findall(selector))
    return defined


def test_the_shared_sheet_holds_the_common_classes() -> None:
    css = load_text(SHARED_CSS_PATH)
    selectors = "\n".join(CSS_SELECTOR.findall(css))
    assert SHARED_CLASSES <= set(CSS_CLASS.findall(selectors))


def test_every_template_class_has_a_css_rule() -> None:
    missing = _template_classes() - _css_classes() - SCRIPT_HOOK_CLASSES
    assert missing == set()


def test_the_shared_sheet_comes_first() -> None:
    pack_css = TEMPLATES_DIR / "sentence_pack" / "mandarin" / "card.css"
    css = load_css_layers(SHARED_CSS_PATH, pack_css)
    assert css.index(".card") < css.index(".romanization")


def test_the_character_links_keep_the_card_color() -> None:
    assert ".char-links a" in load_text(SHARED_CSS_PATH)
    assert "color: inherit" in load_text(SHARED_CSS_PATH)


def test_the_character_font_stack_is_simplified_by_default() -> None:
    assert "Noto Sans SC" in load_text(SHARED_CSS_PATH)


def test_the_cantonese_decks_use_a_traditional_font_stack() -> None:
    for path in (
        TEMPLATES_DIR / "sentence_pack" / "cantonese" / "card.css",
        TEMPLATES_DIR / "vocab" / "cantonese" / "card.css",
    ):
        assert "Noto Sans HK" in load_text(path), path


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


def test_the_cards_align_the_text_to_the_reading_side() -> None:
    shared = load_text(SHARED_CSS_PATH)
    assert "text-align: left" in shared
    centred = [
        path.relative_to(TEMPLATES_DIR)
        for path in TEMPLATES_DIR.rglob("*.css")
        if "text-align: center" in load_text(path)
    ]

    assert centred == []


def test_the_arabic_deck_aligns_the_text_to_the_right() -> None:
    arabic = load_text(TEMPLATES_DIR / "vocab" / "arabic" / "card.css")

    assert "direction: rtl" in arabic
    assert "text-align: right" in arabic
