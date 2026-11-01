"""Tests that the deck builders read their card templates from files."""

import re
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import (
    DEFAULT_TEMPLATE_DIR,
    SHARED_CSS_PATH,
    TemplateSpec,
    assemble_templates,
    apply_field_mapping,
    apply_templates,
    find_best_match,
    load_char_links,
    load_text,
)
from anki_tools.languages import PROFILES
from tests.deck_fixture import build_contract_deck

PACK_PROFILES = tuple(PROFILES[name] for name in sorted(PROFILES))
CLASS_ATTRIBUTE = re.compile(r'class="([^"]+)"')
CSS_SELECTOR = re.compile(r"([^{}]+)\x7b")
CSS_CLASS = re.compile(r"\.([A-Za-z][A-Za-z0-9_-]*)")
SCRIPT_HOOK_CLASSES = {"char-links"}


@pytest.mark.parametrize("profile", PACK_PROFILES, ids=lambda profile: profile.name)
def test_a_pack_card_styles_its_classes_from_the_shared_sheet_and_its_own(
    profile,
) -> None:
    used: set[str] = set()
    for names in CLASS_ATTRIBUTE.findall(profile.front_template + profile.back_template):
        used.update(names.split())
    css = load_text(SHARED_CSS_PATH) + profile.extra_css
    defined = {name for selector in CSS_SELECTOR.findall(css) for name in CSS_CLASS.findall(selector)}

    assert used - defined - SCRIPT_HOOK_CLASSES == set()


def test_apply_templates_writes_the_sheet_and_the_card(tmp_path: Path) -> None:
    path = build_contract_deck(tmp_path)

    models, applied = apply_templates(path, verbose=False)

    assert models == 1
    assert applied == ["CSS", "front", "back"]
    with AnkiPackage(path) as package:
        model = next(iter(package.get_models(include_templates=True).values()))
    assert model["css"].startswith(load_text(SHARED_CSS_PATH))
    template = model["tmpls"][0]
    assert template["qfmt"] == load_text(DEFAULT_TEMPLATE_DIR / "front.html")
    assert template["afmt"].startswith(
        load_text(DEFAULT_TEMPLATE_DIR / "back.html")
    )
    assert template["afmt"].endswith(load_char_links())


def test_the_mapper_takes_the_nearest_field_name() -> None:
    deck_fields = {"Annotated Sentence", "Translated English"}
    mapping = {
        name: find_best_match(name, deck_fields)
        for name in ("Sentence", "Translation", "Audio")
    }

    assert mapping == {
        "Sentence": "Annotated Sentence",
        "Translation": None,
        "Audio": None,
    }
    assert apply_field_mapping("{{Translation}}", {"Translation": "English"}) == (
        "{{English}}"
    )


ROMANIZED_BACK = (
    "{{Audio}}<hr id=answer>{{Sentence}}"
    "{{#Romanization}}<div>{{Romanization}}</div>{{/Romanization}}"
)


@pytest.mark.parametrize(
    "fields, block_kept",
    [
        pytest.param(("Audio", "Sentence"), False, id="no-field-drops-the-block"),
        pytest.param(
            ("Audio", "Sentence", "Romanization"), True, id="field-keeps-the-block"
        ),
    ],
)
def test_the_assembly_keeps_an_optional_block_only_with_its_field(
    fields: tuple[str, ...], block_kept: bool
) -> None:
    pair = assemble_templates(
        TemplateSpec(
            front="{{Audio}}",
            back=ROMANIZED_BACK,
            fields=fields,
            optional_blocks=("Romanization",),
        )
    )

    assert ("{{Romanization}}" in pair.back) is block_kept
    assert pair.back.startswith("{{Audio}}<hr id=answer>{{Sentence}}")


def test_the_assembly_adds_the_script_only_when_the_hook_is_present() -> None:
    hooked = '<div class="char-links">b</div>'
    off = assemble_templates(TemplateSpec(front="f", back="b"))
    on = assemble_templates(TemplateSpec(front="f", back=hooked))

    assert off.back == "b"
    assert on.back == f"{hooked}\n{load_char_links()}"


def test_the_assembly_joins_the_css_layers_in_order() -> None:
    pair = assemble_templates(
        TemplateSpec(front="f", back="b", css_layers=(".card {}", ".target {}"))
    )

    assert pair.css == ".card {}\n.target {}"
