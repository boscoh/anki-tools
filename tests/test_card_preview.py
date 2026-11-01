"""Tests for the browser card preview builder."""

from pathlib import Path

import pytest

from anki_tools.languages import LANGUAGES, profile_languages, vocab_languages
from anki_tools.preview import (
    SENTENCE_LANGUAGES,
    VOCAB_DIR,
    VOCAB_LANGUAGES,
    PreviewSection,
    build_page,
    default_card,
    fix_card,
    render_template,
    sentence_card,
    write_preview,
)


def test_the_page_holds_each_section_and_card() -> None:
    sections = (
        PreviewSection("default", "Default deck style", "intro", (default_card(),)),
        PreviewSection(
            "core",
            "Core-word decks",
            "intro",
            (fix_card("french", "core", "French core-word deck"),),
        ),
    )

    page = build_page(sections, title="Previews")

    assert "<title>Previews</title>" in page
    for section in sections:
        assert f'href="#{section.anchor}"' in page
        assert section.title in page
        for card in section.cards:
            assert card.title in page


@pytest.mark.parametrize(
    "language, mode, shown",
    [
        pytest.param(
            "french", "cull", ["I won't ask you anything else today.", "not"], id="french-cull"
        ),
        pytest.param(
            "spanish", "cull", ["I want to ask for the bill.", "to ask"], id="spanish-cull"
        ),
        pytest.param(
            "french", "core", ["I won't ask you anything else today.", "not"], id="french-core"
        ),
        pytest.param(
            "spanish", "core", ["I want to ask for the bill.", "to ask"], id="spanish-core"
        ),
        pytest.param(
            "mandarin",
            "core",
            ["I am very busy today.", "wǒ jīn tiān hěn máng", "busy"],
            id="mandarin-core",
        ),
        pytest.param(
            "cantonese",
            "core",
            ["I am very busy today.", "ngo5 gam1 jat6 hou2 mong4", "busy"],
            id="cantonese-core",
        ),
    ],
)
def test_a_fix_card_shows_the_output_fields(
    language: str, mode: str, shown: list[str]
) -> None:
    card = fix_card(language, mode, "title")

    assert card.front.strip()
    assert "<b>" in card.back
    for text in shown:
        assert text in card.back, text
    for name in ("Sentence", "Core", "CoreGloss"):
        assert name in card.fields


@pytest.mark.parametrize(
    "language, sentence, target, bold",
    [
        ("french", "Je suis ici.", "Je", "<b>Je</b>"),
        ("spanish", "Yo estoy aquí.", "Yo", "<b>Yo</b>"),
        ("greek", "εσύ είσαι εδώ.", "εσύ", "<b>εσύ</b>"),
    ],
)
def test_the_sentence_pack_card_bolds_the_target(
    language: str, sentence: str, target: str, bold: str
) -> None:
    row = {
        "sentence": sentence,
        "target": target,
        "english": "You are here.",
        "tatoeba_id": "1",
        "tatoeba_author": "someone",
    }

    assert bold in sentence_card(language, "title", row).back


@pytest.mark.parametrize(
    "template, fields, expected",
    [
        ("{{#Tag}}yes{{/Tag}}{{^Tag}}no{{/Tag}}", {"Tag": "x"}, "yes"),
        ("{{#Tag}}yes{{/Tag}}{{^Tag}}no{{/Tag}}", {"Tag": ""}, "no"),
        ("{{text:Front}} {{{Back}}} {{! note}}", {"Front": "a", "Back": "b"}, "a b "),
    ],
    ids=["section-shown", "section-hidden", "filter-triple-brace-and-comment"],
)
def test_render_template_fills_the_mustache_tags(
    template: str, fields: dict[str, str], expected: str
) -> None:
    assert render_template(template, fields) == expected


def test_the_language_lists_follow_the_registry() -> None:
    sentence_names = [name for name, _ in SENTENCE_LANGUAGES]
    vocab_names = [name for name, _ in VOCAB_LANGUAGES]

    for language in profile_languages():
        assert language.name in sentence_names
        assert dict(SENTENCE_LANGUAGES)[language.name] == language.label
    for name in vocab_languages():
        assert name in vocab_names
        assert dict(VOCAB_LANGUAGES)[name] == LANGUAGES[name].label
    assert "french" not in vocab_names


def test_the_written_page_holds_a_panel_for_each_card_that_the_vocab_files_allow(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preview" / "cards.html"

    cards, sections = write_preview(output)

    page = output.read_text(encoding="utf-8")
    assert page.count('class="card-panel"') == cards
    assert page.count("<section id=") == sections
    for name, label in VOCAB_LANGUAGES:
        has_file = (VOCAB_DIR / f"english-{name}.csv").exists()
        assert (f"{label} vocabulary</h3>" in page) is has_file, name
    for name, label in SENTENCE_LANGUAGES:
        has_file = (VOCAB_DIR / f"{name}_sentences.csv").exists()
        assert (f"{label} sentence</h3>" in page) is has_file, name
