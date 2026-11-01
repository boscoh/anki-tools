"""Tests for the browser card preview builder."""

from pathlib import Path

import pytest

from anki_tools.languages import LANGUAGES, profile_languages, vocab_languages
from anki_tools.preview import (
    SENTENCE_LANGUAGES,
    VOCAB_DIR,
    VOCAB_LANGUAGES,
    render_template,
    sentence_card,
    write_preview,
)


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
