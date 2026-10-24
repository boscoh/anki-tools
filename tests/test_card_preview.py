"""Tests for the browser card preview builder."""

from pathlib import Path

from anki_tools.preview import build_sections, render_template, write_preview


def test_write_preview_writes_every_section(tmp_path: Path) -> None:
    output = tmp_path / "cards.html"

    cards, sections = write_preview(output)

    text = output.read_text(encoding="utf-8")
    assert cards == 16
    assert sections == 5
    assert "Anki card previews" in text
    for title in (
        "Default deck style",
        "Swadesh sentence packs",
        "French 1000-card cull",
        "French core-word deck",
        "Swadesh vocabulary decks",
    ):
        assert title in text


def test_vocabulary_cards_hold_only_card_content() -> None:
    vocab = next(section for section in build_sections() if section.anchor == "vocab")
    by_title = {card.title: card for card in vocab.cards}

    assert by_title["French vocabulary"].fields == "audio, french, english"
    assert by_title["Greek vocabulary"].fields == "audio, greek, english"


def test_render_template_hides_and_shows_sections() -> None:
    template = "{{#Tag}}yes{{/Tag}}{{^Tag}}no{{/Tag}}"

    assert render_template(template, {"Tag": "x"}) == "yes"
    assert render_template(template, {"Tag": ""}) == "no"


def test_render_template_handles_filters_and_triple_braces() -> None:
    template = "{{text:Front}} {{{Back}}} {{! note}}"

    result = render_template(template, {"Front": "a", "Back": "b"})

    assert result == "a b "
