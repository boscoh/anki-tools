"""Tests for the Mandarin core output note type."""

from __future__ import annotations

from anki_tools.languages.mandarin import core
from anki_tools.ranking import coreflow


def test_the_templates_read_the_output_audio_field():
    for template in (core.CORE_FRONT_TEMPLATE, core.CORE_BACK_TEMPLATE):
        assert "{[" not in template
        assert "{{Audio}}" in template
    assert "{{Romanization}}" in core.CORE_BACK_TEMPLATE


def test_output_values_copies_the_pinyin_from_the_latin_field():
    values = coreflow.output_values(
        core.MANDARIN_CORE_PROFILE,
        ["Sentence", "Sentence (Latin)", "Sentence (Translation)", "[a]"],
        ["从", "Cóng", "From.", "[sound:a.mp3]"],
        "从",
    )

    assert values == ["[sound:a.mp3]", "从", "From.", "Cóng", "", ""]
