"""Tests for the Mandarin core output note type."""

from __future__ import annotations

from anki_tools.languages.mandarin import core
from anki_tools.ranking import coreflow


def test_output_values_copies_the_pinyin_from_the_latin_field():
    values = coreflow.output_values(
        core.MANDARIN_CORE_PROFILE,
        ["Sentence", "Sentence (Latin)", "Sentence (Translation)", "[a]"],
        ["从", "Cóng", "From.", "[sound:a.mp3]"],
        "从",
    )

    assert values == ["[sound:a.mp3]", "从", "From.", "Cóng", "", ""]
