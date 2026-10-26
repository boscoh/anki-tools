"""Tests for the Mandarin core output note type."""

from __future__ import annotations

from anki_tools.languages.mandarin import core


def test_profile_holds_the_deck_paths():
    """The profile names the Neri deck files and the Mandarin hooks."""
    profile = core.MANDARIN_CORE_PROFILE
    assert profile.input_model_name == "0 2024 Sentences Deck"
    assert profile.input_sentence_field == "Sentence"
    assert profile.input_english_field == "Sentence (Translation)"
    assert profile.input_audio_field == "[a]"
    assert profile.output_model_name == core.DEFAULT_NAME
    assert profile.min_units < profile.max_units
    assert profile.templates
    assert profile.card_css().startswith("/*")
    assert profile.frequency_source == str(core.FREQUENCY_PATH)


def test_the_templates_read_the_output_audio_field():
    """The output note type holds its audio in the ``Audio`` field."""
    for template in (core.CORE_FRONT_TEMPLATE, core.CORE_BACK_TEMPLATE):
        assert "{[" not in template
        assert "{{Audio}}" in template
