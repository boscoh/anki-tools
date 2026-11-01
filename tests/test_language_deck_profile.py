"""One test that pins the agreement of the two mode views of one language."""

from __future__ import annotations

import pytest

from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.languages.french.profile import FRENCH
from anki_tools.languages.spanish import core as spanish_core
from anki_tools.languages.spanish import cull as spanish_cull

VIEWS = [
    pytest.param(
        french_core.FRENCH_CORE_PROFILE,
        french_cull.FRENCH_CULL_PROFILE,
        id="french",
    ),
    pytest.param(
        spanish_core.SPANISH_CORE_PROFILE,
        spanish_cull.SPANISH_CULL_PROFILE,
        id="spanish",
    ),
]


@pytest.mark.parametrize("core_view, cull_view", VIEWS)
def test_the_two_mode_views_agree_on_the_language_facts(core_view, cull_view) -> None:
    assert core_view.input_model_name == cull_view.input_model_name
    assert core_view.input_sentence_field == cull_view.input_sentence_field
    assert core_view.input_english_field == cull_view.input_english_field
    assert core_view.input_audio_field == cull_view.input_audio_field
    assert core_view.romanization_source == cull_view.romanization_source
    assert core_view.sense_path == cull_view.sense_path
    assert core_view.card_css is cull_view.card_css
    assert core_view.highlight is cull_view.highlight
    assert core_view.unit_count is cull_view.unit_count
    assert (core_view.min_units, core_view.max_units) == (
        cull_view.min_units,
        cull_view.max_units,
    )
    assert core_view.english_bounds == cull_view.english_bounds


def test_the_views_use_the_pack_window_and_unit_count() -> None:
    assert french_core.FRENCH_CORE_PROFILE.unit_count is FRENCH.unit_count
    assert (
        french_cull.FRENCH_CULL_PROFILE.min_units,
        french_cull.FRENCH_CULL_PROFILE.max_units,
    ) == (FRENCH.min_units, FRENCH.max_units)


def test_the_french_modes_share_one_frequency_loader() -> None:
    assert (
        french_core.FRENCH_CORE_PROFILE.load_frequency
        is french_cull.FRENCH_CULL_PROFILE.load_frequency
    )
    assert (
        french_core.FRENCH_CORE_PROFILE.read_names
        is french_cull.FRENCH_CULL_PROFILE.read_names
    )
