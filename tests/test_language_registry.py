"""Tests for the one language registry of the ``anki swadesh`` command."""

from __future__ import annotations

from anki_tools.sentence_pack.languages import (
    LANGUAGES,
    PROFILES,
    pipeline_languages,
    profile_languages,
    swadesh_languages,
)


def test_profiles_and_cli_entries_agree() -> None:
    assert {language.name for language in profile_languages()} == set(PROFILES)


def test_every_profile_has_a_pipeline_module_except_mandarin() -> None:
    assert set(pipeline_languages()) == set(PROFILES) - {"mandarin"}


def test_build_list_matches_the_vocabulary_languages() -> None:
    assert swadesh_languages() == (
        "mandarin",
        "spanish",
        "greek",
        "german",
        "cantonese",
        "arabic",
        "hindi",
    )


def test_build_list_is_a_subset_of_the_registry() -> None:
    assert set(swadesh_languages()) <= set(LANGUAGES)
