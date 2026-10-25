"""Tests for the one language registry of the ``anki swadesh`` command."""

from __future__ import annotations

from anki_tools.languages import (
    ALIASES,
    LANGUAGES,
    PROFILES,
    pipeline_languages,
    profile_languages,
    resolve,
    swadesh_languages,
)


def test_profiles_and_cli_entries_agree() -> None:
    assert {language.name for language in profile_languages()} == set(PROFILES)


def test_every_profile_has_a_pipeline_module() -> None:
    assert set(pipeline_languages()) == set(PROFILES)


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


def test_alias_gives_the_canonical_name() -> None:
    assert resolve(" yue ") == "cantonese"
    assert resolve("cmn") == "mandarin"
    assert resolve("zh") == "mandarin"
    assert resolve("el") == "greek"


def test_unknown_name_folds_to_lowercase() -> None:
    assert resolve("Klingon") == "klingon"


def test_every_alias_points_at_a_registry_name() -> None:
    assert set(ALIASES.values()) <= set(LANGUAGES)


def test_only_four_languages_have_a_core_module() -> None:
    assert {name for name, entry in LANGUAGES.items() if entry.core} == {
        "mandarin",
        "french",
        "spanish",
        "cantonese",
    }
