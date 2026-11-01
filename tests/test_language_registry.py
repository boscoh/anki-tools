"""Tests for the one language registry of the ``anki swadesh`` command."""

from __future__ import annotations

import pytest

from anki_tools.languages import (
    ALIASES,
    LANGUAGES,
    PROFILES,
    pipeline_languages,
    profile_languages,
    resolve,
    vocab_languages,
)


def test_profiles_and_cli_entries_agree() -> None:
    assert {language.name for language in profile_languages()} == set(PROFILES)


def test_every_profile_has_a_pipeline_module() -> None:
    assert set(pipeline_languages()) == set(PROFILES)


def test_vocab_list_is_a_subset_of_the_registry() -> None:
    assert set(vocab_languages()) <= set(LANGUAGES)


@pytest.mark.parametrize(
    "name, expected",
    [
        (" yue ", "cantonese"),
        ("cmn", "mandarin"),
        ("zh", "mandarin"),
        ("el", "greek"),
        ("Klingon", "klingon"),
    ],
)
def test_resolve_gives_the_canonical_name_or_folds_to_lowercase(
    name: str, expected: str
) -> None:
    assert resolve(name) == expected


def test_every_alias_points_at_a_registry_name() -> None:
    assert set(ALIASES.values()) <= set(LANGUAGES)


def test_every_language_has_a_unique_display_label() -> None:
    labels = [entry.label for entry in LANGUAGES.values()]
    assert all(labels)
    assert len(set(labels)) == len(labels)
