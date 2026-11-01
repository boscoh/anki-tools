"""Tests for the one language registry of the ``anki swadesh`` command."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

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


def test_importing_the_cli_loads_no_language_package_or_heavy_dependency() -> None:
    code = (
        "import sys, anki_tools.cli\n"
        "print(sorted(name for name in sys.modules if name in "
        "('jieba', 'ToJyutping', 'spacy') or "
        "name.startswith('anki_tools.languages.') and name != "
        "'anki_tools.languages.registry'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_every_language_path_resolves() -> None:
    for entry in LANGUAGES.values():
        assert (entry.profile is None) == (entry.profile_path is None)
        assert (entry.module is None) == (entry.module_path is None)
        assert (entry.audit is None) == (entry.audit_path is None)
        if entry.fix:
            assert entry.fix.core is not None
            assert (entry.fix.cull is None) == (entry.fix.cull_path is None)


@pytest.mark.parametrize(
    "path, error",
    [
        ("anki_tools.languages.greek.nowhere:GREEK", ImportError),
        ("anki_tools.languages.greek.profile:NOPE", AttributeError),
    ],
)
def test_a_bad_path_fails_at_first_read(
    path: str, error: type[BaseException]
) -> None:
    entry = replace(LANGUAGES["greek"], profile_path=path)

    with pytest.raises(error):
        entry.profile
