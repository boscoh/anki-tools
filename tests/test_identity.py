"""Tests for the one identity of every artifact name (ADR 0015)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.ranking.identity import DeckIdentity, default_output


@pytest.mark.parametrize(
    "language, mode, expected",
    [
        ("french", "core", "apkg/french_core.apkg"),
        ("french", "cull", "apkg/french_1000.apkg"),
        ("spanish", "cull", "apkg/spanish_1000.apkg"),
        ("french", "pack", "apkg/swadesh_sentences_french.apkg"),
        ("french", "vocab", "apkg/swadesh_vocab_french.apkg"),
    ],
)
def test_the_default_output_names_one_stem_per_language_and_mode(
    language: str, mode: str, expected: str
) -> None:
    assert default_output(language, mode) == Path(expected)


@pytest.mark.parametrize(
    "output, stem, title",
    [
        ("apkg/french_core.apkg", "french_core", "French Core"),
        ("apkg/spanish_1000.apkg", "spanish_1000", "Spanish 1000"),
    ],
)
def test_the_title_and_the_note_type_are_the_title_case_of_the_stem(
    output: str, stem: str, title: str
) -> None:
    identity = DeckIdentity(Path(output))

    assert identity.stem == stem
    assert identity.title == identity.model_name == title


def test_the_side_files_take_the_output_stem() -> None:
    identity = DeckIdentity(Path("apkg/french_core.apkg"))

    assert [path.name for path in identity.core_sides()] == [
        "french_core.core.csv",
        "french_core.coverage.csv",
        "french_core.manifest.json",
        "french_core.rank.csv",
    ]
    assert [path.name for path in identity.cull_sides()] == [
        "french_core.cull.csv",
        "french_core.coverage.csv",
        "french_core.manifest.json",
    ]
