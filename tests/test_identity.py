"""Tests for the one identity of every artifact name (ADR 0015)."""

from __future__ import annotations

from pathlib import Path

from anki_tools.ranking.identity import DeckIdentity, default_output


def test_the_default_output_names_one_stem_per_language_and_mode() -> None:
    assert default_output("french", "core") == Path("apkg/french_core.apkg")
    assert default_output("french", "cull") == Path("apkg/french_1000.apkg")
    assert default_output("spanish", "cull") == Path("apkg/spanish_1000.apkg")


def test_the_title_and_the_note_type_take_the_output_stem() -> None:
    identity = DeckIdentity(Path("apkg/french_core.apkg"))

    assert identity.stem == "french_core"
    assert identity.title == identity.model_name == "french_core"


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
