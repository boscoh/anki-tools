"""Tests for the one identity of every artifact name (ADR 0015)."""

from __future__ import annotations

from pathlib import Path

from anki_tools.ranking.identity import DeckIdentity


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
