"""Tests for the deck names of the Swadesh sentence packs."""

from __future__ import annotations

from anki_tools.sentence_pack.languages import PROFILES


def test_deck_name_has_the_swadesh_prefix() -> None:
    for name, profile in PROFILES.items():
        assert profile.deck_name == f"Swadesh {name.title()} Sentences"
