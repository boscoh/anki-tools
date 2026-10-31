"""Tests for the deck names of the Swadesh sentence packs."""

from __future__ import annotations

from anki_tools.languages import PROFILES
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack.pack import used_audio_authors


def test_the_default_deck_title_has_the_swadesh_prefix() -> None:
    for name in PROFILES:
        identity = DeckIdentity(default_output(name, "pack"))
        assert identity.title == f"Swadesh Sentences {name.title()}"


def test_the_pack_note_type_matches_its_deck_title() -> None:
    for name in PROFILES:
        identity = DeckIdentity(default_output(name, "pack"))
        assert identity.model_name == identity.title


def test_used_audio_authors_keeps_only_the_native_rows_of_the_pack() -> None:
    authors = {1: "one", 2: "two", 3: "three"}

    kept = used_audio_authors(
        authors,
        [
            {"audio_id": "2", "audio_source": "native"},
            {"audio_id": "3", "audio_source": "tts"},
        ],
    )

    assert kept == {2: "two"}


def test_used_audio_authors_skips_an_empty_or_bad_id() -> None:
    authors = {1: "one"}

    kept = used_audio_authors(
        authors,
        [
            {"audio_id": "0", "audio_source": "native"},
            {"audio_id": "", "audio_source": "native"},
            {"audio_id": "none", "audio_source": "native"},
        ],
    )

    assert kept == {}
