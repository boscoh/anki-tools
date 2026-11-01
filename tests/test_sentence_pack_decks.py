"""Tests for the deck names of the Swadesh sentence packs."""

from __future__ import annotations

import pytest

from anki_tools.languages import PROFILES
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack.pack import used_audio_authors


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_pack_deck_title_has_the_swadesh_prefix_and_names_the_note_type(
    name: str,
) -> None:
    identity = DeckIdentity(default_output(name, "pack"))

    assert identity.title == f"Swadesh Sentences {name.title()}"
    assert identity.model_name == identity.title


AUTHORS = {1: "one", 2: "two", 3: "three"}


@pytest.mark.parametrize(
    "rows, expected",
    [
        pytest.param(
            [
                {"audio_id": "2", "audio_source": "native"},
                {"audio_id": "3", "audio_source": "tts"},
            ],
            {2: "two"},
            id="keeps-only-the-native-rows",
        ),
        pytest.param(
            [
                {"audio_id": "0", "audio_source": "native"},
                {"audio_id": "", "audio_source": "native"},
                {"audio_id": "none", "audio_source": "native"},
            ],
            {},
            id="skips-an-empty-or-bad-id",
        ),
    ],
)
def test_used_audio_authors_reads_the_native_rows_of_the_pack(
    rows: list[dict[str, str]], expected: dict[int, str]
) -> None:
    assert used_audio_authors(AUTHORS, rows) == expected
