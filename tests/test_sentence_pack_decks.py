"""Tests for the deck names of the Swadesh sentence packs."""

from __future__ import annotations

import pytest

from anki_tools.sentence_pack.pack import used_audio_authors


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
