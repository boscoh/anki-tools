"""Tests of the deck readers: the field names and the input read order."""

from __future__ import annotations

from pathlib import Path

from anki_tools.ranking import assemble, deckreader
from tests.deck_fixture import MODEL_NAME, build_contract_deck


def test_the_output_fields_come_from_the_reader_constants() -> None:
    assert assemble.OUTPUT_FIELDS == (
        deckreader.OUTPUT_AUDIO_FIELD,
        deckreader.OUTPUT_SENTENCE_FIELD,
        deckreader.OUTPUT_ENGLISH_FIELD,
        deckreader.OUTPUT_CORE_FIELD,
        deckreader.OUTPUT_GLOSS_FIELD,
    )


def test_read_input_deck_gives_one_card_per_note_in_input_order(
    tmp_path: Path,
) -> None:
    path = build_contract_deck(tmp_path)

    cards = deckreader.read_input_deck(
        path, MODEL_NAME, "Hanzi", "English", "Audio", str.strip
    )

    assert [card.index for card in cards] == list(range(len(cards)))
    assert [card.sentence for card in cards] == [f"第{index}句话" for index in range(1, 7)]
