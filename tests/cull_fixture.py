"""Builders that the cull tests share."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import coreflow, corelist, cullflow
from tests.deck_fixture import MP3_STUB

ENGLISH = "A short English line."
CONTENT_POS = corelist.CONTENT_POS


def card(
    sentence: str,
    index: int = 0,
    english: str = ENGLISH,
    audio: str = "a.mp3",
) -> coreflow.InputCard:
    return coreflow.InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def target(lemma: str, rank: int, deck_count: int = 3) -> corelist.TargetWord:
    return corelist.TargetWord(
        lemma=lemma,
        rank=rank,
        deck_count=deck_count,
        content_share=1.0,
        weight=1 / math.log2(rank + 1),
    )


def freq_row(
    word: str,
    count: int = 10,
    pm: float = 5.0,
    written: float = 5.0,
) -> cullflow.FrequencyRow:
    return cullflow.FrequencyRow(
        word=word,
        tatoeba_count=count,
        tatoeba_pm=pm,
        written_pm=written,
        zipf=4.0,
    )


def select(
    cards: list[coreflow.InputCard],
    tokens: list[list[tuple[str, str]]],
    targets: list[corelist.TargetWord],
    ranks: dict[str, int],
    **options: Any,
) -> cullflow.SelectionResult:
    """Select cards with the test defaults: no exempt word, a cap of 10 cards."""
    options = {"max_cards": 10, "exempt_top": 0, "content_pos": CONTENT_POS} | options
    return cullflow.select_cards(cards, tokens, targets, ranks, **options)


def tiny_deck(
    tmp_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    model_name: str,
    audio: list[str],
    filename: str = "input.apkg",
) -> Path:
    """Write a source deck of ``rows`` with one stub MP3 for each name in ``audio``."""
    media = []
    for audio_name in audio:
        path = tmp_path / audio_name
        path.write_bytes(MP3_STUB)
        media.append(str(path))
    source = tmp_path / filename
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=fields,
        cards=rows,
        media_files=media,
        model_name=model_name,
    )
    return source
