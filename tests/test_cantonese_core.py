"""Tests for the Cantonese core adapter.

The tests read the committed LTL deck. The tokenizer tests need the Cifu
cache, which ``anki freq yue`` fills; without the cache they skip, so a fresh
clone stays green.
"""

from __future__ import annotations

import pytest

from anki_tools.languages.cantonese import core
from anki_tools.languages.cantonese.profile import prepare_text
from anki_tools.ranking.corelist import CONTENT_POS

DECK = core.Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg")
HAS_DECK = DECK.exists()

pytestmark = pytest.mark.skipif(not HAS_DECK, reason="the LTL Cantonese deck is absent")


def test_sentence_reads_plain_text():
    """The sentence field gives Traditional text with an ASCII final mark."""
    assert core._sentence("我睇咗一本書。") == "我睇咗一本書."
    assert core._sentence("點解？") == "點解?"
    assert core._sentence("<b>死</b>開！") == "死開!"


def test_sentence_uses_prepare_text():
    """A simplified form becomes Traditional."""
    assert core._sentence("点解？") == "點解?"


def test_read_names_is_a_stoplist():
    """The name file of the Cantonese pack gives a non-empty stoplist."""
    names = core.read_names()
    assert names
    assert all(isinstance(name, str) and name for name in names)


def test_word_table_gives_triples():
    """The reviewed rows give ``(index, form, gloss)`` triples in table order."""
    table = core.word_table()
    assert len(table) >= 207
    index, form, gloss = table[0]
    assert (index, form, gloss) == (1, "我", "I")


def test_load_input_deck_reads_the_ltl_fields():
    """The adapter reads the sentence, the English, and the audio of a note."""
    cards = core.load_input_deck(DECK)
    assert len(cards) == 999
    card = cards[0]
    assert card.sentence == prepare_text(card.sentence)
    assert card.english
    assert card.audio.endswith(".mp3")
    assert [entry.index for entry in cards[:3]] == [0, 1, 2]


def test_tokenize_deck_tags_every_token_as_content():
    """A Han token gets the constant content tag and the punctuation drops."""
    card = core.InputCard(
        note_id=1, sentence="我睇咗一本書.", english="I read a book.", audio="a.mp3", index=0
    )
    tokens = core.tokenize_deck([card])
    assert all(tag in CONTENT_POS for _text, tag in tokens[0])
    parts = [text for text, _tag in tokens[0]]
    assert parts[0] == "我"
    assert "." not in parts


def test_frequency_ranks_cover_the_deck_words():
    """The cached list ranks the words of a card."""
    rows = core.load_frequency()
    ranks = core.frequency_ranks(rows)
    assert ranks.get("我", 0) > 0
    assert ranks["係"] == 1


def test_profile_holds_the_deck_paths():
    """The profile names the LTL deck files and the Cantonese hooks."""
    profile = core.CANTONESE_CORE_PROFILE
    assert profile.model_name == "LTL Note Type - Cantonese"
    assert profile.sentence_field == "Front"
    assert profile.english_field == "Back"
    assert profile.audio_field == "Audio"
    assert profile.core_field == "Core"
    assert profile.min_units < profile.max_units
    assert profile.templates
    assert profile.card_css().startswith("/*")
