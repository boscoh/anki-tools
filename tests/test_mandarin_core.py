"""Tests for the Mandarin core adapter.

The tests read the committed Neri deck, so a fresh clone skips them when the
file is absent. The frequency test also skips when the local cache
``data/mandarin_frequency.tsv`` is absent, because the file is not tracked.
"""

from __future__ import annotations

import pytest

from anki_tools.languages.mandarin import core
from anki_tools.languages.mandarin.profile import shared_pos
from anki_tools.ranking import coreflow
from anki_tools.ranking.corelist import CONTENT_POS

DECK = core.Path("apkg/Neri's Chinese Sents Part 2.apkg")
HAS_DECK = DECK.exists()
HAS_FREQUENCY = core.FREQUENCY_PATH.exists()

pytestmark = pytest.mark.skipif(not HAS_DECK, reason="the Neri Part 2 deck is absent")


def test_sentence_reads_plain_text():
    """The sentence field gives Simplified text with an ASCII final mark."""
    assert core._sentence("我喝茶。") == "我喝茶."
    assert core._sentence("你叫什么名字？") == "你叫什么名字?"
    assert core._sentence("<b>水</b>很好！") == "水很好!"


def test_sentence_appends_a_final_mark():
    """A text with no final mark gets a full stop, because F4 reads one."""
    assert core._sentence("我从北京来") == "我从北京来."
    assert core._sentence("") == ""


def test_sentence_uses_the_simplified_converter():
    """A Traditional text becomes Simplified."""
    assert core._sentence("我來喝茶。") == "我来喝茶."


def test_word_table_gives_triples():
    """The reviewed rows give ``(index, form, gloss)`` triples in table order."""
    table = core.word_table()
    assert len(table) == 211
    index, form, gloss = table[0]
    assert (index, form, gloss) == (1, "我", "I")


def test_load_input_deck_reads_the_neri_fields():
    """The adapter reads the sentence, the English, and the audio of a note."""
    cards = core.load_input_deck(DECK)
    assert len(cards) == 5009
    card = cards[0]
    assert card.sentence.endswith((".", "?", "!"))
    assert card.english
    assert card.audio.endswith(".mp3")
    assert [entry.index for entry in cards[:3]] == [0, 1, 2]


def test_tokenize_deck_tags_han_tokens():
    """A Han token gets a shared tag and the punctuation drops."""
    card = coreflow.InputCard(
        note_id=1,
        sentence="我在喝水.",
        english="I drink water.",
        audio="a.mp3",
        index=0,
    )
    tokens = core.tokenize_deck([card])
    tags = {tag for _text, tag in tokens[0]}
    assert tags <= CONTENT_POS | {""}
    parts = [text for text, _tag in tokens[0]]
    assert "." not in parts
    assert "水" not in parts



def test_shared_pos_maps_the_jieba_flags():
    """A jieba prefix gives a content tag, and a particle gives none."""
    assert shared_pos("n") == "NOUN"
    assert shared_pos("nr") == "NOUN"
    assert shared_pos("vn") == "VERB"
    assert shared_pos("ad") == "ADJ"
    assert shared_pos("d") == "ADV"
    assert shared_pos("r") == "NOUN"
    assert shared_pos("u") == ""

def test_hits_read_a_character_span():
    """The tier S hit reads the form as a span, not as a jieba token."""
    profile = core.MANDARIN_CORE_PROFILE
    assert profile.hits("我在喝水。", {"水"}) == {"水": 1}
    assert profile.hits("其他的人。", {"其他"}) == {"其他": 1}
    assert profile.hits("这一切很好。", {"切"}) == {}


@pytest.mark.skipif(not HAS_FREQUENCY, reason="the Mandarin frequency cache is absent")
def test_frequency_ranks_cover_the_deck_words():
    """The cached list ranks the forms of the table."""
    rows = core.load_frequency()
    ranks = core.frequency_ranks(rows)
    assert ranks["的"] == 1
    assert ranks["是"] == 2
    assert ranks["我"] == 5


def test_profile_holds_the_deck_paths():
    """The profile names the Neri deck files and the Mandarin hooks."""
    profile = core.MANDARIN_CORE_PROFILE
    assert profile.model_name == "0 2024 Sentences Deck"
    assert profile.sentence_field == "Sentence"
    assert profile.english_field == "Sentence (Translation)"
    assert profile.audio_field == "[a]"
    assert profile.core_field == "Core"
    assert profile.core_gloss_field == "CoreGloss"
    assert profile.min_units < profile.max_units
    assert profile.templates
    assert profile.card_css().startswith("/*")
    assert profile.frequency_source == str(core.FREQUENCY_PATH)


def test_templates_name_the_audio_field():
    """The model holds the audio in the ``[a]`` field, so the template must name it."""
    reference = "{{" + core.AUDIO_FIELD + "}}"
    assert reference in core.CORE_FRONT_TEMPLATE
    assert reference in core.CORE_BACK_TEMPLATE
