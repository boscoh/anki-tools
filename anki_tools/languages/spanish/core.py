"""Spanish core deck: profile and values.

The module builds the core deck of the Spanish sentence deck
``apkg/Spanish 9000 sentences with native audio.apkg``. The Swadesh rows of
``vocab/spanish_words.csv`` give the tier S targets, and the tier F targets
come from ``DEFAULT_CORE_WORDS`` words of the frequency list. Their sense rows
live in ``vocab/spanish_core_senses.tsv``. The generic logic lives in
``anki_tools.ranking.coreflow`` and ``anki_tools.ranking.coregates``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from anki_tools.languages.spanish.frequency import (
    FREQUENCY_PATH,
    SpanishFrequencyRow,
    base_lemma,
    build_frequency_es,
    frequency_ranks_es,
    load_frequency_es,
    write_frequency_tsv,
)

from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.spanish.profile import (
    EXTRA_CSS,
    TOKENIZER,
    SPANISH,
    load_word_table,
    prepare_text,
)
from anki_tools.ranking import coreflow, cullflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.deckprofile import LanguageDeckProfile
from anki_tools.apkg.templates import load_css
from anki_tools.sentence_pack.tokens import tokenize

MODEL_NAME = "Spanish sentences cloze+"
SENTENCE_FIELD = "SpanishSentence"
ENGLISH_FIELD = "EnglishSentence"
AUDIO_FIELD = "Audio"

SENSE_PATH = Path("vocab/spanish_core_senses.tsv")
LOAN_PATH = Path("vocab/spanish_loanwords.txt")

DEFAULT_CORE_WORDS = 50
# The spaCy model gives two artefact lemmas: ``quier`` for ``quieres``, and
# ``buen`` for ``buen`` and ``buena``.
LEMMA_OVERRIDES = {"quier": "querer", "buen": "bueno"}
# A tier F card can hold one content lemma, so the selection deduplicates on a
# lower Jaccard limit than the gate K6 limit of 0.67.
#: The word-set Jaccard value at which two Spanish sentences are duplicates.
SIMILARITY_THRESHOLD = 0.5

DEFAULT_OPTIONS = CoreOptions(
    tail_cards=0, hard_top=50_000, similarity=SIMILARITY_THRESHOLD
)

MIN_UNITS = 4
MAX_UNITS = 10

TEMPLATE_DIR = TEMPLATES_DIR / "spanish_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

_BOLD_PATTERN = re.compile(r"</?b>")


def _plain_sentence(field_value: str) -> str:
    """Remove the bold markup of a sentence field value."""
    return _BOLD_PATTERN.sub("", field_value).strip()


def _sentence(field_value: str) -> str:
    """Give the prepared text of a raw sentence field value."""
    return prepare_text(_plain_sentence(field_value))


def load_input_deck(path: Path) -> list[coreflow.InputCard]:
    """Read the cards of the Spanish sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``Spanish sentences cloze+`` model.
    :returns: One card per note, in the input order.
    """
    return coreflow.read_input_deck(
        path,
        MODEL_NAME,
        SENTENCE_FIELD,
        ENGLISH_FIELD,
        AUDIO_FIELD,
        _sentence,
    )


def tokenize_deck(cards: Sequence[coreflow.InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card.
    """
    return [
        [
            (tok.lemma.lower(), tok.pos)
            for tok in tokenize(TOKENIZER, card.sentence)
            if not tok.punctuation and tok.text.isalpha()
        ]
        for card in cards
    ]


def tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize sentence texts with the Spanish lemmatizer of the cull.

    :param texts: Sentence texts.
    :returns: One list of ``(lemma, part of speech)`` pairs per text.
    """
    return cullflow.tokenize_texts(texts, model=TOKENIZER, lemma_map=base_lemma)

def load_frequency() -> list[SpanishFrequencyRow]:
    """Give the Spanish frequency rows, and build the file when it is absent.

    The frequency module holds no gate function, so the loader writes the built
    rows without an extra check.

    :raises RuntimeError: The source digest differs from the pinned value, or
        the file cannot be written.
    :returns: Rows in rank order.
    """
    if FREQUENCY_PATH.exists():
        return load_frequency_es(FREQUENCY_PATH)
    rows = build_frequency_es()
    write_frequency_tsv(rows, FREQUENCY_PATH)
    return rows


def frequency_ranks(rows: Sequence[Any], spoken_weight: float = 0.5) -> dict[str, int]:
    """Map each Spanish lemma to its 1-based rank.

    :param rows: Frequency cache rows.
    :param spoken_weight: Unused, kept for the profile contract.
    :returns: Lemma to rank.
    """
    return frequency_ranks_es(rows)


def read_names() -> set[str]:
    """Give an empty Spanish name stoplist."""
    return set()


def word_table() -> list[tuple[int, str, str]]:
    """Give the reviewed Spanish Swadesh rows as ``(index, form, gloss)``."""
    return [
        (row.swadesh_index, row.spanish, row.english) for row in load_word_table()
    ]


def card_css() -> str:
    """Give the CSS of the Spanish core deck."""
    return load_css(SHARED_CSS_PATH, EXTRA_CSS)



SPANISH_LANGUAGE = LanguageDeckProfile(
    load_input_deck=load_input_deck,
    tokenize_texts=tokenize_texts,
    load_frequency=load_frequency,
    frequency_ranks=frequency_ranks,
    read_names=read_names,
    input_model_name=MODEL_NAME,
    input_sentence_field=SENTENCE_FIELD,
    input_english_field=ENGLISH_FIELD,
    input_audio_field=AUDIO_FIELD,
    clean_sentence=_sentence,
    card_css=card_css,
    sense_path=SENSE_PATH,
    highlight=SPANISH.highlight,
)

SPANISH_CORE_PROFILE = SPANISH_LANGUAGE.core(
    tokenize_deck=tokenize_deck,
    word_table=word_table,
    loan_path=LOAN_PATH,
    has_orphan=SPANISH.has_orphan,
    hits=SPANISH.hits,
    sense_filter=SPANISH.sense_filter,
    lemma_key=str.casefold,
    templates=CORE_TEMPLATES,
    min_units=MIN_UNITS,
    max_units=MAX_UNITS,
    default_options=DEFAULT_OPTIONS,
    frequency_source=str(FREQUENCY_PATH),
    lemma_overrides=LEMMA_OVERRIDES,
)
