"""Spanish cull adapter: profile, constants, and bound wrappers.

The module holds the Spanish values of the shared cull flow and the
:class:`anki_tools.ranking.cullflow.CullProfile` that ties them together. The
input deck is ``apkg/Spanish 9000 sentences with native audio.apkg``, and the
reserve pack is the Spanish Swadesh pack. The generic logic lives in
``anki_tools/ranking/cullflow`` and ``anki_tools/ranking/cullgates``.

See ``specs/french-1000-sentence-cull.md`` sections 4 to 9.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.spanish import core
from anki_tools.languages.spanish.frequency import (
    FREQUENCY_PATH,
    base_lemma,
    cull_frequency_rows,
    load_frequency_es,
)
from anki_tools.languages.spanish.profile import (
    INTERNAL_MARKS,
    QUOTE_MARKS,
    SWADESH_ROWS,
    TOKENIZER,
    prepare_text,
)
from anki_tools.ranking import coreflow, corelist, cullflow
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.ranking.coreflow import InputCard
from anki_tools.sentence_pack.tokens import load_model

TATOEBA_PATH = Path("data/tatoeba/spa_sentences.tsv")
NAMES_PATH = Path("vocab/spanish_names.txt")


SWADESH_PACK_PATH = default_output("spanish", "pack")
SWADESH_MODEL_NAME = DeckIdentity(SWADESH_PACK_PATH).model_name
SWADESH_TOTAL = SWADESH_ROWS
# ``corteza`` has no card in the stale reserve pack. The shared reserve dedupe
# drops the cards of ``cola``, ``coser``, and ``raíz`` against input cards that
# the cull does not keep, so no kept sentence of the deck holds them.
SWADESH_EXCEPTIONS: tuple[str, ...] = ("cola", "corteza", "coser", "raíz")
# The Spanish ellipsis closes a sentence in the corpus, so the cull accepts it.
FINAL_MARKS = ".?!…"
CARD_CSS_PATH = SHARED_CSS_PATH
TEMPLATE_DIR = TEMPLATES_DIR / "spanish_cull"
FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CARD_TEMPLATES = ((FRONT_TEMPLATE, BACK_TEMPLATE),)

DEFAULT_DECK_NAME = "Spanish 1000 Core Sentences"
LICENSES = {
    "tatoeba": "Tatoeba sentence text: CC BY 2.0 FR",
    "wordfreq": "wordfreq list: Apache License 2.0",
    "audio": "Input audio: license of the source deck",
    "swadesh": "Swadesh word data: CC BY-SA 4.0",
}



def clean_sentence(field_value: str) -> str:
    """Remove the bold markup and normalize the spaces of a sentence field.

    :param field_value: Raw ``SpanishSentence`` field text.
    :returns: The prepared sentence text.
    """
    return prepare_text(cullflow.plain_sentence(field_value))


def load_input_deck(path: Path) -> list[InputCard]:
    """Read the cards of the Spanish sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``Spanish sentences cloze+``
        model.
    :returns: One card per note, in the input order.
    """
    profile = SPANISH_CULL_PROFILE
    return coreflow.read_input_deck(
        path,
        profile.input_model_name,
        profile.input_sentence_field,
        profile.input_english_field,
        profile.input_audio_field,
        profile.clean_sentence,
    )


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card.
    """
    return cullflow.tokenize_texts(
        (card.sentence for card in cards), model=TOKENIZER, lemma_map=base_lemma
    )


def read_cull_frequency(
    frequency_path: Path = FREQUENCY_PATH,
) -> list[cullflow.FrequencyRow]:
    """Give the Spanish frequency rows of the cull.

    :param frequency_path: Spanish frequency cache TSV.
    :returns: One cull row per lemma.
    """
    return cull_frequency_rows(load_frequency_es(frequency_path))



def _concept_forms(concept: str) -> set[str]:
    """Give the lemma and the text forms of one Swadesh concept.

    :param concept: Swadesh target word.
    :returns: The lowercased forms of the concept.
    """
    doc = load_model(TOKENIZER)(concept)
    alpha = [token for token in doc if token.is_alpha]
    return {base_lemma(token.lemma_.lower()) for token in alpha} | {
        token.text.lower() for token in alpha
    }


def _reserve_fields(card: cullflow.ReserveCard) -> list[str]:
    """Give the five output field values of one reserve card.

    :param card: Reserve card.
    :returns: The ``Audio``, ``Sentence``, ``English``, ``Core``, and
        ``CoreGloss`` values.
    """
    return [f"[sound:{card.audio}]", card.html, card.english, card.target, card.gloss]


read_reserve = partial(cullflow.load_reserve, model_name=SWADESH_MODEL_NAME)

SPANISH_CULL_PROFILE = core.SPANISH_LANGUAGE.cull(
    load_frequency=read_cull_frequency,
    frequency_ranks=cullflow.frequency_ranks,
    read_names=partial(cullflow.read_names, NAMES_PATH, optional=True),
    clean_sentence=clean_sentence,
    concept_forms=_concept_forms,
    content_pos=corelist.CONTENT_POS,
    min_lemma_length=corelist.MIN_LEMMA_LENGTH,
    final_marks=FINAL_MARKS,
    quote_marks=QUOTE_MARKS,
    internal_marks=INTERNAL_MARKS,
    reserve_fields=_reserve_fields,
    templates=CARD_TEMPLATES,
    reserve_pack=SWADESH_PACK_PATH,
    reserve_model_name=SWADESH_MODEL_NAME,
    swadesh_total=SWADESH_TOTAL,
    swadesh_exceptions=SWADESH_EXCEPTIONS,
    frequency_path=FREQUENCY_PATH,
    wordfreq_lang="es",
    licenses=LICENSES,
    target_coverage_gate=0.98,
    double_coverage_gate=0.95,
)
