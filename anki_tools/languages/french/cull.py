"""French cull adapter: profile, constants, and bound wrappers.

The module holds the French values of the shared cull flow and the
:class:`anki_tools.ranking.cullflow.CullProfile` that ties them together.
The generic logic lives in ``anki_tools/ranking/cullflow`` and
``anki_tools/ranking/cullgates``.

See ``specs/french-1000-sentence-cull.md`` sections 4 to 9.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.french.profile import (
    EXTRA_CSS,
    FINAL_MARKS,
    INTERNAL_MARKS,
    QUOTE_MARKS,
    TOKENIZER,
)
from anki_tools.ranking import coreflow, corelist, cullflow
from anki_tools.ranking.coreflow import InputCard
from anki_tools.sentence_pack.pack import load_css
from anki_tools.sentence_pack.tokens import load_model

FREQUENCY_PATH = Path("data/french_frequency.tsv")
NAMES_PATH = Path("vocab/french_names.txt")
TATOEBA_PATH = Path("data/tatoeba/fra_sentences.tsv")
MODEL_NAME = "French Sentences"
FRENCH_FIELD = "French"
AUDIO_FIELD = "sound"
ENGLISH_FIELD = "Back"

SWADESH_PACK_PATH = Path("apkg/swadesh_french_sentences.apkg")
SWADESH_MODEL_NAME = "Swadesh French Sentences"
SWADESH_TOTAL = 207
SWADESH_EXCEPTIONS = ("intestins",)

CARD_CSS_PATH = SHARED_CSS_PATH
TEMPLATE_DIR = TEMPLATES_DIR / "french_cull"
FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CARD_TEMPLATES = ((FRONT_TEMPLATE, BACK_TEMPLATE),)

DEFAULT_DECK_NAME = "French 1000 Core Sentences"
SENSE_PATH = Path("vocab/french_core_senses.tsv")
LICENSES = {
    "tatoeba": "Tatoeba sentence text: CC BY 2.0 FR",
    "wordfreq": "wordfreq list: Apache License 2.0",
    "audio": "Input audio: license of the source deck",
}


def load_input_deck(path: Path) -> list[InputCard]:
    """Read the cards of the French sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``French Sentences`` model.
    :returns: One card per note, in the input order.
    """
    profile = FRENCH_CULL_PROFILE
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
    return cullflow.tokenize_texts((card.sentence for card in cards), model=TOKENIZER)


def load_or_build_frequency(
    frequency_path: Path = FREQUENCY_PATH, tatoeba_path: Path = TATOEBA_PATH
) -> list[cullflow.FrequencyRow]:
    """Give the cached French frequency rows, and build the cache when it is absent.

    :param frequency_path: Frequency cache TSV.
    :param tatoeba_path: Tatoeba French sentence TSV.
    :raises RuntimeError: The Tatoeba corpus is absent.
    :returns: Rows in file order.
    """
    if not frequency_path.exists():
        if not tatoeba_path.exists():
            raise RuntimeError(
                f"The Tatoeba French corpus {tatoeba_path} is absent. "
                "Run `anki swadesh fetch french` to fetch it."
            )
        cullflow.build_frequency_cache(
            tatoeba_path, frequency_path, FRENCH_CULL_PROFILE
        )
    return cullflow.load_frequency(frequency_path)


def card_css(css_path: Path = CARD_CSS_PATH) -> str:
    """Give the CSS of the culled deck.

    :param css_path: Base CSS file of the card classes.
    :returns: The base CSS and the French pack classes.
    """
    return load_css(css_path, EXTRA_CSS)


def _concept_forms(concept: str) -> set[str]:
    """Give the lemma and the text forms of one Swadesh concept.

    :param concept: Swadesh target word.
    :returns: The lowercased forms of the concept.
    """
    doc = load_model(TOKENIZER)(concept)
    alpha = [token for token in doc if token.is_alpha]
    return {token.lemma_.lower() for token in alpha} | {
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

FRENCH_CULL_PROFILE = cullflow.CullProfile(
    tokenize_texts=partial(cullflow.tokenize_texts, model=TOKENIZER),
    load_frequency=load_or_build_frequency,
    frequency_ranks=cullflow.frequency_ranks,
    read_names=partial(cullflow.read_names, NAMES_PATH),
    concept_forms=_concept_forms,
    content_pos=corelist.CONTENT_POS,
    min_lemma_length=corelist.MIN_LEMMA_LENGTH,
    final_marks=FINAL_MARKS,
    quote_marks=QUOTE_MARKS,
    internal_marks=INTERNAL_MARKS,
    input_model_name=MODEL_NAME,
    input_sentence_field=FRENCH_FIELD,
    input_english_field=ENGLISH_FIELD,
    input_audio_field=AUDIO_FIELD,
    output_model_name=DEFAULT_DECK_NAME,
    clean_sentence=cullflow.plain_sentence,
    reserve_fields=_reserve_fields,
    templates=CARD_TEMPLATES,
    card_css=card_css,
    sense_path=SENSE_PATH,
    reserve_pack=SWADESH_PACK_PATH,
    reserve_model_name=SWADESH_MODEL_NAME,
    swadesh_total=SWADESH_TOTAL,
    swadesh_exceptions=SWADESH_EXCEPTIONS,
    frequency_path=FREQUENCY_PATH,
    names_path=NAMES_PATH,
    tatoeba_path=TATOEBA_PATH,
    wordfreq_lang="fr",
    default_deck_name=DEFAULT_DECK_NAME,
    licenses=LICENSES,
    target_coverage_gate=0.98,
    double_coverage_gate=0.95,
)

write_cull_deck = partial(
    coreflow.write_deck,
    profile=FRENCH_CULL_PROFILE,
    reserve_fields=_reserve_fields,
)



