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
from typing import Iterable, Sequence

from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.spanish import core
from anki_tools.languages.spanish.frequency import (
    FREQUENCY_PATH,
    base_lemma,
    cull_frequency_rows,
    load_frequency_es,
)
from anki_tools.languages.spanish.profile import (
    EXTRA_CSS,
    INTERNAL_MARKS,
    MODEL_NAME as SWADESH_MODEL_NAME,
    QUOTE_MARKS,
    SWADESH_ROWS,
    TOKENIZER,
    prepare_text,
)
from anki_tools.ranking import coreflow, corelist, cullflow, cullgates
from anki_tools.ranking.coreflow import InputCard
from anki_tools.sentence_pack.pack import load_css
from anki_tools.sentence_pack.tokens import load_model

TATOEBA_PATH = Path("data/tatoeba/spa_sentences.tsv")
NAMES_PATH = Path("vocab/spanish_names.txt")


DEFAULT_MAX_CARDS = cullflow.DEFAULT_MAX_CARDS
DEFAULT_TARGET_WORDS = cullflow.DEFAULT_TARGET_WORDS
DEFAULT_MAX_USES = cullflow.DEFAULT_MAX_USES

MODEL_NAME = core.MODEL_NAME
SENTENCE_FIELD = core.SENTENCE_FIELD
ENGLISH_FIELD = core.ENGLISH_FIELD
AUDIO_FIELD = core.AUDIO_FIELD

SWADESH_PACK_PATH = Path("apkg/swadesh_spanish_sentences.apkg")
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

# Field order of the input model ``Spanish sentences cloze+``. The reserve
# cards join the input model, so they carry its 11 fields.
RESERVE_FIELD_ORDER = (
    "EnglishSentenceID",
    "SpanishSentenceID",
    "EnglishSentence",
    "SpanishSentence",
    "ClozeWord",
    "Difficulty",
    "Text",
    "Audio",
    "NumberOfSpanishWords",
    "Extra",
    "Other",
)


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
        profile.model_name,
        profile.sentence_field,
        profile.english_field,
        profile.audio_field,
        profile.clean_sentence,
    )




def tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize Spanish texts with the shared spaCy pipeline.

    :param texts: Sentence texts.
    :returns: One list of lowercased ``(lemma, part of speech)`` pairs per text.
    """
    return [
        [(base_lemma(lemma), pos) for lemma, pos in card_tokens]
        for card_tokens in cullflow.tokenize_texts(texts, model=TOKENIZER)
    ]


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card.
    """
    return tokenize_texts(card.sentence for card in cards)


def read_names(path: Path = NAMES_PATH) -> set[str]:
    """Read the Spanish name stoplist, or give an empty set without the file.

    :param path: Text file with one name per line.
    :returns: Lowercased names. A missing file gives an empty set.
    """
    if not path.exists():
        return set()
    return cullflow.read_names(path)




def load_frequency(
    frequency_path: Path = FREQUENCY_PATH,
) -> list[cullflow.FrequencyRow]:
    """Give the Spanish frequency rows of the cull.

    :param frequency_path: Spanish frequency cache TSV.
    :returns: One cull row per lemma.
    """
    return cull_frequency_rows(load_frequency_es(frequency_path))


def card_css(css_path: Path = CARD_CSS_PATH) -> str:
    """Give the CSS of the culled deck.

    :param css_path: Base CSS file of the card classes.
    :returns: The base CSS and the Spanish pack classes.
    """
    return load_css(css_path, EXTRA_CSS)


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
    """Give the field values of one reserve card in the input model order.

    :param card: Reserve card.
    :returns: The 11 field values of the ``Spanish sentences cloze+`` model.
    """
    back = f"{card.english} — {card.target}"
    if card.gloss:
        back += f" ({card.gloss})"
    return ["", "", back, card.html, "", "", "", f"[sound:{card.audio}]", "", "", ""]


load_reserve = partial(cullflow.load_reserve, model_name=SWADESH_MODEL_NAME)

write_deck = partial(
    coreflow.write_deck,
    model_name=MODEL_NAME,
    templates=CARD_TEMPLATES,
    card_css=card_css,
    reserve_fields=_reserve_fields,
)

SPANISH_CULL_PROFILE = cullflow.CullProfile(
    tokenize_texts=tokenize_texts,
    load_frequency=load_frequency,
    frequency_ranks=cullflow.frequency_ranks,
    read_names=read_names,
    concept_forms=_concept_forms,
    content_pos=corelist.CONTENT_POS,
    min_lemma_length=corelist.MIN_LEMMA_LENGTH,
    final_marks=FINAL_MARKS,
    quote_marks=QUOTE_MARKS,
    internal_marks=INTERNAL_MARKS,
    model_name=MODEL_NAME,
    sentence_field=SENTENCE_FIELD,
    english_field=ENGLISH_FIELD,
    audio_field=AUDIO_FIELD,
    clean_sentence=clean_sentence,
    reserve_fields=_reserve_fields,
    templates=CARD_TEMPLATES,
    card_css=card_css,
    reserve_pack=SWADESH_PACK_PATH,
    reserve_model_name=SWADESH_MODEL_NAME,
    swadesh_total=SWADESH_TOTAL,
    swadesh_exceptions=SWADESH_EXCEPTIONS,
    frequency_path=FREQUENCY_PATH,
    names_path=NAMES_PATH,
    tatoeba_path=TATOEBA_PATH,
    wordfreq_lang="es",
    default_deck_name=DEFAULT_DECK_NAME,
    licenses=LICENSES,
    target_coverage_gate=0.98,
    double_coverage_gate=0.95,
)


# Re-export the shared flow surface that the CLI and the tests read here.
CullRequest = cullflow.CullRequest
CullRun = cullflow.CullRun
ReserveCard = cullflow.ReserveCard
FrequencyRow = cullflow.FrequencyRow
SelectionResult = cullflow.SelectionResult
CULL_COLUMNS = cullflow.CULL_COLUMNS
COVERAGE_COLUMNS = cullflow.COVERAGE_COLUMNS
UNRANKED_ORDER = cullflow.UNRANKED_ORDER
SWADESH_SOURCE = cullflow.SWADESH_SOURCE
FREQUENCY_SOURCE = cullflow.FREQUENCY_SOURCE
blend_pm = cullflow.blend_pm
frequency_ranks = cullflow.frequency_ranks
lemma_sets = cullflow.lemma_sets
select_cards = cullflow.select_cards
order_cards = cullflow.order_cards
hardest_rank = cullflow.hardest_rank
spoken_mass = cullflow.spoken_mass
default_output = cullflow.default_output
side_paths = cullflow.side_paths
write_cull_csv = cullflow.write_cull_csv
write_coverage_csv = cullflow.write_coverage_csv
coverage_status = cullflow.coverage_status
kept_order_hash = cullflow.kept_order_hash
dedupe_reserve = cullflow.dedupe_reserve
dedupe_reserve_sets = cullflow.dedupe_reserve_sets
swadesh_coverage = partial(cullflow.swadesh_coverage, profile=SPANISH_CULL_PROFILE)
filter_cards = partial(
    cullflow.filter_cards,
    card_rule=partial(cullflow.filter_reason, profile=SPANISH_CULL_PROFILE),
)
_filter_reason = partial(cullflow.filter_reason, profile=SPANISH_CULL_PROFILE)
TargetWord = corelist.TargetWord
build_targets = corelist.build_targets
write_manifest = coreflow.write_manifest

run_gates = cullgates.run_gates
GateInputs = cullgates.CullFiles
gate_inputs = cullgates.gate_inputs


def run_cull(request: CullRequest) -> CullRun:
    """Run the Spanish cull and write the output files.

    :param request: Options of the run.
    :returns: The written paths and the counts of the run.
    """
    return cullflow.run_cull(request, SPANISH_CULL_PROFILE)


def check_cull(output_path: Path) -> int:
    """Run gates G1 to G15 on the files of a Spanish cull output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return cullgates.check_cull(output_path, SPANISH_CULL_PROFILE)
