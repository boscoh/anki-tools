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
from typing import Iterable, Sequence

from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.french.profile import EXTRA_CSS, TOKENIZER
from anki_tools.ranking import cardfilter, coreflow, corelist, cullflow, cullgates
from anki_tools.ranking.coreflow import InputCard
from anki_tools.sentence_pack.pack import load_css
from anki_tools.sentence_pack.tokens import load_model

FREQUENCY_PATH = Path("data/french_frequency.tsv")
NAMES_PATH = Path("vocab/french_names.txt")
TATOEBA_PATH = Path("data/tatoeba/fra_sentences.tsv")

DEFAULT_MAX_CARDS = cullflow.DEFAULT_MAX_CARDS
DEFAULT_TARGET_WORDS = cullflow.DEFAULT_TARGET_WORDS
DEFAULT_MAX_USES = cullflow.DEFAULT_MAX_USES

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
        profile.model_name,
        profile.sentence_field,
        profile.english_field,
        profile.audio_field,
        profile.clean_sentence,
    )


def tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize French texts with the shared spaCy pipeline.

    :param texts: Sentence texts.
    :returns: One list of lowercased ``(lemma, part of speech)`` pairs per text.
    """
    return cullflow.tokenize_texts(texts, model=TOKENIZER)


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card.
    """
    return tokenize_texts(card.sentence for card in cards)


def read_names(path: Path = NAMES_PATH) -> set[str]:
    """Read the French name stoplist.

    :param path: Text file with one name per line.
    :returns: Lowercased names. Blank lines and ``#`` comments are skipped.
    """
    return cullflow.read_names(path)


def load_frequency() -> list[cullflow.FrequencyRow]:
    """Read the cached French frequency list."""
    return cullflow.load_frequency(FREQUENCY_PATH)


def build_frequency_cache(
    tatoeba_path: Path = TATOEBA_PATH, out_path: Path = FREQUENCY_PATH
) -> int:
    """Build the cached French frequency list from the Tatoeba corpus.

    :param tatoeba_path: Tatoeba French sentence TSV.
    :param out_path: Output TSV file.
    :returns: Number of written rows.
    """
    return cullflow.build_frequency_cache(tatoeba_path, out_path, FRENCH_CULL_PROFILE)


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
    """Give the ``French``, ``sound``, ``Back``, and spare field of one card.

    :param card: Reserve card.
    :returns: The field values in the order of the input model.
    """
    back = f"{card.english} — {card.target}"
    if card.gloss:
        back += f" ({card.gloss})"
    return [card.html, f"[sound:{card.audio}]", back, ""]


load_reserve = partial(cullflow.load_reserve, model_name=SWADESH_MODEL_NAME)

write_deck = partial(
    coreflow.write_deck,
    model_name=MODEL_NAME,
    templates=CARD_TEMPLATES,
    card_css=card_css,
    reserve_fields=_reserve_fields,
)

FRENCH_CULL_PROFILE = cullflow.CullProfile(
    tokenize_texts=tokenize_texts,
    load_frequency=load_frequency,
    frequency_ranks=cullflow.frequency_ranks,
    read_names=read_names,
    concept_forms=_concept_forms,
    model_name=MODEL_NAME,
    sentence_field=FRENCH_FIELD,
    english_field=ENGLISH_FIELD,
    audio_field=AUDIO_FIELD,
    clean_sentence=cullflow.plain_sentence,
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
    wordfreq_lang="fr",
    default_deck_name=DEFAULT_DECK_NAME,
    licenses=LICENSES,
)


def run_cull(request: cullflow.CullRequest) -> cullflow.CullRun:
    """Run the French cull and write the output files.

    :param request: Options of the run.
    :returns: The written paths and the counts of the run.
    """
    return cullflow.run_cull(request, FRENCH_CULL_PROFILE)


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
swadesh_coverage = partial(cullflow.swadesh_coverage, profile=FRENCH_CULL_PROFILE)
filter_cards = cardfilter.filter_cards
_filter_reason = cardfilter._filter_reason
TargetWord = corelist.TargetWord
build_targets = corelist.build_targets
write_manifest = coreflow.write_manifest


run_gates = cullgates.run_gates
GateInputs = cullgates.CullFiles
gate_inputs = cullgates.gate_inputs


def check_cull(output_path: Path) -> int:
    """Run gates G1 to G15 on the files of a cull output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return cullgates.check_cull(output_path, FRENCH_CULL_PROFILE)




