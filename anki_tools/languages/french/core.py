"""French core deck: profile, constants, and values.

The module holds the French values of the shared core flow and the
:class:`anki_tools.ranking.scan.CoreProfile` that ties them together. The
generic logic lives in ``anki_tools.ranking.coreflow`` and
``anki_tools.ranking.coregates``.
"""

from __future__ import annotations

from pathlib import Path

from anki_tools.apkg.templates import TEMPLATES_DIR, load_text
from anki_tools.languages.french import cull
from anki_tools.languages.french.profile import FRENCH, load_word_table
from anki_tools.ranking import cullflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.scan import CoreProfile

SENSE_PATH = cull.SENSE_PATH
LOAN_PATH = Path("vocab/french_loanwords.txt")
DEFAULT_NAME = "French Core Sentences"
DEFAULT_CORE_WORDS = 50
DEFAULT_OPTIONS = CoreOptions(tail_cards=120)

MIN_UNITS = 5
MAX_UNITS = 9


TEMPLATE_DIR = TEMPLATES_DIR / "french_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)


def word_table() -> list[tuple[int, str, str]]:
    """Give the reviewed French Swadesh rows as ``(index, form, gloss)``."""
    return [
        (row.swadesh_index, row.french, row.english) for row in load_word_table()
    ]



FRENCH_CORE_PROFILE = CoreProfile(
    load_input_deck=cull.load_input_deck,
    tokenize_deck=cull.tokenize_deck,
    load_frequency=cull.FRENCH_CULL_PROFILE.load_frequency,
    frequency_ranks=cullflow.frequency_ranks,
    read_names=cull.FRENCH_CULL_PROFILE.read_names,
    word_table=word_table,
    sense_path=SENSE_PATH,
    loan_path=LOAN_PATH,
    has_orphan=FRENCH.has_orphan,
    hits=FRENCH.hits,
    sense_filter=FRENCH.sense_filter,
    highlight=FRENCH.highlight,
    lemma_key=str.casefold,
    input_model_name=cull.MODEL_NAME,
    input_sentence_field=cull.FRENCH_FIELD,
    input_english_field=cull.ENGLISH_FIELD,
    input_audio_field=cull.AUDIO_FIELD,
    output_model_name=DEFAULT_NAME,
    templates=CORE_TEMPLATES,
    card_css=cull.card_css,
    min_units=MIN_UNITS,
    max_units=MAX_UNITS,
    default_options=DEFAULT_OPTIONS,
    deck_name=DEFAULT_NAME,
    frequency_source=str(cull.FREQUENCY_PATH),
    outside_set_ratio=1.3,
    outside_rare_ratio=0.1,
)
