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
from anki_tools.ranking.core import CoreOptions

SENSE_PATH = cull.SENSE_PATH
LOAN_PATH = Path("vocab/french_loanwords.txt")
DEFAULT_CORE_WORDS = 50
DEFAULT_OPTIONS = CoreOptions(tail_cards=120)


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



FRENCH_CORE_PROFILE = cull.FRENCH_LANGUAGE.core(
    word_table=word_table,
    loan_path=LOAN_PATH,
    hits=FRENCH.hits,
    lemma_key=str.casefold,
    templates=CORE_TEMPLATES,
    final_marks=FRENCH.final_marks,
    quote_marks=FRENCH.quote_marks,
    internal_marks=FRENCH.internal_marks,
    default_options=DEFAULT_OPTIONS,
    frequency_source=str(cull.FREQUENCY_PATH),
)
