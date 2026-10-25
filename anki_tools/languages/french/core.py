"""French core deck: profile, constants, bound wrappers, and gate check.

The module holds the French values of the shared core flow, the
:class:`anki_tools.ranking.scan.CoreProfile` that ties them together, and the
French-bound reader and gate check. The generic logic lives in
``anki_tools.ranking.coreflow`` and ``anki_tools.ranking.coregates``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

from anki_tools.apkg.templates import TEMPLATES_DIR, load_text
from anki_tools.languages.french import cull
from anki_tools.languages.french.profile import FRENCH, load_word_table
from anki_tools.ranking import coreflow
from anki_tools.ranking import coregates
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.corelist import CoreTarget, SenseWords
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.audit import print_audit

SENSE_PATH = Path("vocab/french_core_senses.tsv")
LOAN_PATH = Path("vocab/french_loanwords.txt")
DEFAULT_NAME = "French Core Sentences"
DEFAULT_CORE_WORDS = 50
DEFAULT_OPTIONS = CoreOptions(tail_cards=120)

MIN_UNITS = 5
MAX_UNITS = 9

MODEL_NAME = cull.MODEL_NAME
FRENCH_FIELD = cull.FRENCH_FIELD
AUDIO_FIELD = cull.AUDIO_FIELD
ENGLISH_FIELD = cull.ENGLISH_FIELD

CORE_FIELD = "Core"
CORE_GLOSS_FIELD = "CoreGloss"

TEMPLATE_DIR = TEMPLATES_DIR / "french_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

# Re-export the three members that the CLI reads from a language module.
CoreRequest = coreflow.CoreRequest
CoreRun = coreflow.CoreRun
default_output = coreflow.default_output

def word_table() -> list[tuple[int, str, str]]:
    """Give the reviewed French Swadesh rows as ``(index, form, gloss)``."""
    return [
        (row.swadesh_index, row.french, row.english) for row in load_word_table()
    ]


def read_senses(path: Path = SENSE_PATH) -> dict[str, SenseWords]:
    """Read the French tier F sense file."""
    return coreflow.read_senses(path)


def read_loanwords(path: Path = LOAN_PATH) -> frozenset[str]:
    """Read the French loan word list."""
    return coreflow.read_loanwords(path)


def filter_core_cards(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    media: set[str],
    names: Collection[str],
    loanwords: Collection[str],
    *,
    min_units: int = MIN_UNITS,
    max_units: int = MAX_UNITS,
) -> tuple[list[int], list[str]]:
    """Apply rules F1 to F9 and C1 to C5 to the French cards."""
    return coreflow.filter_core_cards(
        cards,
        tokens,
        media,
        names,
        loanwords,
        FRENCH_CORE_PROFILE.has_orphan,
        min_units=min_units,
        max_units=max_units,
    )


def _write_core_model(
    output_path: Path,
    cards: Sequence[Any],
    result: coreflow.CoreResult,
    ordered: Sequence[int],
    targets: Sequence[CoreTarget],
    candidates: Sequence[coreflow.CoreCandidate],
) -> int:
    """Set the French core card templates and fill the core word field."""
    return coreflow._write_core_model(
        output_path, cards, result, ordered, targets, candidates, FRENCH_CORE_PROFILE
    )


def run_core(request: CoreRequest) -> CoreRun:
    """Run the French core mode and write the output files."""
    return coreflow.run_core(request, FRENCH_CORE_PROFILE)


FRENCH_CORE_PROFILE = CoreProfile(
    load_input_deck=cull.load_input_deck,
    tokenize_deck=cull.tokenize_deck,
    load_frequency=cull.load_frequency,
    frequency_ranks=cull.frequency_ranks,
    read_names=cull.read_names,
    word_table=word_table,
    sense_path=SENSE_PATH,
    loan_path=LOAN_PATH,
    has_orphan=FRENCH.has_orphan,
    hits=FRENCH.hits,
    sense_filter=FRENCH.sense_filter,
    highlight=FRENCH.highlight,
    lemma_key=str.casefold,
    model_name=MODEL_NAME,
    sentence_field=FRENCH_FIELD,
    english_field=ENGLISH_FIELD,
    audio_field=AUDIO_FIELD,
    core_field=CORE_FIELD,
    core_gloss_field=CORE_GLOSS_FIELD,
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


class CoreFiles(coregates.CoreFiles):
    """French-bound lazy reader of the core output files."""

    def __init__(self, apkg_path: Path) -> None:
        super().__init__(Path(apkg_path), FRENCH_CORE_PROFILE)


def check_core(output_path: Path) -> int:
    """Run gates K1 to K20 on the files of a French core output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return print_audit(coregates.run_gates(CoreFiles(Path(output_path))))
