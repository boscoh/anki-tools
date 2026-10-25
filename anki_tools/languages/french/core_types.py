"""French core deck adapter: profile, constants, and bound wrappers.

The module holds the French-specific values of the shared core flow and the
:class:`anki_tools.ranking.scan.CoreProfile` that ties them together. The
generic logic lives in ``anki_tools.ranking.coreflow`` and
``anki_tools.ranking.coregates``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

from anki_tools.apkg.templates import TEMPLATES_DIR, load_text
from anki_tools.languages.french import cull
from anki_tools.languages.french.profile import FRENCH, load_word_table
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.corelist import (
    CoreTarget,
    SenseWords,
    build_core_targets,
)
from anki_tools.ranking.scan import CoreProfile

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

# Re-export the generic flow surface.
CoreRequest = coreflow.CoreRequest
CoreRun = coreflow.CoreRun
sense_match = coreflow.sense_match
holds_tier_s_sense = coreflow.holds_tier_s_sense
core_hits = coreflow.core_hits
default_output = coreflow.default_output
size_check = coreflow.size_check
outside_lemmas = coreflow.outside_lemmas
side_paths = coreflow.side_paths
write_rank_csv = coreflow.write_rank_csv
kept_order_hash = coreflow.kept_order_hash
card_rows = coreflow.card_rows
coverage_rows_of = coreflow.coverage_rows_of
manifest_counts = coreflow.manifest_counts
write_core_csv = coreflow.write_core_csv
write_coverage_csv = coreflow.write_coverage_csv
write_manifest = coreflow.write_manifest
CORE_COLUMNS = coreflow.CORE_COLUMNS
COVERAGE_COLUMNS = coreflow.COVERAGE_COLUMNS
FILTER_KEYS = coreflow.FILTER_KEYS
TARGET_SEPARATOR = coreflow.TARGET_SEPARATOR
STATUSES = coreflow.STATUSES
__all__ = [
    "AUDIO_FIELD",
    "CORE_BACK_TEMPLATE",
    "CORE_COLUMNS",
    "CORE_FIELD",
    "CORE_FRONT_TEMPLATE",
    "CORE_GLOSS_FIELD",
    "CORE_TEMPLATES",
    "COVERAGE_COLUMNS",
    "CoreRequest",
    "CoreRun",
    "CoreTarget",
    "DEFAULT_CORE_WORDS",
    "DEFAULT_NAME",
    "DEFAULT_OPTIONS",
    "ENGLISH_FIELD",
    "FILTER_KEYS",
    "FRENCH_CORE_PROFILE",
    "FRENCH_FIELD",
    "LOAN_PATH",
    "MAX_UNITS",
    "MIN_UNITS",
    "MODEL_NAME",
    "SENSE_PATH",
    "STATUSES",
    "SenseWords",
    "TARGET_SEPARATOR",
    "TEMPLATE_DIR",
    "build_core_targets",
    "card_rows",
    "core_hits",
    "coverage_rows_of",
    "default_output",
    "filter_core_cards",
    "holds_tier_s_sense",
    "kept_order_hash",
    "manifest_counts",
    "outside_lemmas",
    "read_loanwords",
    "read_senses",
    "run_core",
    "sense_match",
    "side_paths",
    "size_check",
    "word_table",
    "write_core_csv",
    "write_coverage_csv",
    "write_manifest",
    "write_rank_csv",
]

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
