"""Spanish core deck: profile and bound wrappers.

The module builds the core deck of the Spanish sentence deck
``apkg/Spanish 9000 sentences with native audio.apkg``. The Swadesh rows of
``vocab/spanish_words.csv`` give the tier S targets, and the tier F targets
come from ``DEFAULT_CORE_WORDS`` words of the frequency list. Their sense rows
live in ``vocab/spanish_core_senses.tsv``. The generic logic lives in
``anki_tools.ranking.coreflow`` and ``anki_tools.ranking.coregates``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from anki_tools.languages.spanish.frequency import (
    FREQUENCY_PATH,
    frequency_ranks_es,
    load_frequency_es,
)

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.spanish.profile import (
    TOKENIZER,
    SPANISH,
    load_word_table,
    prepare_text,
)
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.pack import load_css
from anki_tools.sentence_pack.tokens import tokenize

MODEL_NAME = "Spanish sentences cloze+"
SENTENCE_FIELD = "SpanishSentence"
ENGLISH_FIELD = "EnglishSentence"
AUDIO_FIELD = "Audio"
CORE_FIELD = "Core"
CORE_GLOSS_FIELD = "CoreGloss"

SENSE_PATH = Path("vocab/spanish_core_senses.tsv")
LOAN_PATH = Path("vocab/spanish_loanwords.txt")

DEFAULT_NAME = "Spanish Core Sentences"
DEFAULT_CORE_WORDS = 50
# A tier F card can hold one content lemma, so the selection deduplicates on a
# lower Jaccard limit than the gate K6 limit of 0.67.
DEFAULT_OPTIONS = CoreOptions(tail_cards=0, hard_top=50_000, similarity=0.5)

MIN_UNITS = 4
MAX_UNITS = 10

TEMPLATE_DIR = TEMPLATES_DIR / "spanish_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

CoreRequest = coreflow.CoreRequest
CoreRun = coreflow.CoreRun
default_output = coreflow.default_output


@dataclass(frozen=True)
class InputCard:
    """One card of the Spanish sentence deck.

    :param note_id: Anki note id.
    :param sentence: Spanish sentence text.
    :param english: English translation.
    :param audio: Name of the audio file in the card.
    :param index: Position of the card in the input order.
    """

    note_id: int
    sentence: str
    english: str
    audio: str
    index: int


_BOLD_PATTERN = re.compile(r"</?b>")


def _plain_sentence(field_value: str) -> str:
    """Remove the bold markup of a sentence field value."""
    return _BOLD_PATTERN.sub("", field_value).strip()


def _audio_name(field_value: str) -> str:
    """Read the audio file name from a sound field."""
    import re

    match = re.search(r"\[sound:([^\]]+)\]", field_value)
    return match.group(1) if match else ""


def load_input_deck(path: Path) -> list[InputCard]:
    """Read the cards of the Spanish sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``Spanish sentences cloze+`` model.
    :returns: One card per note, in the input order.
    """
    with AnkiPackage(path) as package:
        models = package.get_models()
        model_id = coreflow.find_model_id(models, MODEL_NAME)
        fields = [field["name"] for field in models[model_id]["flds"]]
        index = {name: position for position, name in enumerate(fields)}
        sentence = index[SENTENCE_FIELD]
        english = index[ENGLISH_FIELD]
        audio = index[AUDIO_FIELD]

        chosen: dict[int, Any] = {}
        for row in package.get_cards():
            if str(row["mid"]) != model_id:
                continue
            if len(row["flds"].split("\x1f")) != len(fields):
                continue
            note_id = int(row["nid"])
            current = chosen.get(note_id)
            if current is None or int(row["ord"]) < int(current["ord"]):
                chosen[note_id] = row

        cards: list[InputCard] = []
        for note_id, row in chosen.items():
            values = row["flds"].split("\x1f")
            cards.append(
                InputCard(
                    note_id=note_id,
                    sentence=prepare_text(_plain_sentence(values[sentence])),
                    english=values[english].strip(),
                    audio=_audio_name(values[audio]),
                    index=len(cards),
                )
            )
        return cards


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
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


def load_frequency() -> list[Any]:
    """Give the Spanish frequency cache rows."""
    return load_frequency_es()


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
    from anki_tools.languages.spanish.profile import EXTRA_CSS

    return load_css(SHARED_CSS_PATH, EXTRA_CSS)


def run_core(request: CoreRequest) -> CoreRun:
    """Run the Spanish core mode and write the output files."""
    return coreflow.run_core(request, SPANISH_CORE_PROFILE)


def check_core(output_path: Path) -> int:
    """Run gates K1 to K20 on the files of a Spanish core output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    from anki_tools.ranking import coregates

    return coregates.check_core(Path(output_path), SPANISH_CORE_PROFILE)


SPANISH_CORE_PROFILE = CoreProfile(
    load_input_deck=load_input_deck,
    tokenize_deck=tokenize_deck,
    load_frequency=load_frequency,
    frequency_ranks=frequency_ranks,
    read_names=read_names,
    word_table=word_table,
    sense_path=SENSE_PATH,
    loan_path=LOAN_PATH,
    has_orphan=SPANISH.has_orphan,
    hits=SPANISH.hits,
    sense_filter=SPANISH.sense_filter,
    highlight=SPANISH.highlight,
    lemma_key=str.casefold,
    model_name=MODEL_NAME,
    sentence_field=SENTENCE_FIELD,
    english_field=ENGLISH_FIELD,
    audio_field=AUDIO_FIELD,
    core_field=CORE_FIELD,
    core_gloss_field=CORE_GLOSS_FIELD,
    templates=CORE_TEMPLATES,
    card_css=card_css,
    min_units=MIN_UNITS,
    max_units=MAX_UNITS,
    default_options=DEFAULT_OPTIONS,
    deck_name=DEFAULT_NAME,
    frequency_source=str(FREQUENCY_PATH),
)
