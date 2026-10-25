"""Cantonese core deck: profile and bound wrappers.

The module builds the core deck of the Cantonese sentence deck
``apkg/LTL Cantonese Deck Level 1 - Short.apkg``. The Swadesh rows of
``vocab/english-cantonese.csv`` give the tier S targets, and the tier F targets
come from ``DEFAULT_CORE_WORDS`` words of the frequency list. Their sense rows
live in ``vocab/cantonese_core_senses.tsv``. The generic logic lives in
``anki_tools.ranking.coreflow`` and ``anki_tools.ranking.coregates``.

A Cantonese sentence has no part-of-speech tagger, so every Han token gets the
constant tag ``TOKEN_POS``. The tag is a content tag and passes rule C1.

The shared filter F4 reads an ASCII final mark. The adapter gives the final
mark of a sentence its ASCII form and writes that text back to the kept notes.
The model also has an English sound field that the core templates do not use,
so the run blanks that field and removes its media files.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.cantonese.frequency import (
    FREQUENCY_PATH,
    YueFrequency,
    load_yue_frequency,
    read_frequency_tsv,
)
from anki_tools.languages.cantonese.profile import (
    CANTONESE,
    EXTRA_CSS,
    han_tokens,
    load_word_table,
    prepare_text,
)
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.pack import load_css

MODEL_NAME = "LTL Note Type - Cantonese"
SENTENCE_FIELD = "Front"
ENGLISH_FIELD = "Back"
ENGLISH_AUDIO_FIELD = "Eng Audio"
AUDIO_FIELD = "Audio"
CORE_FIELD = "Core"
CORE_GLOSS_FIELD = "CoreGloss"

SENSE_PATH = Path("vocab/cantonese_core_senses.tsv")
LOAN_PATH = Path("vocab/cantonese_loanwords.txt")
NAMES_PATH = Path("vocab/cantonese_names.tsv")

DEFAULT_NAME = "Cantonese Core Sentences"
# The tier F sense rows are not authored yet (anki-eiy). Raise this to 50 with
# the rows in vocab/cantonese_core_senses.tsv.
DEFAULT_CORE_WORDS = 0
# The input deck is small, so the selection deduplicates below the Jaccard limit
# of gate K6 and caps the outside vocabulary of the kept cards.
DEFAULT_OPTIONS = CoreOptions(
    tail_cards=0,
    hard_top=50_000,
    similarity=0.65,
    outside_budget=165,
)

#: Blend weight that the cached Cantonese frequency list holds.
SPOKEN_WEIGHT = 0.7

# Every Han token gets this tag: Cifu and jieba give no part of speech, and rule
# C1 accepts only a verb-like tag, which is a content tag too.
TOKEN_POS = "VERB"

MIN_UNITS = 2
MAX_UNITS = 7

TEMPLATE_DIR = TEMPLATES_DIR / "cantonese_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

CoreRequest = coreflow.CoreRequest
CoreRun = coreflow.CoreRun
default_output = coreflow.default_output

_SOUND_PATTERN = re.compile(r"\[sound:([^\]]+)\]")


InputCard = coreflow.InputCard

_BOLD_PATTERN = re.compile(r"</?b>")
_FINAL_MARKS = {"。": ".", "？": "?", "！": "!"}


def _sentence(field_value: str) -> str:
    """Read the card text from the sentence field.

    The function drops the bold tags that a previous core run wrote, prepares
    the Traditional text, and gives the final mark the ASCII form, because the
    shared filter F4 accepts only ``.``, ``?``, and ``!``.

    :param field_value: Raw value of the sentence field.
    :returns: Traditional text with an ASCII final mark.
    """
    text = prepare_text(_BOLD_PATTERN.sub("", field_value))
    if text and text[-1] in _FINAL_MARKS:
        return text[:-1] + _FINAL_MARKS[text[-1]]
    return text

def load_input_deck(path: Path) -> list[InputCard]:
    """Read the cards of the Cantonese sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``LTL Note Type - Cantonese`` model.
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


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card with the Cantonese tokenizer.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card.
    """
    return [
        [(token, TOKEN_POS) for token in han_tokens(card.sentence)]
        for card in cards
    ]


def load_frequency() -> YueFrequency:
    """Give the Cantonese frequency list.

    The frequency list TSV is read when it exists, because it is faster
    than the blend. Otherwise the blend is built from the Cifu and Apple Daily
    sources.

    :returns: The ranked list with the variant map.
    """
    if FREQUENCY_PATH.exists():
        return read_frequency_tsv(FREQUENCY_PATH)
    return load_yue_frequency()


def frequency_ranks(
    rows: YueFrequency, spoken_weight: float = 0.7
) -> dict[str, int]:
    """Map each Cantonese word to its 1-based rank.

    :param rows: Frequency rows from :func:`load_frequency`.
    :param spoken_weight: Unused: the cached list already holds the blend.
    :returns: Word to rank, with each variant on the rank of its standard form.
    """
    return dict(rows.ranks)


def read_names() -> set[str]:
    """Read the Cantonese name stoplist of rule C3.

    :returns: Names from the first column of the TSV, empty when the file is
        missing.
    """
    source = Path(NAMES_PATH)
    if not source.exists():
        return set()
    names: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines()[1:]:
        name = line.split("\t")[0].strip()
        if name:
            names.add(name)
    return names


def word_table() -> list[tuple[int, str, str]]:
    """Give the reviewed Cantonese Swadesh rows as ``(index, form, gloss)``."""
    return [
        (row.swadesh_index, row.cantonese, row.english)
        for row in load_word_table()
    ]


def card_css() -> str:
    """Give the CSS of the Cantonese core deck."""
    return load_css(SHARED_CSS_PATH, EXTRA_CSS)


def finalize_output(output_path: Path, core_csv: Path) -> int:
    """Rewrite the kept notes that the shared flow leaves in the input text.

    The flow writes bold text, but a card without an assigned target keeps the
    input sentence, and the shared rules read an ASCII final mark. The function
    also blanks the English sound field, because the core templates do not use
    it and gate K8 counts one sound per card.

    :param output_path: Output APKG file.
    :param core_csv: Card CSV of the run.
    :returns: The count of rewritten notes.
    """
    with Path(core_csv).open(encoding="utf-8", newline="") as handle:
        rows = {
            int(row["note_id"]): row["sentence"]
            for row in csv.DictReader(handle)
            if row["kept"] == "1"
        }
    if not rows:
        return 0
    written = 0
    with AnkiPackage(output_path) as package:
        models = package.get_models()
        model_id = coreflow.find_model_id(models, MODEL_NAME)
        names = [field["name"] for field in models[model_id]["flds"]]
        sentence_index = names.index(SENTENCE_FIELD)
        sound_index = (
            names.index(ENGLISH_AUDIO_FIELD)
            if ENGLISH_AUDIO_FIELD in names
            else -1
        )
        for note in package.get_notes():
            note_id = int(note["id"])
            sentence = rows.get(note_id)
            if sentence is None:
                continue
            values = str(note["flds"]).split("\x1f")
            changed = False
            text = values[sentence_index]
            if "<b>" not in text and _BOLD_PATTERN.sub("", text) != sentence:
                package.update_note_field(note_id, sentence_index, sentence)
                changed = True
            if sound_index >= 0:
                sounds = _SOUND_PATTERN.findall(values[sound_index])
                if sounds:
                    package.update_note_field(note_id, sound_index, "")
                    changed = True
                    for sound in sounds:
                        package.delete_media_file(sound)
            written += int(changed)
        package._modified = True
        package.save(output_path)
    return written


def run_core(request: CoreRequest) -> CoreRun:
    """Run the Cantonese core mode and write the output files."""
    run = coreflow.run_core(request, CANTONESE_CORE_PROFILE)
    finalize_output(run.output_path, run.core_csv)
    return run


def check_core(output_path: Path) -> int:
    """Run gates K1 to K20 on the files of a Cantonese core output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    from anki_tools.ranking import coregates

    return coregates.check_core(Path(output_path), CANTONESE_CORE_PROFILE)


CANTONESE_CORE_PROFILE = CoreProfile(
    load_input_deck=load_input_deck,
    tokenize_deck=tokenize_deck,
    load_frequency=load_frequency,
    frequency_ranks=frequency_ranks,
    read_names=read_names,
    word_table=word_table,
    sense_path=SENSE_PATH,
    loan_path=LOAN_PATH,
    has_orphan=CANTONESE.has_orphan,
    hits=CANTONESE.hits,
    sense_filter=CANTONESE.sense_filter,
    highlight=CANTONESE.highlight,
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
