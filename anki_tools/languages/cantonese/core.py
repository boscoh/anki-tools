"""Cantonese core deck: profile and values.

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
The model also has an unused English sound field, which the output note type
does not carry.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.cantonese.frequency import (
    YueFrequency,
    load_yue_frequency,
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
from anki_tools.ranking.deckprofile import LanguageDeckProfile
from anki_tools.sentence_pack.pack import load_css

MODEL_NAME = "LTL Note Type - Cantonese"
SENTENCE_FIELD = "Front"
ENGLISH_FIELD = "Back"
AUDIO_FIELD = "Audio"

SENSE_PATH = Path("vocab/cantonese_core_senses.tsv")
LOAN_PATH = Path("vocab/cantonese_loanwords.txt")
NAMES_PATH = Path("vocab/cantonese_names.tsv")

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

def load_input_deck(path: Path) -> list[coreflow.InputCard]:
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


def tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize sentence texts with the Cantonese tokenizer.

    :param texts: Sentence texts.
    :returns: One list of ``(lemma, part of speech)`` pairs per text.
    """
    return [[(token, TOKEN_POS) for token in han_tokens(text)] for text in texts]


def load_frequency() -> YueFrequency:
    """Give the Cantonese frequency list, built from the Cifu and Apple Daily sources.

    :raises RuntimeError: A gate of the blend fails.
    :returns: The ranked list with the variant map.
    """
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
    input sentence, and the shared rules read an ASCII final mark.

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
        model_id = coreflow.find_only_model_id(models)
        names = [field["name"] for field in models[model_id]["flds"]]
        sentence_index = names.index(coreflow.OUTPUT_SENTENCE_FIELD)
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
            written += int(changed)
        package.touch()
        package.save(output_path)
    return written

CANTONESE_LANGUAGE = LanguageDeckProfile(
    load_input_deck=load_input_deck,
    tokenize_texts=tokenize_texts,
    load_frequency=load_frequency,
    frequency_ranks=frequency_ranks,
    read_names=read_names,
    input_model_name=MODEL_NAME,
    input_sentence_field=SENTENCE_FIELD,
    input_english_field=ENGLISH_FIELD,
    input_audio_field=AUDIO_FIELD,
    clean_sentence=_sentence,
    card_css=card_css,
    sense_path=SENSE_PATH,
    highlight=CANTONESE.highlight,
    romanization_source="Jyutping",
)

CANTONESE_CORE_PROFILE = CANTONESE_LANGUAGE.core(
    word_table=word_table,
    loan_path=LOAN_PATH,
    has_orphan=CANTONESE.has_orphan,
    hits=CANTONESE.hits,
    sense_filter=CANTONESE.sense_filter,
    lemma_key=str.casefold,
    templates=CORE_TEMPLATES,
    min_units=MIN_UNITS,
    max_units=MAX_UNITS,
    default_options=DEFAULT_OPTIONS,
    frequency_source="Cifu and Apple Daily blend",
    outside_set_ratio=1.3,
    outside_rare_ratio=0.1,
    spoken_weight=SPOKEN_WEIGHT,
    finalize=lambda run: finalize_output(run.output_path, run.core_csv),
)
