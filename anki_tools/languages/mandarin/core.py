"""Mandarin core deck: profile and values.

The module builds the core deck of the Mandarin sentence deck
``apkg/Neri's Chinese Sents Part 2.apkg``. The Swadesh rows of
``vocab/english-mandarin.csv`` give the tier S targets, and the tier F targets
come from ``DEFAULT_CORE_WORDS`` words of the frequency cache
``data/mandarin_frequency.tsv``. The generic logic lives in
``anki_tools.ranking.coreflow`` and ``anki_tools.ranking.coregates``.

The tier S form unit is the reviewed word-table form, which is one jieba word.
The hit rule reads the form as a character span in the Han text and then lets
the English gloss of the row decide. The character span keeps the 25 forms that
jieba merges into a longer token, for example 水 inside 喝水, and the guard
rejects the cards where the form takes another sense. A character bigram holds
143 one-character forms zero times, so the module does not use a bigram.
``FALSE_FRIENDS`` blocks the spans where a form is a part of a longer word
with another sense, so 来 does not hit inside 出来 and 起来.

The card count sets the ceiling of the deck. The run holds 140 targets: 90
tier S forms with a card and the 50 tier F words. Gate K3 reports a two-card
share of 0.82 and does not fail.

The tier F sense rows live in ``vocab/mandarin_core_senses.tsv``. Each row
holds the English words that the paired sentences of the deck use for the
word. The first sense gives the ``CoreGloss``. The deck also holds nine loan
words, and ``vocab/mandarin_loanwords.txt`` lists them for rule C4.

The audio of the deck is machine speech from Google TTS on all 5009 notes, and
the model holds no native recording. The manifest and ``ATTRIBUTION.md`` must
name the engine before the deck ships.

The shared filter F4 reads an ASCII final mark. The adapter gives the final mark
of a sentence its ASCII form and appends one when the text has none, then
writes that text back to the kept notes.

The jieba part-of-speech flags map to the shared content tags through
:func:`anki_tools.languages.mandarin.profile.shared_pos`, so rule C1 needs a
verb or an adjective and the tier F word list stays clear of function words.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import SHARED_CSS_PATH, TEMPLATES_DIR, load_text
from anki_tools.languages.mandarin.frequency import (
    FREQUENCY_PATH,
    MAX_ROWS,
    MandarinFrequencyRow,
    build_frequency_zh,
    check_frequency,
    frequency_ranks_zh,
    load_frequency_zh,
    write_frequency_tsv,
)
from anki_tools.languages.mandarin.profile import (
    EXTRA_CSS,
    MANDARIN,
    convert_to_simplified,
    load_user_dictionary,
    load_word_table,
    tokens_with_pos,
)
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.deckprofile import LanguageDeckProfile
from anki_tools.apkg.templates import load_css

MODEL_NAME = "0 2024 Sentences Deck"
SENTENCE_FIELD = "Sentence"
ENGLISH_FIELD = "Sentence (Translation)"
AUDIO_FIELD = "[a]"

SENSE_PATH = Path("vocab/mandarin_core_senses.tsv")
LOAN_PATH = Path("vocab/mandarin_loanwords.txt")

# The reviewed tier F word count of the deck. Every row of
# vocab/mandarin_core_senses.tsv holds a sense, so the 50 targets are checked.
DEFAULT_CORE_WORDS = 50
# The deck holds 2 to 9 jieba tokens in a sentence. The selection lowers the
# Jaccard limit below the gate K6 limit, because a tier F card holds one content
# lemma, and it caps the outside vocabulary of the kept cards.
DEFAULT_OPTIONS = CoreOptions(
    tail_cards=0,
    hard_top=50_000,
    similarity=0.65,
    outside_budget=200,
)

MIN_UNITS = 2
MAX_UNITS = 8

TEMPLATE_DIR = TEMPLATES_DIR / "mandarin_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

_BOLD_PATTERN = re.compile(r"</?b>")
_FINAL_MARKS = {"。": ".", "？": "?", "！": "!"}
_ASCII_FINAL_MARKS = frozenset(".?!")

_dictionary_ready = False


def _sentence(field_value: str) -> str:
    """Read the card text from the sentence field.

    The function drops the bold tags that a previous core run wrote, converts
    the text to Simplified Chinese, and gives the final mark the ASCII form.
    The shared filter F4 accepts only ``.``, ``?``, and ``!``, and most
    sentences of the deck end in no mark at all, so a text with no ASCII final
    mark gets a full stop.

    :param field_value: Raw value of the sentence field.
    :returns: Simplified text with an ASCII final mark.
    """
    text = convert_to_simplified(_BOLD_PATTERN.sub("", field_value)).strip()
    if not text:
        return text
    last = _FINAL_MARKS.get(text[-1])
    if last is not None:
        return text[:-1] + last
    if text[-1] not in _ASCII_FINAL_MARKS:
        return text + "."
    return text


def _ensure_dictionary() -> None:
    """Add the word-table forms to the jieba user dictionary once per process.

    The forms steer the segmenter, so every tokenizer call of the run and of
    the gates must see the same dictionary.
    """
    global _dictionary_ready
    if not _dictionary_ready:
        load_user_dictionary(load_word_table())
        _dictionary_ready = True


def load_input_deck(path: Path) -> list[coreflow.InputCard]:
    """Read the cards of the Mandarin sentence model from an APKG file.

    The package also holds two ``TEXT CARD`` notes of another model, so only the
    cards of ``0 2024 Sentences Deck`` enter the list.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``0 2024 Sentences Deck`` model.
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
    """Tokenize sentence texts with jieba.

    :param texts: Sentence texts.
    :returns: One list of ``(token, part of speech)`` pairs per text.
    """
    _ensure_dictionary()
    return [tokens_with_pos(text) for text in texts]


def load_frequency() -> list[MandarinFrequencyRow]:
    """Give the Mandarin frequency cache rows, and build the file when it is absent.

    :raises RuntimeError: The build fails a gate.
    :returns: Rows in rank order.
    """
    if FREQUENCY_PATH.exists():
        return load_frequency_zh(FREQUENCY_PATH)
    rows = build_frequency_zh(rows=MAX_ROWS)
    failures = [
        f"{gate}: {message}"
        for gate, passed, message in check_frequency(rows)
        if not passed
    ]
    if failures:
        raise RuntimeError(
            f"The Mandarin frequency list {FREQUENCY_PATH} fails its gates: "
            + "; ".join(failures)
        )
    write_frequency_tsv(rows, FREQUENCY_PATH)
    return rows


def frequency_ranks(
    rows: Sequence[MandarinFrequencyRow], spoken_weight: float = 0.5
) -> dict[str, int]:
    """Map each Mandarin form to its 1-based rank.

    :param rows: Frequency cache rows.
    :param spoken_weight: Unused, kept for the profile contract.
    :returns: Lemma to rank.
    """
    return frequency_ranks_zh(rows)


def read_names() -> set[str]:
    """Give an empty Mandarin name stoplist."""
    return set()


def word_table() -> list[tuple[int, str, str]]:
    """Give the reviewed Mandarin Swadesh rows as ``(index, form, gloss)``."""
    return [
        (row.swadesh_index, row.mandarin, row.english)
        for row in load_word_table()
    ]


def card_css() -> str:
    """Give the CSS of the Mandarin core deck."""
    return load_css(SHARED_CSS_PATH, EXTRA_CSS)


def finalize_output(output_path: Path, core_csv: Path) -> int:
    """Rewrite the kept notes that the shared flow leaves in the input text.

    The flow writes bold text only for a card with an assigned target, and a
    card without one keeps the input sentence. The shared rules read an ASCII
    final mark, so the function writes the adapter text of :func:`_sentence` to
    every kept note.

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
            text = values[sentence_index]
            if "<b>" not in text and _BOLD_PATTERN.sub("", text) != sentence:
                package.update_note_field(note_id, sentence_index, sentence)
                written += 1
        package.touch()
        package.save(output_path)
    return written

MANDARIN_LANGUAGE = LanguageDeckProfile(
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
    highlight=MANDARIN.highlight,
    romanization_source="Sentence (Latin)",
)

MANDARIN_CORE_PROFILE = MANDARIN_LANGUAGE.core(
    word_table=word_table,
    loan_path=LOAN_PATH,
    has_orphan=MANDARIN.has_orphan,
    hits=MANDARIN.hits,
    sense_filter=MANDARIN.sense_filter,
    lemma_key=convert_to_simplified,
    templates=CORE_TEMPLATES,
    min_units=MIN_UNITS,
    max_units=MAX_UNITS,
    default_options=DEFAULT_OPTIONS,
    frequency_source=str(FREQUENCY_PATH),
    finalize=lambda run: finalize_output(run.output_path, run.card_csv),
)
