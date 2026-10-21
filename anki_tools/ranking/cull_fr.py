"""Cull the French sentence deck to a smaller deck.

The pipeline extracts the input cards, ranks French lemmas by a blend of the
Tatoeba counts and the ``wordfreq`` list, tokenizes the cards with spaCy, and
builds the target word list. It then filters the cards, selects a greedy cover
of the target words, and orders the kept cards for learning.

See ``specs/french-1000-sentence-cull.md`` sections 4 to 9.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import re
import sqlite3
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.sentence_pack.audit import sha256_value
from anki_tools.sentence_pack.audio import probe_duration
from anki_tools.sentence_pack.languages.french import EXTRA_CSS, TOKENIZER
from anki_tools.sentence_pack.pack import load_css

FREQUENCY_PATH = Path("data/french_frequency.tsv")
NAMES_PATH = Path("vocab/french_names.txt")
TATOEBA_PATH = Path("data/tatoeba/fra_sentences.tsv")

MODEL_NAME = "French Sentences"
FRENCH_FIELD = "French"
AUDIO_FIELD = "sound"
ENGLISH_FIELD = "Back"

SWADESH_PACK_PATH = Path("apkg/swadesh_french_sentences.apkg")
SWADESH_MODEL_NAME = "French Swadesh Sentence"
SWADESH_SENTENCE_FIELD = "Sentence"
SWADESH_ENGLISH_FIELD = "English"
SWADESH_TARGET_FIELD = "Target"
SWADESH_GLOSS_FIELD = "Gloss"
SWADESH_AUDIO_FIELD = "Audio"
SWADESH_TOTAL = 207
SWADESH_EXCEPTIONS = ("intestins",)
RESERVE_TAGS = " swadesh "
CARD_CSS_PATH = Path(__file__).parent.parent.parent / "templates" / "card.css"
FRONT_HINT = "Tap to show the sentence"
REVERSE_HINT = "Tap to show the French"
FRONT_TEMPLATE = (
    '<div class="front-audio">{{sound}}</div>'
    f'<div class="hint">{FRONT_HINT}</div>'
)
BACK_TEMPLATE = (
    "{{sound}}"
    "<hr id=answer>"
    '<div class="sentence">{{French}}</div>'
    '<div class="english">{{Back}}</div>'
)
REVERSE_FRONT_TEMPLATE = (
    '<div class="english">{{Back}}</div>'
    f'<div class="hint">{REVERSE_HINT}</div>'
)
REVERSE_BACK_TEMPLATE = (
    "{{Back}}"
    "<hr id=answer>"
    '<div class="sentence">{{French}}</div>'
    '<div class="front-audio">{{sound}}</div>'
)
CARD_TEMPLATES = (
    (FRONT_TEMPLATE, BACK_TEMPLATE),
    (REVERSE_FRONT_TEMPLATE, REVERSE_BACK_TEMPLATE),
)

CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})
CONTENT_SHARE = 0.5
MIN_DECK_COUNT = 3
MIN_LEMMA_LENGTH = 2
SMALL_COUNT = 5
UNRANKED_RANK = 100_000
UNRANKED_ORDER = 1_000_000
TATOEBA_BATCH = 1000

FILTER_RULES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")
MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12
FINAL_MARKS = frozenset(".?!")
BRACKET_MARKS = frozenset("[]{}<>")
QUOTE_MARKS = frozenset('"«»“”')
DIALOGUE_DASHES = frozenset("-–—")
INTERNAL_MARKS = ",;:"
ELLIPSIS_MARKS = ("...", "…")
URL_MARKS = ("http", "www.")
EXEMPT_GAIN = 0.1
_SOUND_PATTERN = re.compile(r"\[sound:([^\]]+)\]")
_DIGIT_PATTERN = re.compile(r"\d")
_BOLD_PATTERN = re.compile(r"</?b>")
_NO_BREAK_SPACES = {0x00A0: " ", 0x202F: " "}
_WORD_PATTERN = re.compile(r"[^\W\d_]+")

CULL_COLUMNS = (
    "source",
    "sentence",
    "english",
    "audio",
    "kept",
    "reason",
    "targets",
    "gain",
    "hardest_rank",
)
SWADESH_SOURCE = "swadesh"
FREQUENCY_SOURCE = "frequency"
COVERAGE_COLUMNS = ("word", "rank", "count", "status")
COVERED_STATUS = "covered"
PARTIAL_STATUS = "partial"
ABSENT_STATUS = "absent"
LICENSES = {
    "tatoeba": "Tatoeba sentence text: CC BY 2.0 FR",
    "wordfreq": "wordfreq list: Apache License 2.0",
    "audio": "Input audio: license of the source deck",
}
TARGET_COVERAGE_GATE = 0.98
DOUBLE_COVERAGE_GATE = 0.95
AUDIO_PROBE_WORKERS = 8
GATE_SAMPLE = 3


@dataclass(frozen=True)
class InputCard:
    """One card of the input deck.

    :param note_id: Anki note id.
    :param sentence: French sentence text.
    :param english: English translation.
    :param audio: Name of the audio file in the card.
    :param index: Position of the card in the input order.
    """

    note_id: int
    sentence: str
    english: str
    audio: str
    index: int


@dataclass(frozen=True)
class ReserveCard:
    """One reserved Swadesh card of the cull.

    :param sentence: French sentence text without the bold markup.
    :param html: French sentence with the ``<b>`` target markup.
    :param english: English translation.
    :param target: Swadesh target word of the card.
    :param gloss: English gloss of the target word.
    :param audio: Name of the MP3 file in the Swadesh pack.
    """

    sentence: str
    html: str
    english: str
    target: str
    gloss: str
    audio: str


@dataclass(frozen=True)
class FrequencyRow:
    """One row of the cached French frequency list.

    :param word: Lemma.
    :param tatoeba_count: Occurrence count in the Tatoeba French corpus.
    :param tatoeba_pm: Occurrences per million content tokens.
    :param written_pm: Occurrences per million of the ``wordfreq`` list.
    :param zipf: ``wordfreq`` Zipf value.
    """

    word: str
    tatoeba_count: int
    tatoeba_pm: float
    written_pm: float
    zipf: float


@dataclass(frozen=True)
class TargetWord:
    """One target word of the cull.

    :param lemma: Lemma text.
    :param rank: Rank in the blended frequency list.
    :param deck_count: Occurrence count in the input deck.
    :param content_share: Share of the occurrences with a content part of speech.
    :param weight: Selection weight, ``1 / log2(rank + 1)``.
    """

    lemma: str
    rank: int
    deck_count: int
    content_share: float
    weight: float


@dataclass(frozen=True)
class SelectionResult:
    """Result of the greedy cover of spec section 8.

    :param kept: Input indices of the kept cards, in gain order.
    :param uses: Count of kept cards that hold each content lemma.
    :param gain: Gain of each kept card at the moment of its selection.
    :param exempt: Lemmas that are exempt from the card cap.
    :param reasons: Reason per rejected card, ``similar`` or ``max_cards``.
    :param card_targets: Target lemmas per input index, in rank order.
    :param card_hardest: Hardest frequency rank per input index.
    :param reserved: Kept reserve cards of the Swadesh set.
    :param reserved_sets: Content lemmas of each kept reserve card.
    """

    kept: list[int]
    uses: Counter[str]
    gain: dict[int, float]
    exempt: set[str]
    reasons: dict[int, str]
    card_targets: dict[int, tuple[str, ...]] = field(default_factory=dict)
    card_hardest: dict[int, int] = field(default_factory=dict)
    reserved: list[ReserveCard] = field(default_factory=list)
    reserved_sets: list[set[str]] = field(default_factory=list)


def load_input_deck(path: Path) -> list[InputCard]:
    """Read the cards of the French sentence model from an APKG file.

    :param path: Input APKG file.
    :raises RuntimeError: The package has no ``French Sentences`` model.
    :returns: One card per note, in the input order. Notes of another model
        are skipped and the extra templates of a note are dropped.
    """
    with AnkiPackage(path) as package:
        return _deck_cards(package)


def _deck_cards(package: AnkiPackage) -> list[InputCard]:
    """Read the cards of the French sentence model from an open package.

    :param package: Open package.
    :raises RuntimeError: The package has no ``French Sentences`` model.
    :returns: One card per note, in the input order.
    """
    chosen: dict[int, Any] = {}
    models = package.get_models()
    model_id = _find_model_id(models)
    fields = [field["name"] for field in models[model_id]["flds"]]
    index = {name: position for position, name in enumerate(fields)}
    french = index[FRENCH_FIELD]
    sound = index[AUDIO_FIELD]
    english = index[ENGLISH_FIELD]

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
                sentence=_plain_sentence(values[french].strip()),
                english=values[english].strip(),
                audio=_audio_name(values[sound]),
                index=len(cards),
            )
        )
    return cards


def _find_model_id(models: dict[str, dict], name: str = MODEL_NAME) -> str:
    """Find the id of a note model by name.

    :param models: Models dict from :meth:`AnkiPackage.get_models`.
    :param name: Model name.
    :raises RuntimeError: No model has the name.
    :returns: Model id as a string.
    """
    for model_id, model in models.items():
        if model.get("name") == name:
            return model_id
    raise RuntimeError(f"The input APKG has no model {name!r}.")


def _audio_name(field_value: str) -> str:
    """Read the audio file name from a sound field.

    :param field_value: Raw field text, for example ``[sound:a.mp3]``.
    :returns: The file name, or an empty string when the field has no sound tag.
    """
    match = _SOUND_PATTERN.search(field_value)
    return match.group(1) if match else ""


def load_reserve(pack_path: Path = SWADESH_PACK_PATH) -> list[ReserveCard]:
    """Read one reserve card per Swadesh target from the Swadesh APKG.

    :param pack_path: Swadesh sentence pack file.
    :raises RuntimeError: The pack has no ``French Swadesh Sentence`` model.
    :returns: One card per target, in the input order of the pack. A target
        with two cards keeps the first card.
    """
    with AnkiPackage(pack_path) as package:
        return _reserve_cards(package)


def _reserve_cards(package: AnkiPackage) -> list[ReserveCard]:
    """Read the reserve cards from an open Swadesh pack.

    :param package: Open package.
    :raises RuntimeError: The package has no Swadesh sentence model.
    :returns: One card per target, in the input order of the package.
    """
    models = package.get_models()
    model_id = _find_model_id(models, SWADESH_MODEL_NAME)
    fields = [field["name"] for field in models[model_id]["flds"]]
    index = {name: position for position, name in enumerate(fields)}

    chosen: dict[str, ReserveCard] = {}
    for row in package.get_cards():
        if str(row["mid"]) != model_id:
            continue
        values = row["flds"].split("\x1f")
        if len(values) != len(fields):
            continue
        target = values[index[SWADESH_TARGET_FIELD]].strip()
        if not target or target in chosen:
            continue
        chosen[target] = ReserveCard(
            sentence=_plain_sentence(values[index[SWADESH_SENTENCE_FIELD]]),
            html=_bold_sentence(values[index[SWADESH_SENTENCE_FIELD]]),
            english=values[index[SWADESH_ENGLISH_FIELD]].strip(),
            target=target,
            gloss=values[index[SWADESH_GLOSS_FIELD]].strip(),
            audio=_audio_name(values[index[SWADESH_AUDIO_FIELD]]),
        )
    return list(chosen.values())


def _plain_sentence(field_value: str) -> str:
    """Remove the bold markup and the no-break spaces of a reserve sentence.

    :param field_value: Raw ``Sentence`` field text.
    :returns: The sentence text with normal spaces.
    """
    return _BOLD_PATTERN.sub("", field_value).translate(_NO_BREAK_SPACES).strip()



def _bold_sentence(field_value: str) -> str:
    """Keep the bold markup of a reserve sentence and drop the no-break spaces.

    :param field_value: Raw ``Sentence`` field text.
    :returns: The sentence text with normal spaces and its ``<b>`` tags.
    """
    return field_value.translate(_NO_BREAK_SPACES).strip()

def load_frequency(path: Path = FREQUENCY_PATH) -> list[FrequencyRow]:
    """Read the cached French frequency list.

    :param path: TSV file with the columns ``word``, ``tatoeba_count``,
        ``tatoeba_pm``, ``written_pm``, and ``zipf``.
    :returns: Rows in file order.
    """
    rows: list[FrequencyRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                FrequencyRow(
                    word=record["word"],
                    tatoeba_count=int(record["tatoeba_count"]),
                    tatoeba_pm=float(record["tatoeba_pm"]),
                    written_pm=float(record["written_pm"]),
                    zipf=float(record["zipf"]),
                )
            )
    return rows


def blend_pm(row: FrequencyRow, spoken_weight: float = 0.5) -> float:
    """Blend the spoken term and the written term of one row.

    A count below :data:`SMALL_COUNT` uses the written term alone.

    :param row: Frequency row.
    :param spoken_weight: Weight of the Tatoeba term, 0 to 1.
    :returns: Blended occurrences per million.
    """
    if row.tatoeba_count < SMALL_COUNT:
        return row.written_pm
    return spoken_weight * row.tatoeba_pm + (1 - spoken_weight) * row.written_pm


def frequency_ranks(
    rows: Sequence[FrequencyRow], spoken_weight: float = 0.5
) -> dict[str, int]:
    """Rank the lemmas of a frequency list.

    :param rows: Frequency rows.
    :param spoken_weight: Weight of the Tatoeba term, 0 to 1.
    :returns: Lemma to 1-based rank. The most frequent lemma has rank 1.
    """
    scored = [(blend_pm(row, spoken_weight), row.word) for row in rows]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {word: rank for rank, (_score, word) in enumerate(scored, start=1)}


@lru_cache(maxsize=1)
def _load_nlp() -> Any:
    """Load the French spaCy pipeline without the parser and the entity model.

    :returns: The loaded pipeline.
    """
    import spacy

    return spacy.load(TOKENIZER, exclude=["ner", "parser"])


def tokenize_deck(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize the sentence of every card.

    :param cards: Input cards.
    :returns: One list of ``(lemma, part of speech)`` pairs per card. The lemma
        is lowercased and only alphabetic tokens are kept.
    """
    return _tokenize_texts(card.sentence for card in cards)


def _tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize a set of French texts with the shared spaCy pipeline.

    :param texts: Sentence texts.
    :returns: One list of lowercased ``(lemma, part of speech)`` pairs per text.
        Only alphabetic tokens are kept.
    """
    nlp = _load_nlp()
    return [
        [(token.lemma_.lower(), token.pos_) for token in doc if token.is_alpha]
        for doc in nlp.pipe(texts, batch_size=256)
    ]


def tokenize_texts(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
    """Tokenize French texts with the shared spaCy pipeline.

    :param texts: Sentence texts.
    :returns: One list of lowercased ``(lemma, part of speech)`` pairs per text.
    """
    return _tokenize_texts(texts)


def read_names(path: Path = NAMES_PATH) -> set[str]:
    """Read the name stoplist.

    :param path: Text file with one name per line.
    :returns: Lowercased names. Blank lines and ``#`` comments are skipped.
    """
    names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            name = line.strip().lower()
            if name and not name.startswith("#"):
                names.add(name)
    return names


def build_targets(
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    ranks: dict[str, int],
    names: Collection[str],
    target_words: int = 300,
) -> tuple[list[TargetWord], int]:
    """Build the target word list.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param ranks: Lemma to blended frequency rank.
    :param names: Name stoplist.
    :param target_words: Number of targets to keep.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The targets in rank order, and the count of content lemmas before
        the target cut.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")

    totals: Counter[str] = Counter()
    content: Counter[str] = Counter()
    for card_tokens in tokens:
        for lemma, pos in card_tokens:
            totals[lemma] += 1
            if pos in CONTENT_POS:
                content[lemma] += 1

    candidates: list[TargetWord] = []
    for lemma, count in totals.items():
        if content[lemma] < MIN_DECK_COUNT:
            continue
        share = content[lemma] / count
        if share < CONTENT_SHARE:
            continue
        if len(lemma) < MIN_LEMMA_LENGTH or lemma in names:
            continue
        rank = ranks.get(lemma, UNRANKED_RANK)
        candidates.append(
            TargetWord(
                lemma=lemma,
                rank=rank,
                deck_count=content[lemma],
                content_share=share,
                weight=1 / math.log2(rank + 1),
            )
        )

    candidates.sort(key=lambda target: (target.rank, target.lemma))
    return candidates[:target_words], len(candidates)


def filter_cards(
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    min_words: int = 3,
    max_words: int = 12,
) -> tuple[list[int], list[str]]:
    """Apply the filter rules F1 to F9 of spec section 7.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The input indices of the passed cards, and the reason per input
        index. A passed card has the empty string as its reason.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")

    reasons = [""] * len(cards)
    passed: list[int] = []
    seen: set[str] = set()
    for index, card in enumerate(cards):
        reason = _filter_reason(card, tokens[index], min_words, max_words)
        if reason:
            reasons[index] = reason
            continue
        if card.sentence in seen:
            reasons[index] = "F9"
            continue
        seen.add(card.sentence)
        passed.append(index)
    return passed, reasons


def _filter_reason(
    card: InputCard,
    card_tokens: Sequence[tuple[str, str]],
    min_words: int,
    max_words: int,
) -> str:
    """Give the first failed filter rule of one card.

    :param card: Input card.
    :param card_tokens: Lemma and part of speech of the card sentence.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :returns: The rule name ``F1`` to ``F8``, or the empty string on a pass.
    """
    if not MIN_ENGLISH_WORDS <= len(card.english.split()) <= MAX_ENGLISH_WORDS:
        return "F1"
    if not min_words <= len(card_tokens) <= max_words:
        return "F2"

    sentence = card.sentence
    lower = sentence.lower()
    if (
        _DIGIT_PATTERN.search(sentence)
        or any(mark in sentence for mark in BRACKET_MARKS)
        or any(mark in lower for mark in URL_MARKS)
        or any(mark in sentence for mark in ELLIPSIS_MARKS)
    ):
        return "F3"
    if not sentence.endswith(tuple(FINAL_MARKS)):
        return "F4"
    if sum(1 for char in sentence if char in FINAL_MARKS) != 1:
        return "F5"
    if set(sentence) & QUOTE_MARKS or sentence[:1] in DIALOGUE_DASHES:
        return "F6"
    if sum(1 for char in sentence if char in INTERNAL_MARKS) > 1:
        return "F7"
    if any(pos == "PROPN" for _lemma, pos in card_tokens):
        return "F8"
    return ""


def lemma_sets(
    tokens: Sequence[Sequence[tuple[str, str]]],
) -> list[set[str]]:
    """Collect the content lemmas of every card.

    :param tokens: Lemma and part of speech per card.
    :returns: One set of content lemmas per card.
    """
    return [
        {lemma for lemma, pos in card_tokens if pos in CONTENT_POS}
        for card_tokens in tokens
    ]


def reserve_lemma_sets(
    reserve: Sequence[ReserveCard],
    nlp_tokens: Sequence[Sequence[tuple[str, str]]] | None = None,
) -> list[set[str]]:
    """Collect the content lemmas of every reserve card.

    :param reserve: Reserve cards.
    :param nlp_tokens: Tokenized reserve sentences. The function tokenizes the
        reserve sentences when this is missing.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: One set of content lemmas per reserve card.
    """
    if nlp_tokens is None:
        nlp_tokens = _tokenize_texts(card.sentence for card in reserve)
    if len(nlp_tokens) != len(reserve):
        raise ValueError("The reserve list and the token list differ in size.")
    return lemma_sets(nlp_tokens)


def dedupe_reserve(
    reserve: Sequence[ReserveCard],
    reserve_sets: Sequence[set[str]],
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    similarity: float = 0.67,
) -> list[ReserveCard]:
    """Drop the reserve cards that the input deck already holds.

    A reserve card is a duplicate when its sentence is in the input deck, or
    when its lemma Jaccard against an input card reaches the limit.

    :param reserve: Reserve cards.
    :param reserve_sets: Content lemmas per reserve card.
    :param cards: Input cards.
    :param tokens: Lemma and part of speech per input card.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The reserve cards that the input deck does not hold, in order.
    """
    kept, _sets = dedupe_reserve_sets(reserve, reserve_sets, cards, tokens, similarity)
    return kept


def dedupe_reserve_sets(
    reserve: Sequence[ReserveCard],
    reserve_sets: Sequence[set[str]],
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    similarity: float = 0.67,
) -> tuple[list[ReserveCard], list[set[str]]]:
    """Drop the duplicate reserve cards and return the lemma sets of the rest.

    The input cards get a lemma inverted index, so the test visits only the
    input cards that share a lemma with the reserve card.

    :param reserve: Reserve cards.
    :param reserve_sets: Content lemmas per reserve card.
    :param cards: Input cards.
    :param tokens: Lemma and part of speech per input card.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The kept reserve cards and their content lemma sets, in order.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    if len(reserve) != len(reserve_sets):
        raise ValueError("The reserve list and the lemma set list differ in size.")

    sentences = {card.sentence for card in cards}
    sets = lemma_sets(tokens)
    index: dict[str, list[int]] = {}
    for position, lemma_set in enumerate(sets):
        for lemma in lemma_set:
            index.setdefault(lemma, []).append(position)

    kept_cards: list[ReserveCard] = []
    kept_sets: list[set[str]] = []
    kept_index: dict[str, list[int]] = {}
    for card, lemma_set in zip(reserve, reserve_sets):
        if card.sentence in sentences:
            continue
        if _is_similar(lemma_set, sets, index, similarity):
            continue
        if _is_similar(lemma_set, kept_sets, kept_index, similarity):
            continue
        kept_cards.append(card)
        kept_sets.append(lemma_set)
        for lemma in lemma_set:
            kept_index.setdefault(lemma, []).append(len(kept_sets) - 1)
    return kept_cards, kept_sets


def swadesh_coverage(
    concepts: Sequence[str],
    sentences: Sequence[str],
    tokens: Sequence[Sequence[tuple[str, str]]],
) -> tuple[int, int]:
    """Count the Swadesh concepts that the merged deck covers.

    A concept is covered when its form, or the lemma of its head word, appears
    in at least one kept card. The card does not need to be assigned to that
    concept.

    :param concepts: Swadesh target words, one per concept of the pack.
    :param sentences: French text of every kept card.
    :param tokens: Lemma and part of speech of every kept card.
    :raises ValueError: A sentence list and its token list differ in size.
    :returns: The covered concept count and the Swadesh total of 207.
    """
    if len(sentences) != len(tokens):
        raise ValueError("The sentence list and the token list differ in size.")
    lemmas = {lemma for card_tokens in tokens for lemma, _pos in card_tokens}
    forms = {
        form
        for sentence in sentences
        for form in (match.casefold() for match in _WORD_PATTERN.findall(sentence))
    }
    merged = lemmas | forms
    nlp = _load_nlp()
    covered = 0
    for doc in nlp.pipe(concepts, batch_size=64):
        concept_forms = {token.lemma_.lower() for token in doc if token.is_alpha}
        concept_forms |= {token.text.lower() for token in doc if token.is_alpha}
        if concept_forms & merged:
            covered += 1
    return covered, SWADESH_TOTAL


def select_cards(
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    targets: Sequence[TargetWord],
    ranks: dict[str, int],
    *,
    max_cards: int = 1000,
    max_uses: int = 4,
    exempt_top: int = 100,
    similarity: float = 0.67,
    passed: Sequence[int] | None = None,
    reserved: Sequence[ReserveCard] = (),
    reserved_sets: Sequence[set[str]] = (),
) -> SelectionResult:
    """Select the kept cards with the greedy cover of spec section 8.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param targets: Target words in rank order.
    :param ranks: Lemma to blended frequency rank.
    :param max_cards: Card cap.
    :param max_uses: Most kept cards that hold one non-exempt lemma.
    :param exempt_top: Number of top targets that are exempt from the cap.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :param passed: Input indices that passed the filter. Every card applies
        when empty.
    :param reserved: Reserve cards that the selection keeps.
    :param reserved_sets: Content lemmas per reserve card.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The kept cards in gain order, the lemma uses, and the reasons.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    if len(reserved) != len(reserved_sets):
        raise ValueError("The reserve list and the lemma set list differ in size.")

    sets = lemma_sets(tokens)
    exempt = {target.lemma for target in targets[:exempt_top]}
    weights = {target.lemma: target.weight for target in targets}

    def weight(lemma: str) -> float:
        value = weights.get(lemma)
        if value is None:
            value = 1 / math.log2(ranks.get(lemma, UNRANKED_RANK) + 1)
            weights[lemma] = value
        return value

    uses: Counter[str] = Counter()
    for lemma_set in reserved_sets:
        for lemma in lemma_set:
            uses[lemma] += 1
    frequency_cards = max_cards - len(reserved)

    def gain(index: int) -> float:
        total = 0.0
        for lemma in sets[index]:
            if lemma in exempt:
                total += EXEMPT_GAIN * weight(lemma)
            elif uses[lemma] < max_uses:
                total += weight(lemma) / (uses[lemma] + 1)
        return total

    def capped(index: int) -> bool:
        return any(
            lemma not in exempt and uses[lemma] >= max_uses for lemma in sets[index]
        )

    indices = list(range(len(cards))) if passed is None else list(passed)
    heap: list[tuple[float, int, int]] = []
    for index in indices:
        value = gain(index)
        if value > 0:
            heapq.heappush(heap, (-value, len(tokens[index]), index))

    kept: list[int] = []
    picked: dict[int, float] = {}
    reasons: dict[int, str] = {}
    shared: dict[str, list[int]] = {}

    while heap and len(kept) < frequency_cards:
        stored, _words, index = heapq.heappop(heap)
        current = gain(index)
        if capped(index):
            reasons[index] = "use_cap"
            continue
        if current <= 0:
            continue
        if abs(current + stored) > 1e-12:
            heapq.heappush(heap, (-current, len(tokens[index]), index))
            continue
        if _is_similar(sets[index], sets, shared, similarity):
            reasons[index] = "similar"
            continue
        picked[index] = current
        kept.append(index)
        for lemma in sets[index]:
            uses[lemma] += 1
            shared.setdefault(lemma, []).append(index)

    for index in indices:
        if index not in picked:
            reasons.setdefault(index, "max_cards")

    target_ranks = {target.lemma: target.rank for target in targets}
    card_targets: dict[int, tuple[str, ...]] = {}
    card_hardest: dict[int, int] = {}
    for index, card_tokens in enumerate(tokens):
        hits = {
            lemma
            for lemma, pos in card_tokens
            if pos in CONTENT_POS and lemma in target_ranks
        }
        if hits:
            card_targets[index] = tuple(
                sorted(hits, key=lambda lemma: (target_ranks[lemma], lemma))
            )
        card_hardest[index] = hardest_rank(card_tokens, ranks)

    return SelectionResult(
        kept=kept,
        uses=uses,
        gain=picked,
        exempt=exempt,
        reasons=reasons,
        card_targets=card_targets,
        card_hardest=card_hardest,
        reserved=list(reserved),
        reserved_sets=list(reserved_sets),
    )


def _is_similar(
    lemma_set: set[str],
    sets: Sequence[set[str]],
    shared: dict[str, list[int]],
    similarity: float,
) -> bool:
    """Test a card against the kept cards that share a lemma with it.

    :param lemma_set: Content lemmas of the new card.
    :param sets: Content lemmas of every card, by input index.
    :param shared: Lemma to input indices of the kept cards that hold it.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :returns: True when a kept card reaches the limit.
    """
    others: set[int] = set()
    for lemma in lemma_set:
        kept_indices = shared.get(lemma)
        if kept_indices:
            others.update(kept_indices)
    size = len(lemma_set)
    for other in others:
        other_set = sets[other]
        common = len(lemma_set & other_set)
        if common and common / (size + len(other_set) - common) >= similarity:
            return True
    return False


def order_cards(
    kept: Sequence[int],
    tokens: Sequence[Sequence[tuple[str, str]]],
    ranks: dict[str, int],
    order: str = "coverage",
) -> list[int]:
    """Order the kept cards for learning.

    :param kept: Input indices of the kept cards.
    :param tokens: Lemma and part of speech per card.
    :param ranks: Lemma to blended frequency rank.
    :param order: ``coverage`` sorts on ``hardest_rank``; ``input`` keeps the
        given order.
    :raises ValueError: The order is not ``coverage`` or ``input``.
    :returns: The ordered input indices.
    """
    if order == "input":
        return list(kept)
    if order != "coverage":
        raise ValueError(f"Unknown card order {order!r}.")

    return sorted(
        kept,
        key=lambda index: (
            hardest_rank(tokens[index], ranks),
            len(tokens[index]),
            index,
        ),
    )


def hardest_rank(card_tokens: Sequence[tuple[str, str]], ranks: dict[str, int]) -> int:
    """Give the blended frequency rank of the hardest content lemma of a card.

    :param card_tokens: Lemma and part of speech of the card sentence.
    :param ranks: Lemma to blended frequency rank.
    :returns: The lowest rank, or :data:`UNRANKED_ORDER` without a content lemma.
    """
    return min(
        (
            ranks.get(lemma, UNRANKED_ORDER)
            for lemma, pos in card_tokens
            if pos in CONTENT_POS
        ),
        default=UNRANKED_ORDER,
    )


def spoken_mass(rows: Sequence[FrequencyRow], lemmas: Collection[str]) -> float:
    """Give the share of the Tatoeba content tokens that a lemma set covers.

    :param rows: Frequency rows with the Tatoeba counts.
    :param lemmas: Lemmas of the kept cards.
    :returns: A share from 0 to 1.
    """
    total = sum(row.tatoeba_count for row in rows)
    if not total:
        return 0.0
    known = set(lemmas)
    covered = sum(row.tatoeba_count for row in rows if row.word in known)
    return covered / total


def default_output(apkg_path: Path) -> Path:
    """Build the default output path of a cull run from the input name.

    :param apkg_path: Input APKG file.
    :returns: Path next to the input with the suffix ``_culled.apkg``.
    """
    path = Path(apkg_path)
    return path.with_name(f"{path.stem}_culled.apkg")


def side_paths(output_path: Path) -> tuple[Path, Path, Path]:
    """Build the three side file paths of an output APKG.

    :param output_path: Output APKG file.
    :returns: The cull CSV, the coverage CSV, and the manifest paths.
    """
    path = Path(output_path)
    stem = path.with_suffix("") if path.suffix else path
    return (
        Path(f"{stem}.cull.csv"),
        Path(f"{stem}.coverage.csv"),
        Path(f"{stem}.manifest.json"),
    )


def write_cull_csv(
    cards: Sequence[InputCard],
    reasons: Sequence[str],
    selection: SelectionResult,
    ordered: Sequence[int],
    path: Path,
    reserve: Sequence[ReserveCard] = (),
) -> None:
    """Write one row per card: the reserve cards, the kept cards, then the rest.

    :param cards: Input cards.
    :param reasons: Filter reason per input index, empty for a passed card.
    :param selection: Result of the selection.
    :param ordered: Input indices of the kept cards in the final order.
    :param path: Output CSV file.
    :param reserve: Kept reserve cards, written first.
    """
    kept = set(ordered)
    rows = [_reserve_row(card) for card in reserve]
    rows += [_cull_row(cards[index], True, "", selection, index) for index in ordered]
    for index, card in enumerate(cards):
        if index in kept:
            continue
        reason = reasons[index] or selection.reasons.get(index, "")
        rows.append(_cull_row(card, False, reason, selection, index))
    _write_rows(path, CULL_COLUMNS, rows)


def _reserve_row(card: ReserveCard) -> dict[str, Any]:
    """Build one row of the cull CSV for a reserve card.

    :param card: Reserve card.
    :returns: One row mapping with the Swadesh source.
    """
    return {
        "source": SWADESH_SOURCE,
        "sentence": card.sentence,
        "english": card.english,
        "audio": card.audio,
        "kept": "1",
        "reason": "",
        "targets": card.target,
        "gain": "",
        "hardest_rank": "",
    }


def _cull_row(
    card: InputCard,
    kept: bool,
    reason: str,
    selection: SelectionResult,
    index: int,
) -> dict[str, Any]:
    """Build one row of the cull CSV.

    :param card: Input card.
    :param kept: True for a kept card.
    :param reason: Reason of a rejected card.
    :param selection: Result of the selection.
    :param index: Input index of the card.
    :returns: One row mapping.
    """
    return {
        "source": FREQUENCY_SOURCE,
        "sentence": card.sentence,
        "english": card.english,
        "audio": card.audio,
        "kept": "1" if kept else "0",
        "reason": reason,
        "targets": " ".join(selection.card_targets.get(index, ())),
        "gain": f"{selection.gain[index]:.6f}" if kept else "",
        "hardest_rank": str(selection.card_hardest.get(index, UNRANKED_ORDER)),
    }


def write_coverage_csv(
    targets: Sequence[TargetWord], selection: SelectionResult, path: Path
) -> None:
    """Write one row per target word: word, rank, count, status.

    :param targets: Target words in rank order.
    :param selection: Result of the selection.
    :param path: Output CSV file.
    """
    rows = []
    for target in targets:
        count = selection.uses.get(target.lemma, 0)
        rows.append(
            {
                "word": target.lemma,
                "rank": target.rank,
                "count": count,
                "status": coverage_status(count),
            }
        )
    _write_rows(path, COVERAGE_COLUMNS, rows)


def coverage_status(count: int) -> str:
    """Give the coverage status of a target count.

    :param count: Count of kept cards that hold the target word.
    :returns: ``covered`` for two or more cards, ``partial`` for one, ``absent`` for zero.
    """
    if count >= 2:
        return COVERED_STATUS
    if count == 1:
        return PARTIAL_STATUS
    return ABSENT_STATUS


def kept_order_hash(
    cards: Sequence[InputCard],
    ordered: Sequence[int],
    reserve: Sequence[ReserveCard] = (),
) -> str:
    """Hash the kept sentences in their final order.

    :param cards: Input cards.
    :param ordered: Input indices of the kept cards in the final order.
    :param reserve: Kept reserve cards, which come before the input cards.
    :returns: ``sha256:`` followed by the digest of the joined sentences.
    """
    sentences = [card.sentence for card in reserve]
    sentences += [cards[index].sentence for index in ordered]
    text = "\n".join(sentences)
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def write_manifest(
    path: Path,
    *,
    source: dict[str, Any],
    params: dict[str, Any],
    counts: dict[str, int],
    frequency: dict[str, Any],
    files: dict[str, str],
    kept_order: str,
    reserve_pack: str = "",
    swadesh_exceptions: Sequence[str] = (),
) -> None:
    """Write the manifest JSON of a cull run.

    :param path: Output JSON file.
    :param source: Input APKG path, deck name, and model name.
    :param params: Options of the run.
    :param counts: Card, lemma, and target counts.
    :param frequency: Frequency source paths and the spoken weight.
    :param files: ``sha256:`` values of the output files.
    :param kept_order: Hash of the kept sentences in their final order.
    :param reserve_pack: Swadesh pack of the reserve cards, empty without a
        reserve.
    :param swadesh_exceptions: Swadesh concepts that no source sentence covers.
    """
    payload: dict[str, Any] = {
        "build_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "params": params,
        "counts": counts,
        "frequency": frequency,
        "files": files,
        "licenses": LICENSES,
        "kept_order": kept_order,
    }
    if reserve_pack:
        payload["reserve_pack"] = reserve_pack
        payload["swadesh_exceptions"] = list(swadesh_exceptions)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    target.write_text(text + "\n", encoding="utf-8")


def insert_reserve_cards(
    package: AnkiPackage,
    reserve: Sequence[ReserveCard],
    reserve_pack_path: Path,
    deck_id: int,
) -> None:
    """Insert the reserve notes and cards into an open package.

    The note and card rows copy a kept row of the input model, so every column
    of the package schema stays valid.

    :param package: Open output package after the non-kept cards are deleted.
    :param reserve: Reserve cards to insert.
    :param reserve_pack_path: Swadesh APKG with the reserve media files.
    :param deck_id: Target deck id.
    :raises RuntimeError: The package has no template row, or a reserve audio
        file is not in the Swadesh pack.
    """
    conn = _connection(package)
    note_template = conn.execute("SELECT * FROM notes LIMIT 1").fetchone()
    card_template = conn.execute("SELECT * FROM cards LIMIT 1").fetchone()
    if note_template is None or card_template is None:
        raise RuntimeError("The input package has no note to use as a template.")
    note_columns = [row[1] for row in conn.execute("PRAGMA table_info(notes)")]
    card_columns = [row[1] for row in conn.execute("PRAGMA table_info(cards)")]
    next_note = int(conn.execute("SELECT MAX(id) FROM notes").fetchone()[0] or 0) + 1
    next_card = int(conn.execute("SELECT MAX(id) FROM cards").fetchone()[0] or 0) + 1
    mod = int(time.time() * 1000)

    with AnkiPackage(reserve_pack_path) as swadesh:
        if swadesh.temp_dir is None:
            raise RuntimeError("The Swadesh package has no temporary directory.")
        temp_dir = Path(swadesh.temp_dir)
        media = swadesh.get_media_mapping()
        reverse = {name: key for key, name in media.items()}
        for card in reserve:
            key = reverse.get(card.audio)
            if key is None:
                raise RuntimeError(
                    f"The reserve audio {card.audio} is not in {reserve_pack_path}."
                )
            package.add_media_file(temp_dir / key, card.audio)

    note_sql = (
        f"INSERT INTO notes ({','.join(note_columns)}) "
        f"VALUES ({','.join('?' for _ in note_columns)})"
    )
    card_sql = (
        f"INSERT INTO cards ({','.join(card_columns)}) "
        f"VALUES ({','.join('?' for _ in card_columns)})"
    )
    for card in reserve:
        back = f"{card.english} — {card.target}"
        if card.gloss:
            back += f" ({card.gloss})"
        fields = "\x1f".join(
            [card.html, f"[sound:{card.audio}]", back, ""]
        )
        note = dict(note_template)
        note.update(
            id=next_note,
            guid=str(uuid.uuid4()),
            mod=mod,
            usn=-1,
            tags=RESERVE_TAGS,
            flds=fields,
            sfld=card.sentence,
            csum=0,
        )
        conn.execute(note_sql, [note[column] for column in note_columns])
        row = dict(card_template)
        row.update(
            id=next_card,
            nid=next_note,
            did=deck_id,
            ord=0,
            mod=mod,
            usn=-1,
            type=0,
            queue=0,
            due=0,
            ivl=0,
            factor=0,
            reps=0,
            lapses=0,
            left=0,
            odue=0,
            odid=0,
            flags=0,
            data="",
        )
        conn.execute(card_sql, [row[column] for column in card_columns])
        next_note += 1
        next_card += 1
    conn.commit()
    package._modified = True


def card_css(css_path: Path = CARD_CSS_PATH) -> str:
    """Give the CSS of the culled deck.

    The classes and the sizes are the same as the French Swadesh pack, so the
    two decks look alike.

    :param css_path: Base CSS file of the card classes.
    :returns: The base CSS and the French pack classes.
    """
    return load_css(css_path, EXTRA_CSS)


def apply_card_templates(
    package: AnkiPackage, model_id: str | None = None
) -> None:
    """Set the Swadesh-style templates and CSS on the deck model.

    The model keeps a template that the note does not use, so the function
    writes only the templates that the model has.

    :param package: Open package.
    :param model_id: Model id, or None to find the French sentence model.
    :raises RuntimeError: The package has no French sentence model.
    """
    models = package.get_models(include_templates=True)
    model_id = model_id or _find_model_id(models)
    package.update_model_template(model_id=model_id, css=card_css())
    count = len(models[model_id].get("tmpls", ()))
    for index, (qfmt, afmt) in enumerate(CARD_TEMPLATES[:count]):
        package.update_model_template(
            model_id=model_id, qfmt=qfmt, afmt=afmt, template_index=index
        )


def write_deck(
    input_path: Path,
    cards: Sequence[InputCard],
    kept: Sequence[int],
    output_path: Path,
    deck_name: str,
    reserve: Sequence[ReserveCard] = (),
    reserve_pack_path: Path = SWADESH_PACK_PATH,
) -> dict[str, int]:
    """Write the culled deck to an APKG file.

    The cards of every note that is not kept are deleted, and the media files
    that no kept card uses are removed with them.

    :param input_path: Input APKG file.
    :param cards: Input cards.
    :param kept: Input indices of the kept cards.
    :param output_path: Output APKG file.
    :param deck_name: Name of the output deck.
    :param reserve: Reserve cards to add to the input model.
    :param reserve_pack_path: Swadesh APKG with the reserve media files.
    :raises RuntimeError: The output note count differs from the kept count.
    :returns: Note, media, and card counts of the output.
    """
    kept_notes = {cards[index].note_id for index in kept}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with AnkiPackage(input_path) as package:
        doomed = [
            int(row["id"])
            for row in package.get_cards()
            if int(row["nid"]) not in kept_notes
        ]
        if doomed:
            package.delete_cards(doomed, cleanup_audio=True)
        rename_deck(package, deck_name)
        if reserve:
            deck_id = int(next(iter(package.get_decks())))
            insert_reserve_cards(package, reserve, reserve_pack_path, deck_id)
        apply_card_templates(package)
        package.save(target)

    expected = len(kept_notes) + len(reserve)
    with AnkiPackage(target) as package:
        notes = len(package.get_notes())
        media = len(package.get_media_mapping())
    if notes != expected:
        raise RuntimeError(f"The output APKG has {notes} notes, not {expected}.")
    return {"notes": notes, "media": media, "items": expected}


def rename_deck(package: AnkiPackage, deck_name: str) -> None:
    """Rename the deck of the kept cards and delete the decks that stay empty.

    :param package: Open package.
    :param deck_name: New deck name.
    :raises RuntimeError: The collection has no card.
    """
    cards = package.get_cards()
    counts = Counter(str(row["did"]) for row in cards)
    if not counts:
        raise RuntimeError("The collection has no card to place in a deck.")
    main = int(counts.most_common(1)[0][0])
    for row in cards:
        if int(row["did"]) != main:
            package.move_card(int(row["id"]), main)
    set_deck_name(package, main, deck_name)
    for deck_id in list(package.get_decks()):
        if int(deck_id) != main:
            delete_deck(package, int(deck_id))


def _connection(package: AnkiPackage) -> sqlite3.Connection:
    """Give the open database connection of a package.

    :param package: Open package.
    :raises RuntimeError: The package has no open connection.
    :returns: The SQLite connection.
    """
    if package.conn is None:
        raise RuntimeError("The package database is closed.")
    return package.conn


def set_deck_name(package: AnkiPackage, deck_id: int, deck_name: str) -> None:
    """Set the name of one deck in either database format.

    :param package: Open package.
    :param deck_id: Deck id.
    :param deck_name: New deck name.
    """
    if package.db_format == "anki21b":
        conn = _connection(package)
        conn.execute("UPDATE decks SET name = ? WHERE id = ?", (deck_name, deck_id))
        conn.commit()
    else:
        decks = _legacy_decks(package)
        decks[str(deck_id)]["name"] = deck_name
        _write_legacy_decks(package, decks)
    package._modified = True


def delete_deck(package: AnkiPackage, deck_id: int) -> None:
    """Delete one empty deck in either database format.

    :param package: Open package.
    :param deck_id: Deck id.
    """
    if package.db_format == "anki21b":
        conn = _connection(package)
        conn.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
        conn.commit()
    else:
        decks = _legacy_decks(package)
        if decks.pop(str(deck_id), None) is None:
            return
        _write_legacy_decks(package, decks)
    package._modified = True


def _legacy_decks(package: AnkiPackage) -> dict[str, dict]:
    """Read the deck JSON of a legacy collection.

    :raises ValueError: The deck cell is not a JSON object.
    """
    row = _connection(package).execute("SELECT decks FROM col").fetchone()
    try:
        decks = json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"bad deck JSON: {error}") from error
    if not isinstance(decks, dict):
        raise ValueError("the deck cell is not an object")
    return decks


def _write_legacy_decks(package: AnkiPackage, decks: dict[str, dict]) -> None:
    """Write the deck JSON of a legacy collection."""
    conn = _connection(package)
    conn.execute("UPDATE col SET decks = ? WHERE id = 1", (json.dumps(decks),))
    conn.commit()


def _write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    """Write CSV rows with a fixed column order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


GateRow = tuple[str, bool, str]
GateCheck = Callable[["GateInputs"], tuple[bool, str]]


class GateInputs:
    """Lazy reader for the APKG, the CSVs, and the manifest of a cull run.

    A read error is cached and re-raised for each gate that needs the file, so
    one missing input marks only the dependent gates as failed.

    :param apkg_path: Output APKG file.
    :param cull_csv: Cull CSV file.
    :param coverage_csv: Coverage CSV file.
    :param manifest_path: Manifest JSON file.
    """

    def __init__(
        self,
        apkg_path: Path,
        cull_csv: Path,
        coverage_csv: Path,
        manifest_path: Path,
    ) -> None:
        """Store the paths and start an empty cache."""
        self.apkg_path = Path(apkg_path)
        self.cull_csv = Path(cull_csv)
        self.coverage_csv = Path(coverage_csv)
        self.manifest_path = Path(manifest_path)
        self._cache: dict[str, tuple[bool, Any]] = {}

    def _load(self, name: str, loader: Callable[[], Any]) -> Any:
        """Load one value once and cache the value or the error."""
        if name not in self._cache:
            try:
                self._cache[name] = (True, loader())
            except Exception as error:  # noqa: BLE001 - cached for every gate
                self._cache[name] = (False, error)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    def rows(self) -> list[dict[str, str]]:
        """Read the cull CSV rows."""
        return self._load("rows", lambda: _read_csv(self.cull_csv))

    def coverage(self) -> list[dict[str, str]]:
        """Read the coverage CSV rows."""
        return self._load("coverage", lambda: _read_csv(self.coverage_csv))

    def manifest(self) -> dict[str, Any]:
        """Read the manifest JSON object."""
        return self._load("manifest", lambda: _read_json(self.manifest_path))

    def params(self) -> dict[str, Any]:
        """Read the run options of the manifest."""
        params = self.manifest().get("params", {})
        return dict(params)

    def output(self) -> dict[str, Any]:
        """Read the output APKG and probe the duration of its audio."""
        return self._load("output", lambda: _read_output(self.apkg_path))

    def tokens(self) -> list[list[tuple[str, str]]]:
        """Tokenize the sentences of the output deck."""
        return self._load("tokens", lambda: tokenize_deck(self.output()["cards"]))


def gate_inputs(apkg_path: Path) -> GateInputs:
    """Build the gate inputs of an output APKG.

    :param apkg_path: Output APKG file.
    :returns: A lazy reader for the APKG and its three side files.
    """
    cull_csv, coverage_csv, manifest_path = side_paths(apkg_path)
    return GateInputs(apkg_path, cull_csv, coverage_csv, manifest_path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file.

    :raises ValueError: The file is not a JSON object.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"bad JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def _read_output(path: Path) -> dict[str, Any]:
    """Read the notes, the decks, and the media of an output APKG.

    :param path: Output APKG file.
    :returns: Cards, note count, model names, deck names, media names, and the
    :returns: Cards, note count, model names, deck names, media names, the
        card templates and CSS per model, and the audio duration per file.
    """
    with AnkiPackage(path) as package:
        full = package.get_models(include_templates=True)
        models = {key: model.get("name") for key, model in full.items()}
        templates = {
            model.get("name"): [
                (tmpl.get("qfmt", ""), tmpl.get("afmt", ""))
                for tmpl in model.get("tmpls", ())
            ]
            for model in full.values()
        }
        css = {model.get("name"): model.get("css", "") for model in full.values()}
        notes = package.get_notes()
        cards = _deck_cards(package)
        media = package.get_media_mapping()
        names = {card.audio for card in cards if card.audio}
        durations = _probe_audio(package.temp_dir, media, names)
        decks = {deck.get("name") for deck in package.get_decks().values()}
    return {
        "cards": cards,
        "notes": len(notes),
        "note_models": {models.get(str(row["mid"])) for row in notes},
        "decks": decks,
        "media": set(media.values()),
        "audio": durations,
        "templates": templates,
        "css": css,
    }


def _probe_audio(
    temp_dir: str | None, media: dict[str, str], names: Collection[str]
) -> dict[str, float | None]:
    """Probe the duration of the audio files of the kept cards.

    :param temp_dir: Directory with the unpacked media files.
    :param media: Media file id to name mapping.
    :param names: File names to probe.
    :returns: File name to duration in seconds. The value is None when the
        file is missing or is not a valid MP3.
    """
    if not temp_dir:
        return {}
    paths = {
        name: Path(temp_dir) / file_id
        for file_id, name in media.items()
        if name in names
    }

    def probe(item: tuple[str, Path]) -> tuple[str, float | None]:
        name, path = item
        if not path.exists():
            return name, None
        try:
            return name, probe_duration(path)
        except RuntimeError:
            return name, None

    with ThreadPoolExecutor(max_workers=AUDIO_PROBE_WORKERS) as pool:
        return dict(pool.map(probe, sorted(paths.items())))


def run_gates(inputs: GateInputs) -> list[GateRow]:
    """Run gates G1 to G13 of the spec on the written files.

    :param inputs: Lazy reader for the output files.
    :returns: One ``(gate, passed, message)`` row per gate.
    """
    results: list[GateRow] = []
    for gate, check in GATES:
        try:
            passed, message = check(inputs)
        except Exception as error:  # noqa: BLE001 - a gate reports read failures
            passed, message = False, f"{type(error).__name__}: {error}"
        results.append((gate, passed, message))
    return results


def _sample(items: Sequence[str], limit: int = GATE_SAMPLE) -> str:
    """Join a bounded number of items for a gate message."""
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f" (+{len(items) - limit} more)"
    return shown


def _kept_rows(inputs: GateInputs) -> list[dict[str, str]]:
    """Read the kept rows of the cull CSV."""
    return [row for row in inputs.rows() if row.get("kept") == "1"]


def _exempt_lemmas(inputs: GateInputs) -> set[str]:
    """Read the target lemmas that are exempt from the card cap."""
    top = int(inputs.params().get("exempt_top", 0))
    return {row["word"] for row in inputs.coverage()[:top]}


def gate_card_count(inputs: GateInputs) -> tuple[bool, str]:
    """G1: the card count is the cap, or the pool has no more cards."""
    kept = len(_kept_rows(inputs))
    cap = int(inputs.params().get("max_cards", 0))
    pooled = int(inputs.manifest().get("counts", {}).get("passed", 0))
    if kept == cap:
        return True, f"{kept} cards, the cap"
    if kept == pooled:
        return True, f"{kept} cards, the pool has no more"
    return False, f"{kept} cards, not the cap {cap} and below the pool {pooled}"


def gate_use_cap(inputs: GateInputs) -> tuple[bool, str]:
    """G2: no non-exempt lemma is in more than the cap of kept cards."""
    limit = int(inputs.params().get("max_uses", 0))
    exempt = _exempt_lemmas(inputs)
    frequency_sentences = {
        row["sentence"]
        for row in inputs.rows()
        if row.get("source") == FREQUENCY_SOURCE and row.get("kept") == "1"
    }
    uses: Counter[str] = Counter()
    for card, card_tokens in zip(inputs.output()["cards"], inputs.tokens()):
        if card.sentence not in frequency_sentences:
            continue
        uses.update({lemma for lemma, pos in card_tokens if pos in CONTENT_POS})
    over = {
        lemma: count
        for lemma, count in uses.items()
        if lemma not in exempt and count > limit
    }
    if over:
        names = [f"{lemma} {count}" for lemma, count in sorted(over.items())]
        return False, f"{len(over)} lemma(s) over {limit}: {_sample(names)}"
    most = max(
        (count for lemma, count in uses.items() if lemma not in exempt), default=0
    )
    return True, f"max {most} uses outside the exempt {len(exempt)}"


def gate_target_coverage(inputs: GateInputs) -> tuple[bool, str]:
    """G3: target coverage reaches the gate share."""
    rows = inputs.coverage()
    covered = sum(1 for row in rows if int(row.get("count", 0)) >= 1)
    share = covered / len(rows) if rows else 0.0
    message = f"{covered}/{len(rows)} targets covered ({share:.3f}"
    message += f", gate {TARGET_COVERAGE_GATE})"
    return bool(rows) and share >= TARGET_COVERAGE_GATE, message


def gate_double_coverage(inputs: GateInputs) -> tuple[bool, str]:
    """G4: the share of targets with two cards reaches the gate share."""
    rows = inputs.coverage()
    twice = sum(1 for row in rows if int(row.get("count", 0)) >= 2)
    share = twice / len(rows) if rows else 0.0
    message = f"{twice}/{len(rows)} targets with 2 cards ({share:.3f}"
    message += f", gate {DOUBLE_COVERAGE_GATE})"
    return bool(rows) and share >= DOUBLE_COVERAGE_GATE, message


def gate_filter(inputs: GateInputs) -> tuple[bool, str]:
    """G5: every kept card passes the filter rules."""
    min_words = int(inputs.params().get("min_words", 0))
    max_words = int(inputs.params().get("max_words", 0))
    rows = _kept_rows(inputs)
    tokens = tokenize_texts([row["sentence"] for row in rows])
    problems: list[str] = []
    seen: set[str] = set()
    for row, card_tokens in zip(rows, tokens, strict=True):
        card = InputCard(
            note_id=0,
            sentence=row["sentence"],
            english=row["english"],
            audio="",
            index=0,
        )
        reason = _filter_reason(card, card_tokens, min_words, max_words)
        if reason == "F8" and row.get("source") == SWADESH_SOURCE:
            reason = ""
        if not reason and card.sentence in seen:
            reason = "F9"
        seen.add(card.sentence)
        if reason:
            problems.append(f"{reason} {card.sentence[:40]}")
    if problems:
        return False, f"{len(problems)} kept card(s) fail: {_sample(problems)}"
    return True, f"{len(rows)} kept cards pass F1 to F9"


def gate_similarity(inputs: GateInputs) -> tuple[bool, str]:
    """G6: no kept card pair reaches the lemma Jaccard limit."""
    limit = float(inputs.params().get("similarity", 0.67))
    cards = inputs.output()["cards"]
    sets = lemma_sets(inputs.tokens())
    shared: dict[str, list[int]] = {}
    problems: list[str] = []
    for index, lemma_set in enumerate(sets):
        if _is_similar(lemma_set, sets, shared, limit):
            problems.append(cards[index].sentence[:40])
        for lemma in lemma_set:
            shared.setdefault(lemma, []).append(index)
    if problems:
        return False, f"{len(problems)} duplicate card(s): {_sample(problems)}"
    return True, f"{len(sets)} kept cards below the Jaccard limit {limit}"


def gate_audio(inputs: GateInputs) -> tuple[bool, str]:
    """G7: every kept card has a valid MP3 file."""
    output = inputs.output()
    media = output["media"]
    durations = output["audio"]
    problems: list[str] = []
    for card in output["cards"]:
        if not card.audio:
            problems.append(f"no sound: {card.sentence[:30]}")
        elif card.audio not in media:
            problems.append(f"missing media: {card.audio}")
        elif not durations.get(card.audio):
            problems.append(f"bad MP3: {card.audio}")
    if problems:
        return False, f"{len(problems)} card(s): {_sample(problems)}"
    return True, f"{len(output['cards'])} cards have a valid MP3 file"


def gate_note_count(inputs: GateInputs) -> tuple[bool, str]:
    """G8: the note count matches the CSV and the manifest."""
    notes = int(inputs.output()["notes"])
    kept = len(_kept_rows(inputs))
    cards = int(inputs.manifest().get("counts", {}).get("cards", 0))
    message = f"{notes} notes, {kept} kept rows, manifest {cards}"
    return notes == kept == cards, message


def gate_media(inputs: GateInputs) -> tuple[bool, str]:
    """G9: the APKG holds exactly the media files of the kept cards."""
    output = inputs.output()
    media = output["media"]
    used = {card.audio for card in output["cards"] if card.audio}
    extra = sorted(media - used)
    missing = sorted(used - media)
    if extra or missing:
        return (
            False,
            f"{len(extra)} extra, {len(missing)} missing: {_sample(extra + missing)}",
        )
    return True, f"{len(media)} media files, all used by the kept cards"


def gate_identity(inputs: GateInputs) -> tuple[bool, str]:
    """G10: the deck name and the note model are the requested values."""
    output = inputs.output()
    name = str(inputs.params().get("name"))
    problems: list[str] = []
    if output["decks"] != {name}:
        problems.append(f"decks {sorted(map(str, output['decks']))}")
    if output["note_models"] != {MODEL_NAME}:
        problems.append(f"models {sorted(map(str, output['note_models']))}")
    if problems:
        return False, f"expected {name!r} and {MODEL_NAME!r}: {_sample(problems)}"
    return True, f"deck {name!r} and model {MODEL_NAME!r}"


def gate_kept_order(inputs: GateInputs) -> tuple[bool, str]:
    """G11: the kept set and order match the manifest of the run."""
    rows = inputs.rows()
    kept = _kept_rows(inputs)
    if rows[: len(kept)] != kept:
        return False, "the kept rows are not the first rows of the cull CSV"
    text = "\n".join(row["sentence"] for row in kept)
    digest = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    expected = inputs.manifest().get("kept_order")
    if digest != expected:
        return False, f"kept order {digest} is not the manifest {expected}"
    return True, f"{len(kept)} kept cards in the manifest order"


def gate_coverage_rows(inputs: GateInputs) -> tuple[bool, str]:
    """G12: the coverage CSV has one row per target word."""
    rows = inputs.coverage()
    wanted = int(inputs.params().get("target_words", 0))
    if len(rows) != wanted:
        return False, f"{len(rows)} coverage rows, not {wanted}"
    words = [row.get("word", "") for row in rows]
    if len(set(words)) != len(words):
        return False, f"{len(words) - len(set(words))} duplicate target word(s)"
    wrong = [
        row["word"]
        for row in rows
        if row.get("status") != coverage_status(int(row.get("count", 0)))
    ]
    if wrong:
        return False, f"{len(wrong)} row(s) with a wrong status: {_sample(wrong)}"
    return True, f"{len(rows)} coverage rows, one per target word"


def gate_manifest_hashes(inputs: GateInputs) -> tuple[bool, str]:
    """G13: the manifest hashes match the two CSV files."""
    files = inputs.manifest().get("files", {})
    problems: list[str] = []
    for key, path in (
        ("cull_csv", inputs.cull_csv),
        ("coverage_csv", inputs.coverage_csv),
    ):
        actual = sha256_value(path)
        if files.get(key) != actual:
            problems.append(f"{key} manifest {files.get(key)} != {actual}")
    if problems:
        return False, f"{len(problems)} hash(es) differ: {_sample(problems)}"
    return True, "2 file hashes match"


def gate_swadesh(inputs: GateInputs) -> tuple[bool, str]:
    """G14: the merged deck covers the Swadesh concepts, minus the exceptions."""
    counts = inputs.manifest().get("counts", {})
    covered = int(counts.get("swadesh_covered", 0))
    total = int(counts.get("swadesh_total", 0))
    exceptions = inputs.manifest().get("swadesh_exceptions", [])
    needed = total - len(exceptions)
    if covered < needed:
        return False, f"Swadesh {covered}/{total}, need {needed}"
    return True, f"Swadesh {covered}/{total}, {len(exceptions)} exception(s)"


def gate_card_templates(inputs: GateInputs) -> tuple[bool, str]:
    """G15: the deck model holds the Swadesh-style templates and CSS."""
    output = inputs.output()
    templates = output["templates"].get(MODEL_NAME, [])
    if not templates:
        return False, f"the model {MODEL_NAME!r} has no card template"
    expected = CARD_TEMPLATES[: len(templates)]
    problems = [
        f"card {index + 1} is {qfmt!r} / {afmt!r}"
        for index, (qfmt, afmt) in enumerate(templates)
        if (qfmt, afmt) != expected[index]
    ]
    css = output["css"].get(MODEL_NAME, "")
    for name in ("sentence", "english", "hint"):
        if f".{name}" not in css:
            problems.append(f"the CSS has no .{name} class")
    if problems:
        return False, f"{len(problems)} template problem(s): {_sample(problems)}"
    return True, f"{len(templates)} card template(s) and the CSS classes match"


GATES: tuple[tuple[str, GateCheck], ...] = (
    ("G1", gate_card_count),
    ("G2", gate_use_cap),
    ("G3", gate_target_coverage),
    ("G4", gate_double_coverage),
    ("G5", gate_filter),
    ("G6", gate_similarity),
    ("G7", gate_audio),
    ("G8", gate_note_count),
    ("G9", gate_media),
    ("G10", gate_identity),
    ("G11", gate_kept_order),
    ("G12", gate_coverage_rows),
    ("G13", gate_manifest_hashes),
    ("G14", gate_swadesh),
    ("G15", gate_card_templates),
)


def build_frequency_cache(
    tatoeba_path: Path = TATOEBA_PATH, out_path: Path = FREQUENCY_PATH
) -> int:
    """Build the cached French frequency list from the Tatoeba corpus.

    One spaCy pass counts the lemmas of ``fra_sentences.tsv``. The written term
    comes from ``wordfreq``.

    :param tatoeba_path: Tatoeba French sentence TSV, columns ``id``, ``lang``,
        and ``text``.
    :param out_path: Output TSV file.
    :returns: Number of written rows.
    """
    from wordfreq import zipf_frequency

    counts, content_tokens = count_tatoeba_lemmas(tatoeba_path)
    rows: list[FrequencyRow] = []
    for word, count in counts.items():
        zipf = zipf_frequency(word, "fr")
        rows.append(
            FrequencyRow(
                word=word,
                tatoeba_count=count,
                tatoeba_pm=count / content_tokens * 1e6 if content_tokens else 0.0,
                written_pm=10 ** (zipf - 3),
                zipf=zipf,
            )
        )

    rows.sort(key=lambda row: row.word)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "tatoeba_count", "tatoeba_pm", "written_pm", "zipf"])
        for row in rows:
            writer.writerow(
                [
                    row.word,
                    row.tatoeba_count,
                    f"{row.tatoeba_pm:.6f}",
                    f"{row.written_pm:.6f}",
                    f"{row.zipf:.3f}",
                ]
            )
    return len(rows)


def count_tatoeba_lemmas(path: Path) -> tuple[Counter[str], int]:
    """Count the lemmas of a Tatoeba sentence file.

    :param path: Tatoeba French sentence TSV.
    :returns: Lemma counts, and the count of tokens with a content part of speech.
    """
    nlp = _load_nlp()
    counts: Counter[str] = Counter()
    content_tokens = 0
    for doc in nlp.pipe(_tatoeba_texts(path), batch_size=64):
        for token in doc:
            if not token.is_alpha:
                continue
            counts[token.lemma_.lower()] += 1
            if token.pos_ in CONTENT_POS:
                content_tokens += 1
    return counts, content_tokens


def _tatoeba_texts(path: Path) -> Iterator[str]:
    """Read the sentence text of a Tatoeba sentence file.

    :param path: Tatoeba French sentence TSV.
    :returns: One sentence text per row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.reader(handle, delimiter="\t"):
            if len(record) >= 3:
                yield record[2]
