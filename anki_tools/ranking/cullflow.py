"""Shared cull flow: profile, selection, reserve pack, files, and the run.

The module filters the cards of a sentence deck, selects a greedy cover of the
target words, orders the kept cards, inserts the reserve cards of a Swadesh
pack, and writes the cull CSV, the coverage CSV, the manifest, and the deck. A
language package supplies a :class:`CullProfile`, and this module reads every
language value from it.

The cull gates live in ``anki_tools/ranking/cullgates.py``. The deck writer
stays in ``anki_tools/ranking/coreflow.py``, because the core flow shares it.

See ``specs/french-1000-sentence-cull.md`` sections 4 to 9.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import coreflow
from anki_tools.ranking.cardfilter import (
    _WORD_PATTERN,
    english_rule,
    filter_cards,
    pos_rule,
    script_rule,
)
from anki_tools.ranking.coreflow import InputCard, write_deck, write_manifest
from anki_tools.ranking.corelist import UNRANKED_RANK, TargetWord, build_targets, is_content
from anki_tools.ranking.cover import greedy_cover
from anki_tools.ranking.coverage import write_rows
from anki_tools.ranking.duplicates import at_limit
from anki_tools.sentence_pack.audit import sha256_value
from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.tokens import load_model

DEFAULT_MAX_CARDS = 1000
DEFAULT_TARGET_WORDS = 300
DEFAULT_MAX_USES = 4

SMALL_COUNT = 5
UNRANKED_ORDER = 1_000_000
TATOEBA_BATCH = 1000
TOKENIZE_BATCH = 256
FILTER_RULES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")
EXEMPT_GAIN = 0.1

# The shape guard names the mark rules F5 to F8. The cull keeps the names of the
# shared card filter, where F4 to F7 are the mark rules.
MARK_RULES = {"F5": "F4", "F6": "F5", "F7": "F6", "F8": "F7"}
LENGTH_RULES = ("F2",)

RESERVE_SENTENCE_FIELD = "Sentence"
RESERVE_ENGLISH_FIELD = "English"
RESERVE_TARGET_FIELD = "Target"
RESERVE_GLOSS_FIELD = "Gloss"
RESERVE_AUDIO_FIELD = "Audio"

SWADESH_SOURCE = "swadesh"
FREQUENCY_SOURCE = "frequency"
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
COVERAGE_COLUMNS = ("word", "rank", "count", "status")
COVERED_STATUS = "covered"
PARTIAL_STATUS = "partial"
ABSENT_STATUS = "absent"

_BOLD_PATTERN = re.compile(r"</?b>")
_NO_BREAK_SPACES = {0x00A0: " ", 0x202F: " "}


@dataclass(frozen=True)
class ReserveCard:
    """One reserved Swadesh card of the cull.

    :param sentence: Reserve sentence text without the bold markup.
    :param html: Reserve sentence with the bold target markup.
    :param english: English translation.
    :param target: Swadesh target word of the card.
    :param gloss: English gloss of the target word.
    :param audio: Name of the MP3 file in the reserve pack.
    """

    sentence: str
    html: str
    english: str
    target: str
    gloss: str
    audio: str


@dataclass(frozen=True)
class FrequencyRow:
    """One row of the cached frequency list.

    :param word: Lemma.
    :param tatoeba_count: Occurrence count in the Tatoeba corpus.
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
class SelectionResult:
    """Result of the greedy cover of spec section 8.

    :param kept: Input indices of the kept cards, in gain order.
    :param uses: Count of kept cards that hold each content lemma.
    :param gain: Gain of each kept card at the moment of its selection.
    :param exempt: Lemmas that are exempt from the card cap.
    :param reasons: Reason per rejected card, ``similar`` or ``max_cards``.
    :param card_targets: Target lemmas per input index, in rank order.
    :param card_hardest: Hardest frequency rank per input index.
    :param reserved: Kept reserve cards of the reserve pack.
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


@dataclass(frozen=True)
class CullProfile:
    """Language hooks of the shared cull flow.

    :param tokenize_texts: Tokenize sentence texts into ``(lemma, part of
        speech)`` pairs.
    :param load_frequency: Read the frequency rows of the language.
    :param frequency_ranks: Rank the lemmas by the blended frequency.
    :param read_names: Read the name stoplist.
    :param concept_forms: Give the forms of one reserve concept, for example the
        lemma and the text forms of a Latin concept or the characters of a Han
        concept.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :param min_lemma_length: Shortest accepted lemma.
    :param final_marks: Accepted sentence-final marks.
    :param quote_marks: Quotation marks and dialogue dashes of the mark rule.
    :param internal_marks: Internal punctuation marks of the mark rule.
    :param model_name: Name of the input note model.
    :param sentence_field: Name of the sentence field of the model.
    :param english_field: Name of the English field of the model.
    :param audio_field: Name of the audio field of the model.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :param reserve_fields: Give the field values of one reserve card.
    :param templates: Card templates of the cull deck.
    :param card_css: CSS of the cull deck.
    :param reserve_pack: Reserve pack of the language, or None without one.
    :param reserve_model_name: Name of the note model of the reserve pack.
    :param swadesh_total: Concept count of the reserve pack.
    :param swadesh_exceptions: Reserve concepts that no source sentence covers.
    :param frequency_path: Frequency file of the language.
    :param names_path: Name stoplist of the language.
    :param tatoeba_path: Tatoeba sentence file of the language.
    :param wordfreq_lang: ``wordfreq`` language code.
    :param default_deck_name: Name of the output deck.
    :param licenses: License text per data source.
    :param target_coverage_gate: Gate share of the targets with one card.
    :param double_coverage_gate: Gate share of the targets with two cards.
    """

    tokenize_texts: Callable[[Iterable[str]], list[list[tuple[str, str]]]]
    load_frequency: Callable[[], Sequence[FrequencyRow]]
    frequency_ranks: Callable[[Sequence[FrequencyRow], float], dict[str, int]]
    read_names: Callable[[], set[str]]
    concept_forms: Callable[[str], set[str]]

    content_pos: frozenset[str]
    min_lemma_length: int
    final_marks: str
    quote_marks: str
    internal_marks: str

    model_name: str
    sentence_field: str
    english_field: str
    audio_field: str
    clean_sentence: Callable[[str], str]
    reserve_fields: Callable[[ReserveCard], Sequence[str]]

    templates: tuple[tuple[str, str], ...]
    card_css: Callable[[], str]

    reserve_pack: Path | None
    reserve_model_name: str
    swadesh_total: int
    swadesh_exceptions: tuple[str, ...]

    frequency_path: Path
    names_path: Path
    tatoeba_path: Path
    wordfreq_lang: str

    default_deck_name: str
    licenses: Mapping[str, str]

    target_coverage_gate: float
    double_coverage_gate: float


@dataclass(frozen=True)
class CardShape:
    """Shape-guard values of one card of the cull.

    The cull measures the length window with the tokenizer output, so the unit
    count comes from the card and not from the sentence text.

    :param units: Token count of the card sentence.
    :param final_marks: Accepted sentence-final marks of the language.
    :param quote_marks: Quotation marks and dialogue dashes of the language.
    :param internal_marks: Internal punctuation marks of the language.
    """

    units: int
    final_marks: str
    quote_marks: str
    internal_marks: str

    def unit_count(self, sentence: str) -> int:
        """Give the token count of the card.

        :param sentence: Prepared sentence text.
        :returns: The token count of the card.
        """
        return self.units


def filter_reason(
    card: InputCard,
    card_tokens: Sequence[tuple[str, str]],
    min_words: int,
    max_words: int,
    profile: CullProfile,
) -> str:
    """Give the first failed filter rule F1 to F8 of one card of the cull.

    The English window, the script rule, and the proper-noun rule come from the
    shared card filter. The length window and the mark rules come from the
    sentence shape guard, so the marks are the marks of the language profile.

    :param card: Input card.
    :param card_tokens: Lemma and part of speech of the card sentence.
    :param min_words: Minimum token count of a card.
    :param max_words: Maximum token count of a card.
    :param profile: Language hooks.
    :returns: The rule name ``F1`` to ``F8``, or the empty string on a pass.
    """
    reason = english_rule(card)
    if reason:
        return reason
    shape = CardShape(
        len(card_tokens),
        profile.final_marks,
        profile.quote_marks,
        profile.internal_marks,
    )
    reason = check_sentence_shape(
        card.sentence, shape, min_words, max_words, keys=LENGTH_RULES
    )
    if reason:
        return reason
    reason = script_rule(card.sentence)
    if reason:
        return reason
    reason = check_sentence_shape(
        card.sentence, shape, min_words, max_words, keys=MARK_RULES
    )
    if reason:
        return MARK_RULES[reason]
    return pos_rule(card_tokens)


@dataclass(frozen=True)
class CullRequest:
    """Options of one cull run.

    :param input_path: Input APKG file.
    :param output_path: Output APKG file.
    :param max_cards: Card cap.
    :param target_words: Number of target words.
    :param max_uses: Most kept cards that hold one non-exempt lemma.
    :param exempt_top: Number of top targets that are exempt from the cap.
    :param spoken_weight: Weight of the Tatoeba term, 0 to 1.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :param order: ``coverage`` or ``input`` card order.
    :param name: Output deck name, or None for the profile value.
    :param reserve_pack: Reserve pack of the run, or None for the profile value.
    :param no_reserve: Run the frequency cull without the reserve cards.
    """

    input_path: Path
    output_path: Path
    max_cards: int = DEFAULT_MAX_CARDS
    target_words: int = DEFAULT_TARGET_WORDS
    max_uses: int = DEFAULT_MAX_USES
    exempt_top: int = 100
    spoken_weight: float = 0.5
    min_words: int = 3
    max_words: int = 12
    similarity: float = 0.67
    order: str = "coverage"
    name: str | None = None
    reserve_pack: Path | None = None
    no_reserve: bool = False


@dataclass(frozen=True)
class CullRun:
    """Written files and counts of one cull run.

    :param output_path: Output APKG file.
    :param cull_csv: Cull CSV file.
    :param coverage_csv: Coverage CSV file.
    :param manifest_path: Manifest JSON file.
    :param cards: Count of kept cards, with the reserve cards.
    :param frequency_cards: Count of kept input cards.
    :param reserve: Count of kept reserve cards.
    :param targets: Count of target words.
    """

    output_path: Path
    cull_csv: Path
    coverage_csv: Path
    manifest_path: Path
    cards: int
    frequency_cards: int
    reserve: int
    targets: int


def load_frequency(path: Path) -> list[FrequencyRow]:
    """Read the cached frequency list.

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


def tokenize_texts(
    texts: Iterable[str], *, model: str
) -> list[list[tuple[str, str]]]:
    """Tokenize texts with a shared spaCy model.

    :param texts: Sentence texts.
    :param model: spaCy model name.
    :returns: One list of lowercased ``(lemma, part of speech)`` pairs per text.
        Only alphabetic tokens are kept.
    """
    nlp = load_model(model)
    return [
        [(token.lemma_.lower(), token.pos_) for token in doc if token.is_alpha]
        for doc in nlp.pipe(texts, batch_size=TOKENIZE_BATCH)
    ]


def read_names(path: Path) -> set[str]:
    """Read a name stoplist.

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


def plain_sentence(field_value: str) -> str:
    """Remove the bold markup and the no-break spaces of a reserve sentence.

    :param field_value: Raw ``Sentence`` field text.
    :returns: The sentence text with normal spaces.
    """
    return _BOLD_PATTERN.sub("", field_value).translate(_NO_BREAK_SPACES).strip()


def bold_sentence(field_value: str) -> str:
    """Keep the bold markup of a reserve sentence and drop the no-break spaces.

    :param field_value: Raw ``Sentence`` field text.
    :returns: The sentence text with normal spaces and its ``<b>`` tags.
    """
    return field_value.translate(_NO_BREAK_SPACES).strip()


def load_reserve(pack_path: Path, model_name: str) -> list[ReserveCard]:
    """Read one reserve card per target from a reserve pack.

    :param pack_path: Reserve sentence pack file.
    :param model_name: Name of the reserve note model.
    :raises RuntimeError: The pack has no model with that name.
    :returns: One card per target, in the input order of the pack. A target
        with two cards keeps the first card.
    """
    with AnkiPackage(pack_path) as package:
        return _reserve_cards(package, model_name)


def _reserve_cards(package: AnkiPackage, model_name: str) -> list[ReserveCard]:
    """Read the reserve cards from an open reserve pack.

    :param package: Open package.
    :param model_name: Name of the reserve note model.
    :raises RuntimeError: The package has no model with that name.
    :returns: One card per target, in the input order of the package.
    """
    models = package.get_models()
    model_id = coreflow.find_model_id(models, model_name)
    fields = [field["name"] for field in models[model_id]["flds"]]
    index = {name: position for position, name in enumerate(fields)}

    chosen: dict[str, ReserveCard] = {}
    for row in package.get_cards():
        if str(row["mid"]) != model_id:
            continue
        values = row["flds"].split("\x1f")
        if len(values) != len(fields):
            continue
        target = values[index[RESERVE_TARGET_FIELD]].strip()
        if not target or target in chosen:
            continue
        sentence = values[index[RESERVE_SENTENCE_FIELD]]
        chosen[target] = ReserveCard(
            sentence=plain_sentence(sentence),
            html=bold_sentence(sentence),
            english=values[index[RESERVE_ENGLISH_FIELD]].strip(),
            target=target,
            gloss=values[index[RESERVE_GLOSS_FIELD]].strip(),
            audio=coreflow.sound_name(values[index[RESERVE_AUDIO_FIELD]]),
        )
    return list(chosen.values())


def lemma_sets(
    tokens: Sequence[Sequence[tuple[str, str]]],
    content_pos: frozenset[str],
) -> list[set[str]]:
    """Collect the content lemmas of every card.

    :param tokens: Lemma and part of speech per card.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :returns: One set of content lemmas per card.
    """
    return [
        {lemma for lemma, pos in card_tokens if is_content(pos, content_pos)}
        for card_tokens in tokens
    ]


def reserve_lemma_sets(
    reserve: Sequence[ReserveCard],
    profile: CullProfile,
    nlp_tokens: Sequence[Sequence[tuple[str, str]]] | None = None,
) -> list[set[str]]:
    """Collect the content lemmas of every reserve card.

    :param reserve: Reserve cards.
    :param profile: Language hooks.
    :param nlp_tokens: Tokenized reserve sentences. The function tokenizes the
        reserve sentences when this is missing.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: One set of content lemmas per reserve card.
    """
    if nlp_tokens is None:
        nlp_tokens = profile.tokenize_texts(card.sentence for card in reserve)
    if len(nlp_tokens) != len(reserve):
        raise ValueError("The reserve list and the token list differ in size.")
    return lemma_sets(nlp_tokens, profile.content_pos)


def dedupe_reserve(
    reserve: Sequence[ReserveCard],
    reserve_sets: Sequence[set[str]],
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    similarity: float = 0.67,
    *,
    content_pos: frozenset[str],
) -> list[ReserveCard]:
    """Drop the reserve cards that the input deck already holds.

    A reserve card is a duplicate when its sentence is in the input deck, or
    when its lemma Jaccard against an input card reaches the limit.

    :param reserve: Reserve cards.
    :param reserve_sets: Content lemmas per reserve card.
    :param cards: Input cards.
    :param tokens: Lemma and part of speech per input card.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The reserve cards that the input deck does not hold, in order.
    """
    kept, _sets = dedupe_reserve_sets(
        reserve, reserve_sets, cards, tokens, similarity, content_pos=content_pos
    )
    return kept


def dedupe_reserve_sets(
    reserve: Sequence[ReserveCard],
    reserve_sets: Sequence[set[str]],
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    similarity: float = 0.67,
    *,
    content_pos: frozenset[str],
) -> tuple[list[ReserveCard], list[set[str]]]:
    """Drop the duplicate reserve cards and return the lemma sets of the rest.

    The input cards get a lemma inverted index, so the test visits only the
    input cards that share a lemma with the reserve card.

    :param reserve: Reserve cards.
    :param reserve_sets: Content lemmas per reserve card.
    :param cards: Input cards.
    :param tokens: Lemma and part of speech per input card.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The kept reserve cards and their content lemma sets, in order.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    if len(reserve) != len(reserve_sets):
        raise ValueError("The reserve list and the lemma set list differ in size.")

    sentences = {card.sentence for card in cards}
    sets = lemma_sets(tokens, content_pos)
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
    profile: CullProfile,
) -> tuple[int, int]:
    """Count the reserve concepts that the merged deck covers.

    A concept is covered when its form, or the lemma of its head word, appears
    in at least one kept card. The card does not need to be assigned to that
    concept.

    :param concepts: Reserve target words, one per concept of the pack.
    :param sentences: Sentence text of every kept card.
    :param tokens: Lemma and part of speech of every kept card.
    :param profile: Language hooks.
    :raises ValueError: A sentence list and its token list differ in size.
    :returns: The covered concept count and the concept total of the profile.
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
    covered = 0
    for concept in concepts:
        if profile.concept_forms(concept) & merged:
            covered += 1
    return covered, profile.swadesh_total


def select_cards(
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    targets: Sequence[TargetWord],
    ranks: dict[str, int],
    *,
    max_cards: int = DEFAULT_MAX_CARDS,
    max_uses: int = DEFAULT_MAX_USES,
    exempt_top: int = 100,
    similarity: float = 0.67,
    passed: Sequence[int] | None = None,
    reserved: Sequence[ReserveCard] = (),
    reserved_sets: Sequence[set[str]] = (),
    content_pos: frozenset[str],
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
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :raises ValueError: A card list and its token list differ in size.
    :returns: The kept cards in gain order, the lemma uses, and the reasons.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    if len(reserved) != len(reserved_sets):
        raise ValueError("The reserve list and the lemma set list differ in size.")

    sets = lemma_sets(tokens, content_pos)
    exempt = {target.lemma for target in targets[:exempt_top]}
    weights = {target.lemma: target.weight for target in targets}

    def weight(lemma: str) -> float:
        value = weights.get(lemma)
        if value is None:
            # The weight uses the core rank sentinel and hardest_rank uses
            # UNRANKED_ORDER, as the original cull run did.
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
    picked: dict[int, float] = {}
    shared: dict[str, list[int]] = {}

    def key(index: int, value: float) -> tuple[float, int, int]:
        return (-value, len(tokens[index]), index)

    def reject(index: int) -> str | None:
        if _is_similar(sets[index], sets, shared, similarity):
            return "similar"
        return None

    def on_pick(index: int, value: float) -> None:
        picked[index] = value
        for lemma in sets[index]:
            uses[lemma] += 1
            shared.setdefault(lemma, []).append(index)

    kept, reasons = greedy_cover(
        indices,
        gain=gain,
        key=key,
        reject=reject,
        on_pick=on_pick,
        limit=frequency_cards,
        precheck=lambda index: "use_cap" if capped(index) else None,
    )

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
            if is_content(pos, content_pos) and lemma in target_ranks
        }
        if hits:
            card_targets[index] = tuple(
                sorted(hits, key=lambda lemma: (target_ranks[lemma], lemma))
            )
        card_hardest[index] = hardest_rank(card_tokens, ranks, content_pos)

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
    for other in others:
        if at_limit(lemma_set, sets[other], similarity):
            return True
    return False


def order_cards(
    kept: Sequence[int],
    tokens: Sequence[Sequence[tuple[str, str]]],
    ranks: dict[str, int],
    order: str = "coverage",
    *,
    content_pos: frozenset[str],
) -> list[int]:
    """Order the kept cards for learning.

    :param kept: Input indices of the kept cards.
    :param tokens: Lemma and part of speech per card.
    :param ranks: Lemma to blended frequency rank.
    :param order: ``coverage`` sorts on ``hardest_rank``; ``input`` keeps the
        given order.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
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
            hardest_rank(tokens[index], ranks, content_pos),
            len(tokens[index]),
            index,
        ),
    )


def hardest_rank(
    card_tokens: Sequence[tuple[str, str]],
    ranks: dict[str, int],
    content_pos: frozenset[str],
) -> int:
    """Give the blended frequency rank of the hardest content lemma of a card.

    :param card_tokens: Lemma and part of speech of the card sentence.
    :param ranks: Lemma to blended frequency rank.
    :param content_pos: Parts of speech that count as content, empty for every
        token.
    :returns: The rank of the least frequent content lemma, or
        :data:`UNRANKED_ORDER` without a content lemma.
    """
    return max(
        (
            ranks.get(lemma, UNRANKED_ORDER)
            for lemma, pos in card_tokens
            if is_content(pos, content_pos)
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
    write_rows(path, CULL_COLUMNS, rows)


def _reserve_row(card: ReserveCard) -> dict[str, Any]:
    """Build one row of the cull CSV for a reserve card.

    :param card: Reserve card.
    :returns: One row mapping with the reserve source.
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
    write_rows(path, COVERAGE_COLUMNS, rows)


def coverage_status(count: int) -> str:
    """Give the coverage status of a target count.

    :param count: Count of kept cards that hold the target word.
    :returns: ``covered`` for two or more cards, ``partial`` for one, ``absent``
        for zero.
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


def run_cull(request: CullRequest, profile: CullProfile) -> CullRun:
    """Run stages C0 to C5 of the cull and write the four output files.

    :param request: Options of the run.
    :param profile: Language hooks.
    :returns: The written paths and the counts of the run.
    """
    apkg_path = Path(request.input_path)
    output_path = Path(request.output_path)
    name = request.name or profile.default_deck_name
    reserve_pack = (
        request.reserve_pack if request.reserve_pack is not None else profile.reserve_pack
    )
    cards = coreflow.read_input_deck(
        apkg_path,
        profile.model_name,
        profile.sentence_field,
        profile.english_field,
        profile.audio_field,
        profile.clean_sentence,
    )
    frequency = profile.load_frequency()
    ranks = profile.frequency_ranks(frequency, request.spoken_weight)
    names = profile.read_names()
    tokens = profile.tokenize_texts(card.sentence for card in cards)
    targets, content_lemmas = build_targets(
        cards,
        tokens,
        ranks,
        names,
        target_words=request.target_words,
        min_lemma_length=profile.min_lemma_length,
        content_pos=profile.content_pos,
    )
    passed, reasons = filter_cards(
        cards,
        tokens,
        min_words=request.min_words,
        max_words=request.max_words,
        card_rule=partial(filter_reason, profile=profile),
    )
    reserve: list[ReserveCard] = []
    reserve_sets: list[set[str]] = []
    all_reserve: list[ReserveCard] = []
    if not request.no_reserve and reserve_pack is not None:
        all_reserve = load_reserve(reserve_pack, profile.reserve_model_name)
        reserve = list(all_reserve)
        reserve_sets = reserve_lemma_sets(reserve, profile)
        reserve, reserve_sets = dedupe_reserve_sets(
            reserve,
            reserve_sets,
            cards,
            tokens,
            request.similarity,
            content_pos=profile.content_pos,
        )
    selection = select_cards(
        cards,
        tokens,
        targets,
        ranks,
        max_cards=request.max_cards,
        max_uses=request.max_uses,
        exempt_top=request.exempt_top,
        similarity=request.similarity,
        passed=passed,
        reserved=reserve,
        reserved_sets=reserve_sets,
        content_pos=profile.content_pos,
    )
    ordered = order_cards(
        selection.kept,
        tokens,
        ranks,
        order=request.order,
        content_pos=profile.content_pos,
    )

    rejected = Counter(reasons)
    kept_lemmas = set().union(*reserve_sets) if reserve_sets else set()
    kept_lemmas.update(
        lemma
        for index in selection.kept
        for lemma, pos in tokens[index]
        if is_content(pos, profile.content_pos)
    )
    covered = sum(1 for target in targets if selection.uses[target.lemma])
    twice = sum(1 for target in targets if selection.uses[target.lemma] >= 2)
    exempt = selection.exempt
    exempt_max = max(
        (
            count
            for lemma, count in selection.uses.items()
            if lemma in exempt
        ),
        default=0,
    )
    frequency_uses: Counter[str] = Counter()
    for index in selection.kept:
        frequency_uses.update(
            {
                lemma
                for lemma, pos in tokens[index]
                if is_content(pos, profile.content_pos)
            }
        )
    reserve_uses: Counter[str] = Counter()
    for lemma_set in reserve_sets:
        reserve_uses.update(lemma_set)
    frequency_max = max(
        (count for lemma, count in frequency_uses.items() if lemma not in exempt),
        default=0,
    )
    reserve_max = max(
        (count for lemma, count in reserve_uses.items() if lemma not in exempt),
        default=0,
    )
    all_concepts = [card.target for card in all_reserve]
    if reserve:
        all_concepts += list(profile.swadesh_exceptions)
        merged_sentences = [card.sentence for card in reserve]
        merged_sentences += [cards[index].sentence for index in ordered]
        merged_tokens = profile.tokenize_texts(merged_sentences)
        swadesh_covered, swadesh_total = swadesh_coverage(
            all_concepts, merged_sentences, merged_tokens, profile
        )
    else:
        swadesh_covered, swadesh_total = 0, profile.swadesh_total
    mass = spoken_mass(frequency, kept_lemmas)

    print(f"Input:              {apkg_path}")
    print(f"Input cards:        {len(cards)}")
    print(f"Passed:             {len(passed)}")
    for rule in FILTER_RULES:
        print(f"  Rejected {rule}:       {rejected.get(rule, 0)}")
    print(f"Targets:            {len(targets)}")
    print(f"Content lemmas:     {content_lemmas}")
    print(f"Reserve:            {len(reserve)}")
    print(f"Swadesh:            {swadesh_covered}/{swadesh_total}")
    print(f"Cards:              {len(selection.kept) + len(reserve)}")
    print(f"  Frequency cards:  {len(selection.kept)}")
    print(f"Distinct lemmas:    {len(kept_lemmas)}")
    print(f"Covered targets:    {covered}")
    print(f"Targets with 2:     {twice}")
    print(f"Max uses frequency: {frequency_max}")
    print(f"Max uses reserve:   {reserve_max}")
    print(f"Max uses exempt:    {exempt_max}")
    print(f"spoken_mass:        {mass:.3f}")
    print()
    print("First 10 kept cards")
    target_ranks = {target.lemma: target.rank for target in targets}
    for position, index in enumerate(ordered[:10], start=1):
        card = cards[index]
        hits = sorted(
            {
                lemma
                for lemma, pos in tokens[index]
                if is_content(pos, profile.content_pos) and lemma in target_ranks
            },
            key=lambda lemma: target_ranks[lemma],
        )
        print(f"{position:>2}. {card.sentence}")
        print(f"    {card.english}")
        print(f"    targets: {', '.join(hits)}")

    cull_csv, coverage_csv, manifest_path = side_paths(output_path)
    write_cull_csv(cards, reasons, selection, ordered, cull_csv, reserve=reserve)
    write_coverage_csv(targets, selection, coverage_csv)
    write_deck(
        apkg_path,
        cards,
        ordered,
        output_path,
        name,
        profile.model_name,
        templates=profile.templates,
        card_css=profile.card_css,
        reserve=reserve,
        reserve_pack_path=reserve_pack,
        reserve_fields=profile.reserve_fields,
    )
    kept_order = kept_order_hash(cards, ordered, reserve=reserve)
    write_manifest(
        manifest_path,
        source={
            "input": str(apkg_path),
            "deck": name,
            "model": profile.model_name,
        },
        params={
            "max_cards": request.max_cards,
            "target_words": request.target_words,
            "max_uses": request.max_uses,
            "exempt_top": request.exempt_top,
            "spoken_weight": request.spoken_weight,
            "min_words": request.min_words,
            "max_words": request.max_words,
            "similarity": request.similarity,
            "order": request.order,
            "name": name,
        },
        counts={
            "input": len(cards),
            "passed": len(passed),
            "cards": len(ordered) + len(reserve),
            "frequency_cards": len(ordered),
            "reserve": len(reserve),
            "content": len(kept_lemmas),
            "targets": len(targets),
            "covered": covered,
            "covered2": twice,
            "swadesh_covered": swadesh_covered,
            "swadesh_total": swadesh_total,
        },
        frequency={
            "path": str(profile.frequency_path),
            "spoken_weight": request.spoken_weight,
        },
        files={
            "cull_csv": sha256_value(cull_csv),
            "coverage_csv": sha256_value(coverage_csv),
            "input_apkg": sha256_value(apkg_path),
        },
        kept_order=kept_order,
        licenses=profile.licenses,
        reserve_pack=str(reserve_pack) if reserve else "",
        swadesh_exceptions=profile.swadesh_exceptions if reserve else (),
    )
    print()
    print(f"wrote {output_path}")
    print(f"wrote {cull_csv}")
    print(f"wrote {coverage_csv}")
    print(f"wrote {manifest_path}")
    return CullRun(
        output_path=output_path,
        cull_csv=cull_csv,
        coverage_csv=coverage_csv,
        manifest_path=manifest_path,
        cards=len(ordered) + len(reserve),
        frequency_cards=len(ordered),
        reserve=len(reserve),
        targets=len(targets),
    )


def build_frequency_cache(
    tatoeba_path: Path, out_path: Path, profile: CullProfile
) -> int:
    """Build the cached frequency list from the Tatoeba corpus.

    One tokenize pass counts the lemmas of the Tatoeba file. The written term
    comes from ``wordfreq``.

    :param tatoeba_path: Tatoeba sentence TSV, columns ``id``, ``lang``, and
        ``text``.
    :param out_path: Output TSV file.
    :param profile: Language hooks.
    :returns: Number of written rows.
    """
    from wordfreq import zipf_frequency

    counts, content_tokens = count_tatoeba_lemmas(tatoeba_path, profile)
    rows: list[FrequencyRow] = []
    for word, count in counts.items():
        zipf = zipf_frequency(word, profile.wordfreq_lang)
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


def count_tatoeba_lemmas(
    path: Path, profile: CullProfile
) -> tuple[Counter[str], int]:
    """Count the lemmas of a Tatoeba sentence file.

    :param path: Tatoeba sentence TSV.
    :param profile: Language hooks.
    :returns: Lemma counts, and the count of tokens with a content part of
        speech.
    """
    counts: Counter[str] = Counter()
    content_tokens = 0
    for tokens in profile.tokenize_texts(_tatoeba_texts(path)):
        for lemma, pos in tokens:
            counts[lemma] += 1
            if is_content(pos, profile.content_pos):
                content_tokens += 1
    return counts, content_tokens


def _tatoeba_texts(path: Path) -> Iterator[str]:
    """Read the sentence text of a Tatoeba sentence file.

    :param path: Tatoeba sentence TSV.
    :returns: One sentence text per row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.reader(handle, delimiter="\t"):
            if len(record) >= 3:
                yield record[2]
