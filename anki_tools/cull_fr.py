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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Sequence

from anki_tools.package import AnkiPackage
from anki_tools.sentence_pack.audit import sha256_value
from anki_tools.sentence_pack.audio import probe_duration
from anki_tools.sentence_pack.languages.french import TOKENIZER

FREQUENCY_PATH = Path("data/french_frequency.tsv")
NAMES_PATH = Path("vocab/french_names.txt")
TATOEBA_PATH = Path("data/tatoeba/fra_sentences.tsv")

MODEL_NAME = "French Sentences"
FRENCH_FIELD = "French"
AUDIO_FIELD = "sound"
ENGLISH_FIELD = "Back"

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

CULL_COLUMNS = (
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
    """

    kept: list[int]
    uses: Counter[str]
    gain: dict[int, float]
    exempt: set[str]
    reasons: dict[int, str]
    card_targets: dict[int, tuple[str, ...]] = field(default_factory=dict)
    card_hardest: dict[int, int] = field(default_factory=dict)


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
                sentence=values[french].strip(),
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
    nlp = _load_nlp()
    texts = (card.sentence for card in cards)
    return [
        [(token.lemma_.lower(), token.pos_) for token in doc if token.is_alpha]
        for doc in nlp.pipe(texts, batch_size=256)
    ]


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
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The kept cards in gain order, the lemma uses, and the reasons.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")

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

    while heap and len(kept) < max_cards:
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
) -> None:
    """Write one row per input card: the kept cards in order, then the rest.

    :param cards: Input cards.
    :param reasons: Filter reason per input index, empty for a passed card.
    :param selection: Result of the selection.
    :param ordered: Input indices of the kept cards in the final order.
    :param path: Output CSV file.
    """
    kept = set(ordered)
    rows = [_cull_row(cards[index], True, "", selection, index) for index in ordered]
    for index, card in enumerate(cards):
        if index in kept:
            continue
        reason = reasons[index] or selection.reasons.get(index, "")
        rows.append(_cull_row(card, False, reason, selection, index))
    _write_rows(path, CULL_COLUMNS, rows)


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


def kept_order_hash(cards: Sequence[InputCard], ordered: Sequence[int]) -> str:
    """Hash the kept sentences in their final order.

    :param cards: Input cards.
    :param ordered: Input indices of the kept cards in the final order.
    :returns: ``sha256:`` followed by the digest of the joined sentences.
    """
    text = "\n".join(cards[index].sentence for index in ordered)
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
) -> None:
    """Write the manifest JSON of a cull run.

    :param path: Output JSON file.
    :param source: Input APKG path, deck name, and model name.
    :param params: Options of the run.
    :param counts: Card, lemma, and target counts.
    :param frequency: Frequency source paths and the spoken weight.
    :param files: ``sha256:`` values of the output files.
    :param kept_order: Hash of the kept sentences in their final order.
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    target.write_text(text + "\n", encoding="utf-8")


def write_deck(
    input_path: Path,
    cards: Sequence[InputCard],
    kept: Sequence[int],
    output_path: Path,
    deck_name: str,
) -> dict[str, int]:
    """Write the culled deck to an APKG file.

    The cards of every note that is not kept are deleted, and the media files
    that no kept card uses are removed with them.

    :param input_path: Input APKG file.
    :param cards: Input cards.
    :param kept: Input indices of the kept cards.
    :param output_path: Output APKG file.
    :param deck_name: Name of the output deck.
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
        package.save(target)

    with AnkiPackage(target) as package:
        notes = len(package.get_notes())
        media = len(package.get_media_mapping())
    if notes != len(kept_notes):
        raise RuntimeError(f"The output APKG has {notes} notes, not {len(kept_notes)}.")
    return {"notes": notes, "media": media, "items": len(kept_notes)}


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
        audio duration per file.
    """
    with AnkiPackage(path) as package:
        models = {key: model.get("name") for key, model in package.get_models().items()}
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
    uses: Counter[str] = Counter()
    for card_tokens in inputs.tokens():
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
    cards = inputs.output()["cards"]
    tokens = inputs.tokens()
    problems: list[str] = []
    seen: set[str] = set()
    for card, card_tokens in zip(cards, tokens, strict=True):
        reason = _filter_reason(card, card_tokens, min_words, max_words)
        if not reason and card.sentence in seen:
            reason = "F9"
        seen.add(card.sentence)
        if reason:
            problems.append(f"{reason} {card.sentence[:40]}")
    if problems:
        return False, f"{len(problems)} kept card(s) fail: {_sample(problems)}"
    return True, f"{len(cards)} kept cards pass F1 to F9"


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
