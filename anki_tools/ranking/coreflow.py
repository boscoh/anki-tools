"""Shared core deck flow: targets, filter, selection, and files.

The module builds the core list of spec section 2, applies the rules of section
3, runs the selection of section 4, orders the cards of section 6, states the
values of section 8, and records the counts of section 9. A language package
supplies the :class:`anki_tools.ranking.scan.CoreProfile` adapters, and this
module reads every language-specific value from it.

The written package, the side files, the manifest, and the report come from the
assembler in ``anki_tools/ranking/assemble.py``. The shared gates live in
``anki_tools/ranking/coregates.py``.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Protocol, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import (
    OutputShape,
    find_model_id,
    find_only_model_id,
)
from anki_tools.ranking import assemble
from anki_tools.ranking.cardfilter import _WORD_PATTERN, filter_cards
from anki_tools.ranking.core import CoreCandidate, CoreOptions, CoreResult
from anki_tools.ranking.core import select as select_core
from anki_tools.ranking.core import target_quotas
from anki_tools.ranking.corelist import CONTENT_POS, CoreTarget, SenseWords, build_core_targets, build_targets
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.scan import CoreProfile, scan
from anki_tools.sentence_pack.audio import probe_duration
from anki_tools.sentence_pack.english_sense import english_tokens

AUDIO_PROBE_WORKERS = 8

CORE_COLUMNS: tuple[str, ...] = (
    "note_id",
    "index",
    "sentence",
    "english",
    "audio",
    "kept",
    "pass",
    "reason",
    "targets",
    "assigned",
    "value",
    "cost",
    "units",
    "new_lemmas",
    "sense",
    "bold",
)
COVERAGE_COLUMNS: tuple[str, ...] = (
    "key",
    "word",
    "gloss",
    "tier",
    "position",
    "quota",
    "count",
    "status",
)
FILTER_KEYS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "C1", "C2", "C3", "C4", "C5")
TARGET_SEPARATOR = "|"
STATUSES = ("covered", "partial", "absent", "absent_sense", "unchecked")

OUTPUT_FIELDS: tuple[str, ...] = ("Audio", "Sentence", "English", "Core", "CoreGloss")
OUTPUT_AUDIO_FIELD = "Audio"
OUTPUT_SENTENCE_FIELD = "Sentence"
OUTPUT_ENGLISH_FIELD = "English"
OUTPUT_ROMANIZATION_FIELD = "Romanization"
OUTPUT_CORE_FIELD = "Core"
OUTPUT_GLOSS_FIELD = "CoreGloss"

#: Reason of a kept card that holds no core word, so the run does not write it.
NO_CORE_REASON = "no_core"

ROMANIZATION_BLOCK = re.compile(
    r"\{\{#Romanization\}\}.*?\{\{/Romanization\}\}", re.DOTALL
)


def deck_templates(profile: "DeckProfile") -> tuple[tuple[str, str], ...]:
    """Give the templates of a run, minus the romanization block when the
    language has no romanization field.

    :param profile: Language hooks of the run.
    :returns: The template pairs of the output note type.
    """
    if profile.romanization_source is not None:
        return tuple(profile.templates)
    return tuple(
        (qfmt, ROMANIZATION_BLOCK.sub("", afmt).strip())
        for qfmt, afmt in profile.templates
    )


def output_fields(profile: "DeckProfile") -> tuple[str, ...]:
    """Give the output field names of a deck profile.

    :param profile: Language hooks of the run.
    :returns: ``Audio``, ``Sentence``, ``English``, the optional
        ``Romanization``, ``Core``, and ``CoreGloss``.
    """
    names = [*OUTPUT_FIELDS[:3]]
    if profile.romanization_source is not None:
        names.append(OUTPUT_ROMANIZATION_FIELD)
    names.extend(OUTPUT_FIELDS[3:])
    return tuple(names)


class DeckProfile(Protocol):
    """Contract of the output values and the written note type."""

    @property
    def input_model_name(self) -> str:
        """Give the name of the input note type."""
        ...

    @property
    def input_sentence_field(self) -> str:
        """Give the name of the input sentence field."""
        ...

    @property
    def input_english_field(self) -> str:
        """Give the name of the input English field."""
        ...

    @property
    def input_audio_field(self) -> str:
        """Give the name of the input audio field."""
        ...

    @property
    def romanization_source(self) -> str | None:
        """Give the input field of the romanization line, or None."""
        ...

    @property
    def templates(self) -> tuple[tuple[str, str], ...]:
        """Give the card templates of the output."""
        ...

    def card_css(self) -> str:
        """Give the CSS of the output note type."""
        ...



@dataclass(frozen=True)
class CoreRequest:
    """Options of one core run.

    :param input_path: Input APKG file.
    :param output_path: Output APKG file.
    :param core_words: Tier F words after the Swadesh set.
    :param options: Selection settings.
    :param min_units: Minimum word count of a card, or None for the profile
        value.
    :param max_units: Maximum word count of a card, or None for the profile
        value.
    :param spoken_weight: Weight of the spoken term in the blend.
    :param sense_path: Sense file of the tier F targets, or None for the
        profile value.
    :param loan_path: Loan word list of rule C4, or None for the profile value.
    :param bold_target: Bold the assigned target in the sentence.
    :param max_unchecked_sense: Unchecked targets that gate K16 accepts.
    :param max_unranked: Unranked tier F targets that gate K20 accepts.
    """

    input_path: Path
    output_path: Path
    core_words: int
    options: CoreOptions
    min_units: int | None = None
    max_units: int | None = None
    spoken_weight: float = 0.5
    sense_path: Path | None = None
    loan_path: Path | None = None
    bold_target: bool = True
    max_unchecked_sense: int = 0
    max_unranked: int = 0


@dataclass
class CoreRun:
    """Written files and counts of one core run.

    :param output_path: Output APKG file.
    :param core_csv: Card CSV file.
    :param coverage_csv: Coverage CSV file.
    :param manifest_path: Manifest JSON file.
    :param rank_csv: Ranking CSV of ``reorder_deck``.
    :param cards: Count of kept cards.
    :param targets: Count of targets that the input deck holds.
    :param covered1: Targets with at least one card.
    :param covered2: Targets with at least two cards.
    """

    output_path: Path
    core_csv: Path
    coverage_csv: Path
    manifest_path: Path
    rank_csv: Path
    cards: int = 0
    targets: int = 0
    covered1: int = 0
    covered2: int = 0


def read_senses(path: Path) -> dict[str, SenseWords]:
    """Read the sense file of the tier F targets.

    The file has the columns ``word``, ``senses``, and ``false_friends``. The
    two word columns hold comma-separated English words. A missing file gives
    no rows, so every tier F target is unchecked.

    :param path: TSV file.
    :returns: Lemma to its sense words, false-friend words, and first sense.
    """
    table: dict[str, SenseWords] = {}
    source = Path(path)
    if not source.exists():
        return table
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            word = (row.get("word") or "").strip().casefold()
            if not word:
                continue
            senses = _word_set(row.get("senses") or "")
            table[word] = SenseWords(
                senses=senses,
                false_friends=_word_set(row.get("false_friends") or ""),
                first=_first_word(row.get("senses") or ""),
            )
    return table


def _word_set(text: str) -> frozenset[str]:
    """Split a comma-separated word list and casefold it."""
    return frozenset(
        part.strip().casefold() for part in text.split(",") if part.strip()
    )


def _first_word(text: str) -> str:
    """Give the first word of a comma-separated list, casefolded."""
    for part in text.split(","):
        word = part.strip().casefold()
        if word:
            return word
    return ""


def read_loanwords(path: Path) -> frozenset[str]:
    """Read the loan word list of rule C4.

    :param path: Text file, one word per line. A ``#`` line is a comment.
    :returns: Casefolded words, empty when the file is missing.
    """
    source = Path(path)
    if not source.exists():
        return frozenset()
    words = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            words.add(text.casefold())
    return frozenset(words)


def sense_match(
    english: str,
    senses: frozenset[str],
    false_friends: frozenset[str],
) -> bool:
    """Test one hit against the English sense of the target.

    :param english: Paired English sentence.
    :param senses: Sense words of the target. An empty set accepts the hit and
        makes the target unchecked.
    :param false_friends: False-friend words of the target.
    :returns: True when the English shows the sense.
    """
    if not senses:
        return True
    tokens = set(english_tokens(english))
    return bool(senses & tokens) and not bool(false_friends & tokens)


def holds_tier_s_sense(target: CoreTarget, english: str) -> bool:
    """Test a tier S hit against the Swadesh gloss of the row.

    :param target: Tier S target.
    :param english: Paired English sentence.
    :returns: True when the English shows the gloss.
    """
    from anki_tools.sentence_pack.english_sense import holds_sense

    return holds_sense(target.gloss, english)


def core_hits(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    indexes: Sequence[int],
    targets: Sequence[CoreTarget],
    profile: CoreProfile,
) -> tuple[dict[int, tuple[str, ...]], dict[str, int], dict[str, bool]]:
    """Collect the target hits of each candidate card.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param indexes: Input indices of the candidate cards.
    :param targets: Core list of section 2.
    :param profile: Language hooks.
    :returns: Target keys per input index, the sense failures per target key,
        and the presence of each target in the pool.
    """
    by_form: dict[str, list[CoreTarget]] = {}
    by_lemma: dict[str, list[CoreTarget]] = {}
    for target in targets:
        if target.tier == "S":
            by_form.setdefault(target.form, []).append(target)
        else:
            by_lemma.setdefault(profile.lemma_key(target.form), []).append(target)
    form_set = frozenset(by_form)
    sense_failures: Counter[str] = Counter()
    present: dict[str, bool] = {}
    hits: dict[int, tuple[str, ...]] = {}
    for index in indexes:
        card = cards[index]
        found: list[str] = []
        for form in profile.hits(card.sentence, form_set):
            for target in by_form[form]:
                present[target.key] = True
                if profile.sense_filter is not None and profile.sense_filter(
                    form, card.english
                ):
                    sense_failures[target.key] += 1
                    continue
                if holds_tier_s_sense(target, card.english):
                    found.append(target.key)
                else:
                    sense_failures[target.key] += 1
        for lemma, pos in tokens[index]:
            if pos not in CONTENT_POS:
                continue
            for target in by_lemma.get(profile.lemma_key(lemma), ()):
                present[target.key] = True
                if sense_match(card.english, target.senses, target.false_friends):
                    found.append(target.key)
                else:
                    sense_failures[target.key] += 1
        hits[index] = tuple(sorted(set(found)))
    return hits, dict(sense_failures), present


def _has_predicate(tokens: Sequence[tuple[str, str]]) -> bool:
    """Give rule C1: the sentence has a verb, auxiliary, or adjective."""
    return any(pos in {"VERB", "AUX", "ADJ"} for _lemma, pos in tokens)


def _holds_name(sentence: str, names: Collection[str]) -> bool:
    """Give rule C3: a whole word of the name list occurs in the sentence."""
    words = {word.casefold() for word in _WORD_PATTERN.findall(sentence)}
    return any(name.casefold() in words for name in names)


def _holds_loanword(sentence: str, loanwords: Collection[str]) -> bool:
    """Give rule C4: a loan word occurs in the sentence."""
    words = {word.casefold() for word in _WORD_PATTERN.findall(sentence)}
    for loan in loanwords:
        parts = [
            part.casefold() for part in _WORD_PATTERN.findall(loan)
        ]
        if parts and all(part in words for part in parts):
            return True
    return False


def filter_core_cards(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    media: set[str],
    names: Collection[str],
    loanwords: Collection[str],
    has_orphan: Any,
    *,
    min_units: int,
    max_units: int,
) -> tuple[list[int], list[str]]:
    """Apply rules F1 to F9 and C1 to C5 to the cards.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param media: File names of the input media map.
    :param names: Name list of rule C3.
    :param loanwords: Loan word list of rule C4.
    :param has_orphan: Language hook of rule C2.
    :param min_units: First unit count of the window.
    :param max_units: Last unit count of the window.
    :raises ValueError: The card list and the token list differ in size.
    :returns: The input indices that pass, and the reason per input index. A
        passed card has the empty string as its reason.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    passed, cull_reasons = filter_cards(
        cards, tokens, min_words=min_units, max_words=max_units
    )
    reasons = list(cull_reasons)
    kept: list[int] = []
    for index in passed:
        card = cards[index]
        if not _has_predicate(tokens[index]):
            reasons[index] = "C1"
        elif has_orphan(card.sentence):
            reasons[index] = "C2"
        elif _holds_name(card.sentence, names):
            reasons[index] = "C3"
        elif _holds_loanword(card.sentence, loanwords):
            reasons[index] = "C4"
        elif not card.audio or card.audio not in media:
            reasons[index] = "C5"
        else:
            kept.append(index)
    return kept, reasons



def size_check(quotas: dict[str, int], max_cards: int) -> None:
    """Stop a run that cannot give the quota its examples.

    :param quotas: Quota per target key.
    :param max_cards: Card cap of the run.
    :raises ValueError: The card budget is too small for the quota.
    """
    total = sum(quotas.values())
    if total > 2.1 * max_cards:
        raise ValueError(
            f"{total} examples for {len(quotas)} targets cannot fit "
            f"{max_cards} cards; raise --max-cards or lower --per-word"
        )


def outside_lemmas(
    index: int,
    tokens: Sequence[Sequence[tuple[str, str]]],
    core_forms: frozenset[str],
) -> tuple[str, ...]:
    """Give the content lemmas of a card that are outside the core list.

    :param index: Input index of the card.
    :param tokens: Lemma and part of speech per card.
    :param core_forms: Casefolded forms and lemmas of the core targets.
    :returns: Sorted outside lemmas.
    """
    return tuple(
        sorted(
            {
                lemma
                for lemma, pos in tokens[index]
                if pos in CONTENT_POS
                and lemma.casefold() not in core_forms
            }
        )
    )



def kept_order_hash(cards: Sequence[Any], ordered: Sequence[int]) -> str:
    """Hash the kept sentences in their final order.

    :param cards: Input cards.
    :param ordered: Kept cards in card order.
    :returns: ``sha256:`` followed by the digest of the joined sentences.
    """
    text = "\n".join(cards[index].sentence for index in ordered)
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _media_names(apkg_path: Path) -> set[str]:
    """Give the file names of the input media map."""
    with AnkiPackage(apkg_path) as package:
        return set(package.get_media_mapping().values())


def card_rows(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    result: CoreResult,
    ordered: Sequence[int],
    candidates: Sequence[CoreCandidate],
    targets: Sequence[CoreTarget],
    reasons: Sequence[str],
    *,
    assigned: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Build the card CSV rows in card order.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param result: Selection result.
    :param ordered: Kept cards in card order.
    :param candidates: Cards that passed the filter.
    :param targets: Core list.
    :param reasons: Filter reason per input index.
    :param assigned: Target key of each kept card.
    :returns: One row per kept card, then one row per other input card.
    """
    by_key = {target.key: target for target in targets}
    by_index = {candidate.index: candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []
    introduced: set[str] = set()
    for index in ordered:
        candidate = by_index[index]
        new = [word for word in candidate.outside if word not in introduced]
        introduced.update(candidate.outside)
        target = assigned.get(index, "")
        unchecked = sum(
            1 for key in candidate.targets if not by_key[key].checked
        )
        rows.append(_row(cards, tokens, index, candidate, reasons[index]))
        rows[-1].update(
            {
                "kept": 1,
                "targets": TARGET_SEPARATOR.join(
                    result.credited.get(index, candidate.targets)
                ),
                "pass": result.pick_pass.get(index, ""),
                "assigned": target,
                "value": f"{result.score.get(index, 0.0):.4f}",
                "cost": f"{result.cost.get(index, 0.0):.4f}",
                "new_lemmas": len(new),
                "sense": unchecked,
            }
        )
    kept = set(ordered)
    for index in range(len(cards)):
        if index in kept:
            continue
        candidate = by_index.get(index)
        rows.append(_row(cards, tokens, index, candidate, reasons[index]))
        rows[-1].update(
            {
                "kept": 0,
                "reason": reasons[index] or result.reasons.get(index, ""),
            }
        )
    return rows


def _row(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    index: int,
    candidate: CoreCandidate | None,
    reason: str,
) -> dict[str, Any]:
    """Build one card CSV row with the values that do not depend on a pick."""
    return {
        "note_id": cards[index].note_id,
        "index": index,
        "sentence": cards[index].sentence,
        "english": cards[index].english,
        "audio": cards[index].audio,
        "kept": 0,
        "pass": "",
        "reason": reason,
        "targets": TARGET_SEPARATOR.join(candidate.targets) if candidate else "",
        "assigned": "",
        "value": "0.0000",
        "cost": "0.0000",
        "units": candidate.units if candidate else len(tokens[index]),
        "new_lemmas": 0,
        "sense": 0,
        "bold": 0,
    }


def coverage_rows_of(
    targets: Sequence[CoreTarget],
    result: CoreResult,
    present: dict[str, bool],
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    """Build the coverage CSV rows of every core target.

    :param targets: Core list.
    :param result: Selection result.
    :param present: Presence of each target in the pool.
    :param quotas: Quota per target key.
    :returns: One row per target, in core order.
    """
    rows: list[dict[str, Any]] = []
    for target in targets:
        count = result.uses.get(target.key, 0)
        quota = quotas.get(target.key, 0)
        if count == 0 and not present.get(target.key, False):
            status = "absent"
        elif count == 0:
            status = "absent_sense"
        elif not target.checked:
            status = "unchecked"
        elif count >= quota:
            status = "covered"
        else:
            status = "partial"
        rows.append(
            {
                "key": target.key,
                "word": target.word,
                "gloss": target.gloss,
                "tier": target.tier,
                "position": target.position,
                "quota": quota,
                "count": count,
                "status": status,
            }
        )
    return rows


def manifest_counts(
    rows: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    result: CoreResult,
    candidates: Sequence[CoreCandidate],
    ordered: Sequence[int],
    ranks: dict[str, int],
    options: CoreOptions,
    *,
    bolded: int = 0,
    unbolded: int = 0,
    dropped_no_core: int = 0,
) -> dict[str, int]:
    """Give the manifest counts of a run.

    :param rows: Card CSV rows.
    :param coverage: Coverage CSV rows.
    :param result: Selection result.
    :param candidates: Cards that passed the filter.
    :param ordered: Kept cards in card order.
    :param ranks: Lemma to blended frequency rank.
    :param options: Selection settings.
    :param bolded: Count of kept cards with a bold target.
    :param unbolded: Count of kept cards without a bold target.
    :param dropped_no_core: Count of kept cards that hold no core word.
    :returns: Counts for the manifest.
    """
    by_index = {candidate.index: candidate for candidate in candidates}
    core_rows = [
        row
        for row in rows
        if int(row["kept"]) == 1 and row["pass"] != "tail"
    ]
    outside = {
        word
        for row in core_rows
        for word in by_index[int(row["index"])].outside
    }
    rare = sum(1 for word in outside if ranks.get(word, 10**6) > options.hard_top)
    tail_words = sum(
        int(row["new_lemmas"])
        for row in rows
        if int(row["kept"]) == 1 and row["pass"] == "tail"
    )
    statuses = Counter(row["status"] for row in coverage)
    return {
        "cards": len(ordered),
        "targets": len(coverage),
        "absent": statuses["absent"],
        "absent_sense": statuses["absent_sense"],
        "unchecked": statuses["unchecked"],
        "unranked": sum(
            1 for row in coverage if row["tier"] == "F" and ranks.get(row["word"]) is None
        ),
        "covered1": sum(1 for row in coverage if int(row["count"]) >= 1),
        "covered2": sum(1 for row in coverage if int(row["count"]) >= 2),
        "full": sum(
            1 for row in coverage if int(row["count"]) >= int(row["quota"])
        ),
        "outside": len(outside),
        "outside_rare": rare,
        "tail_words": tail_words,
        "sense_rows": 0,
        "dropped_no_core": dropped_no_core,
        "bolded": bolded,
        "unbolded": unbolded,
        "name_hits": 0,
        "loan_hits": 0,
        "rows": len(rows),
    }




@dataclass(frozen=True)
class InputCard:
    """One card of an input sentence deck.

    :param note_id: Anki note id.
    :param sentence: Sentence text after the language clean step.
    :param english: English translation.
    :param audio: Name of the audio file in the card.
    :param index: Position of the card in the input order.
    """

    note_id: int
    sentence: str
    english: str
    audio: str
    index: int


_SOUND_PATTERN = re.compile(r"\[sound:([^\]]+)\]")


def sound_name(field_value: str) -> str:
    """Read the audio file name from a sound field.

    :param field_value: Raw field text, for example ``[sound:a.mp3]``.
    :returns: The file name, or an empty string when the field has no sound tag.
    """
    match = _SOUND_PATTERN.search(field_value)
    return match.group(1) if match else ""


def read_input_deck(
    path: Path,
    model_name: str,
    sentence_field: str,
    english_field: str,
    audio_field: str,
    clean_sentence: Callable[[str], str],
) -> list[InputCard]:
    """Read one card per note of the sentence model in an APKG file.

    :param path: Input APKG file.
    :param model_name: Name of the note model.
    :param sentence_field: Name of the sentence field.
    :param english_field: Name of the English field.
    :param audio_field: Name of the audio field.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :raises RuntimeError: The package has no model with that name.
    :returns: One card per note, in the input order.
    """
    with AnkiPackage(path) as package:
        models = package.get_models()
        model_id = find_model_id(models, model_name)
        fields = [field["name"] for field in models[model_id]["flds"]]
        index = {name: position for position, name in enumerate(fields)}
        sentence = index[sentence_field]
        english = index[english_field]
        audio = index[audio_field]

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
            sound = _SOUND_PATTERN.search(values[audio])
            cards.append(
                InputCard(
                    note_id=note_id,
                    sentence=clean_sentence(values[sentence]),
                    english=values[english].strip(),
                    audio=sound.group(1) if sound else "",
                    index=len(cards),
                )
            )
        return cards



def output_values(
    profile: DeckProfile,
    input_names: Sequence[str],
    input_values: Sequence[str],
    sentence: str,
) -> list[str]:
    """Give the output field values of one kept note.

    :param profile: Language hooks.
    :param input_names: Field names of the input note type.
    :param input_values: Field values of the input note.
    :param sentence: Clean sentence text of the card.
    :returns: One value per output field. ``Core`` and ``CoreGloss`` are empty.
    """
    index = {name: position for position, name in enumerate(input_names)}

    def value(name: str) -> str:
        position = index.get(name)
        return input_values[position] if position is not None else ""

    values = {
        OUTPUT_AUDIO_FIELD: value(profile.input_audio_field),
        OUTPUT_SENTENCE_FIELD: sentence,
        OUTPUT_ENGLISH_FIELD: value(profile.input_english_field),
        OUTPUT_CORE_FIELD: "",
        OUTPUT_GLOSS_FIELD: "",
    }
    if profile.romanization_source is not None:
        values[OUTPUT_ROMANIZATION_FIELD] = value(profile.romanization_source)
    return [values[name] for name in output_fields(profile)]


def note_values(
    input_path: Path,
    cards: Sequence[Any],
    kept: Sequence[int],
    profile: DeckProfile,
    core_values: Mapping[int, tuple[str, str]],
    *,
    bold: bool = True,
) -> tuple[dict[int, list[str]], int, int]:
    """Give the output field values of every kept note.

    The values hold the core word, the gloss, and the bold text, so the written
    note type is complete in one pass.

    :param input_path: Input APKG file.
    :param cards: Input cards.
    :param kept: Input indices of the kept cards.
    :param profile: Language hooks of the run.
    :param core_values: Input index to ``(word, gloss)`` pair.
    :param bold: Bold the core word in the sentence.
    :returns: Note id to field values, the count of bolded notes, and the count
        of the others.
    """
    names = output_fields(profile)
    word_index = names.index(OUTPUT_CORE_FIELD)
    gloss_index = names.index(OUTPUT_GLOSS_FIELD)
    sentence_index = names.index(OUTPUT_SENTENCE_FIELD)
    highlight = profile.highlight
    by_note = {cards[index].note_id: index for index in kept}
    values: dict[int, list[str]] = {}
    bolded = 0
    with AnkiPackage(input_path) as package:
        models = package.get_models()
        input_names = [
            field["name"]
            for field in models[find_model_id(models, profile.input_model_name)]["flds"]
        ]
        for row in package.get_notes():
            note_id = int(row["id"])
            index = by_note.get(note_id)
            if index is None:
                continue
            card = cards[index]
            field_values = output_values(
                profile,
                input_names,
                str(row["flds"]).split("\x1f"),
                card.sentence,
            )
            word, gloss = core_values.get(index, ("", ""))
            field_values[word_index] = word
            field_values[gloss_index] = gloss
            if bold and highlight is not None and word:
                marked = highlight(card.sentence, word)
                field_values[sentence_index] = marked
                bolded += int(marked != card.sentence)
            values[note_id] = field_values
    if not bold:
        return values, 0, 0
    if highlight is None:
        return values, 0, len(values)
    return values, bolded, len(values) - bolded

def _media_probe_path(path: Path) -> Path:
    """Give a readable path for an audio file, decompressing zstd media."""
    head = path.read_bytes()[:4] if path.exists() else b""
    if head != b"\x28\xb5\x2f\xfd":
        return path
    import io

    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(path.read_bytes())) as reader:
        data = reader.read()
    target = path.with_name(path.name + ".probe")
    target.write_bytes(data)
    return target


def probe_audio(
    media_dir: str | None, media: dict[str, str], names: Collection[str]
) -> dict[str, float | None]:
    """Probe the duration of the audio files of the kept cards.

    :param media_dir: Directory with the unpacked media files.
    :param media: Media file id to name mapping.
    :param names: File names to probe.
    :returns: File name to duration in seconds. The value is None when the
        file is missing or is not a valid MP3.
    """
    if not media_dir:
        return {}
    paths = {
        name: Path(media_dir) / file_id
        for file_id, name in media.items()
        if name in names
    }

    def probe(item: tuple[str, Path]) -> tuple[str, float | None]:
        name, path = item
        if not path.exists():
            return name, None
        probe_path = _media_probe_path(path)
        try:
            return name, probe_duration(probe_path)
        except RuntimeError:
            return name, None
        finally:
            if probe_path != path:
                probe_path.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=AUDIO_PROBE_WORKERS) as pool:
        return dict(pool.map(probe, sorted(paths.items())))


_BOLD_TAG = re.compile(r"</?b>")


def read_output_deck(path: Path) -> list[InputCard]:
    """Read one card per note of the only note type of an output APKG.

    The function reads the ``Audio``, ``Sentence``, and ``English`` fields, and
    it removes the bold tags from the sentence. A note without audio is
    skipped.

    :param path: Output APKG file.
    :raises RuntimeError: The package holds two or more note types.
    :returns: One card per note, in the input order.
    """
    cards: list[InputCard] = []
    with AnkiPackage(path) as package:
        models = package.get_models()
        model_id = find_only_model_id(models)
        names = [field["name"] for field in models[model_id]["flds"]]
        index = {name: position for position, name in enumerate(names)}
        audio_index = index[OUTPUT_AUDIO_FIELD]
        sentence_index = index[OUTPUT_SENTENCE_FIELD]
        english_index = index[OUTPUT_ENGLISH_FIELD]

        chosen: dict[int, Any] = {}
        for row in package.get_cards():
            if str(row["mid"]) != model_id:
                continue
            if len(row["flds"].split("\x1f")) != len(names):
                continue
            note_id = int(row["nid"])
            current = chosen.get(note_id)
            if current is None or int(row["ord"]) < int(current["ord"]):
                chosen[note_id] = row

        for note_id, row in chosen.items():
            values = row["flds"].split("\x1f")
            sound = _SOUND_PATTERN.search(values[audio_index])
            if sound is None:
                continue
            cards.append(
                InputCard(
                    note_id=note_id,
                    sentence=_BOLD_TAG.sub("", values[sentence_index]),
                    english=values[english_index].strip(),
                    audio=sound.group(1),
                    index=len(cards),
                )
            )
    return cards


def read_output(path: Path) -> dict[str, Any]:
    """Read the notes, the decks, and the media of an output APKG.

    :param path: Output APKG file.
    :raises RuntimeError: The package holds two or more note types.
    :returns: Cards, note count, field names, model names, deck names, media
        names, the card ordinals, the card templates and CSS per model, and the
        audio duration per file.
    """
    cards = read_output_deck(path)
    with AnkiPackage(path) as package:
        full = package.get_models(include_templates=True)
        model = full[find_only_model_id(full)]
        name = model.get("name")
        notes = package.get_notes()
        media = package.get_media_mapping()
        names = {card.audio for card in cards if card.audio}
        durations = probe_audio(package.media_dir(), media, names)
        decks = {deck.get("name") for deck in package.get_decks().values()}
        ords = [int(row["ord"]) for row in package.get_cards()]
    return {
        "cards": cards,
        "notes": len(notes),
        "fields": [field["name"] for field in model.get("flds", ())],
        "note_models": {name},
        "decks": decks,
        "media": set(media.values()),
        "audio": durations,
        "ords": ords,
        "templates": {
            name: [
                (template.get("qfmt", ""), template.get("afmt", ""))
                for template in model.get("tmpls", ())
            ]
        },
        "css": {name: model.get("css", "")},
    }


def core_keys(
    result: CoreResult,
    ordered: Sequence[int],
    candidates: Sequence[CoreCandidate],
    targets: Sequence[CoreTarget],
) -> dict[int, str]:
    """Give the target key that each kept card teaches.

    The assigned target wins. A card without an assigned target teaches the
    target of the lowest core position that it holds.

    :param result: Selection result with the assignment.
    :param ordered: Kept cards in card order.
    :param candidates: Cards that passed the filter, with the targets they hold.
    :param targets: Core list.
    :returns: Input index to target key. A card without a target has no key.
    """
    by_key = {target.key: target for target in targets}
    held = {candidate.index: candidate.targets for candidate in candidates}
    keys: dict[int, str] = {}
    for index in ordered:
        key = result.assigned.get(index)
        if key is None and held.get(index):
            key = min(held[index], key=lambda item: (by_key[item].position, item))
        if key is not None:
            keys[index] = key
    return keys


def core_values(
    keys: Mapping[int, str],
    targets: Sequence[CoreTarget],
) -> dict[int, tuple[str, str]]:
    """Give the core word and the gloss that each kept card teaches.

    :param keys: Input index to target key.
    :param targets: Core list.
    :returns: Input index to ``(word, gloss)`` pair.
    """
    by_key = {target.key: target for target in targets}
    return {
        index: (by_key[key].form, by_key[key].gloss)
        for index, key in keys.items()
    }


def coverage_glosses(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Give the word-to-gloss map of a coverage CSV.

    The first row of a word wins, so the lowest core position sets the gloss.

    :param rows: Coverage CSV rows.
    :returns: Word to its coverage gloss.
    """
    glosses: dict[str, str] = {}
    for row in rows:
        word = (row.get("word") or "").strip()
        if word:
            glosses.setdefault(word, (row.get("gloss") or "").strip())
    return glosses


def run_core(request: CoreRequest, profile: CoreProfile) -> CoreRun:
    """Run stages K0 to K6 of the core mode and write the output files.

    :param request: Options of the run.
    :param profile: Language hooks.
    :raises ValueError: The input has no core target, or the budget cannot
        hold the quota.
    :returns: The written paths and the counts of the run.
    """
    apkg_path = Path(request.input_path)
    output_path = Path(request.output_path)
    min_units = request.min_units if request.min_units is not None else profile.min_units
    max_units = request.max_units if request.max_units is not None else profile.max_units
    sense_path = request.sense_path or profile.sense_path
    loan_path = request.loan_path or profile.loan_path
    identity = DeckIdentity(output_path)

    scanned = scan(profile, apkg_path, request.spoken_weight)
    cards = scanned.cards
    tokens = scanned.tokens
    ranks = scanned.ranks
    names = scanned.names
    loanwords = read_loanwords(loan_path)
    senses = read_senses(sense_path)
    media = _media_names(apkg_path)

    table = profile.word_table()
    frequency_targets, _content = build_targets(
        cards,
        tokens,
        ranks,
        names,
        target_words=None,
        min_deck_count=request.options.per_word + 1,
    )
    targets, merged = build_core_targets(
        table,
        frequency_targets,
        senses,
        request.core_words,
    )
    passed, reasons = filter_core_cards(
        cards,
        tokens,
        media,
        names,
        loanwords,
        profile.has_orphan,
        min_units=min_units,
        max_units=max_units,
    )
    hits, sense_failures, present = core_hits(
        cards, tokens, passed, targets, profile
    )
    targets = [target for target in targets if present.get(target.key, False)]
    if not targets:
        raise ValueError("No core target occurs in a candidate card.")
    positions = {target.key: target.position for target in targets}

    shares = {
        target.key: sum(1 for index in passed if target.key in hits[index])
        / len(passed)
        for target in targets
    }
    quotas = target_quotas(positions, shares, request.options)
    size_check(quotas, request.options.max_cards)

    core_forms = frozenset(target.form.casefold() for target in targets)
    candidates = [
        CoreCandidate(
            index=index,
            targets=hits[index],
            outside=outside_lemmas(index, tokens, core_forms),
            units=len(tokens[index]),
        )
        for index in passed
    ]
    result = select_core(
        candidates,
        positions=positions,
        quotas=quotas,
        ranks=ranks,
        options=request.options,
    )
    ordered = result.order
    keys = core_keys(result, ordered, candidates, targets)
    core_words = core_values(keys, targets)
    kept = [index for index in ordered if index in keys]
    dropped = [index for index in ordered if index not in keys]
    reasons = list(reasons)
    for index in dropped:
        reasons[index] = NO_CORE_REASON

    note_fields, bolded, unbolded = note_values(
        apkg_path,
        cards,
        kept,
        profile,
        core_words,
        bold=request.bold_target,
    )

    rows = card_rows(
        cards, tokens, result, kept, candidates, targets, reasons, assigned=keys
    )
    coverage_rows = coverage_rows_of(targets, result, present, quotas)
    counts = manifest_counts(
        rows, coverage_rows, result, candidates, kept, ranks, request.options,
        bolded=bolded, unbolded=unbolded, dropped_no_core=len(dropped),
    )
    counts["rejected_c3"] = sum(1 for reason in reasons if reason == "C3")
    counts["rejected_c4"] = sum(1 for reason in reasons if reason == "C4")
    counts["sense_failures"] = sum(sense_failures.values())
    counts["sense_rows"] = sum(1 for target in targets if target.checked)

    filtered = Counter(reasons)
    report: list[tuple[str | None, str]] = [
        ("Input", str(apkg_path)),
        ("Input cards", str(len(cards))),
        ("Passed", str(len(passed))),
    ]
    report += [
        (f"  Rejected {rule}", str(filtered[rule]))
        for rule in FILTER_KEYS
        if filtered[rule]
    ]
    report += [
        ("Targets", str(len(targets))),
        ("Quota total", str(sum(quotas.values()))),
        ("Cards", str(len(kept))),
        ("Covered once", str(counts["covered1"])),
        ("Covered twice", str(counts["covered2"])),
        ("Outside lemmas", str(counts["outside"])),
        ("Rare outside", str(counts["outside_rare"])),
        ("Unchecked", str(counts["unchecked"])),
        ("Merged S rows", str(len(merged))),
    ]

    files = assemble.write_run(
        assemble.DeckRun(
            identity=identity,
            input_path=apkg_path,
            shape=OutputShape(
                deck_name=identity.title,
                model_name=identity.model_name,
                input_model_name=profile.input_model_name,
                fields=output_fields(profile),
                templates=deck_templates(profile),
                css=profile.card_css(),
            ),
            values=note_fields,
            card_table=assemble.SideTable("core.csv", CORE_COLUMNS, rows),
            coverage_table=assemble.SideTable(
                "coverage.csv", COVERAGE_COLUMNS, coverage_rows
            ),
            manifest=assemble.ManifestValues(
                input_model=profile.input_model_name,
                params={
                    "core_words": request.core_words,
                    "per_word": request.options.per_word,
                    "third_examples": request.options.third_examples,
                    "ubiquitous_share": request.options.ubiquitous_share,
                    "max_cards": request.options.max_cards,
                    "completion_cards": request.options.completion_cards,
                    "tail_cards": request.options.tail_cards,
                    "outside_budget": request.options.outside_budget,
                    "support_top": request.options.support_top,
                    "hard_top": request.options.hard_top,
                    "lambda_new": request.options.lambda_new,
                    "lambda_repeat": request.options.lambda_repeat,
                    "lambda_len": request.options.lambda_len,
                    "lambda_seed": request.options.lambda_seed,
                    "simple_units": request.options.simple_units,
                    "similarity": request.options.similarity,
                    "min_units": min_units,
                    "max_units": max_units,
                    "spoken_weight": request.spoken_weight,
                    "bold_target": request.bold_target,
                    "max_unchecked_sense": request.max_unchecked_sense,
                    "max_unranked": request.max_unranked,
                },
                counts=counts,
                frequency={
                    "path": profile.frequency_source,
                    "spoken_weight": request.spoken_weight,
                },
                kept_order=kept_order_hash(cards, kept),
            ),
            report=report,
            rank_rows=[
                (rank, cards[index].note_id, cards[index].sentence, 1)
                for rank, index in enumerate(kept, start=1)
            ],
        )
    )
    run = CoreRun(
        output_path=files.output_path,
        core_csv=files.card_csv,
        coverage_csv=files.coverage_csv,
        manifest_path=files.manifest_path,
        rank_csv=files.rank_csv or identity.side("rank.csv"),
        cards=len(kept),
        targets=len(targets),
        covered1=counts["covered1"],
        covered2=counts["covered2"],
    )
    if profile.finalize is not None:
        profile.finalize(run)
    return run
