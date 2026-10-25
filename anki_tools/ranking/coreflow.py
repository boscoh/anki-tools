"""Shared core deck flow: targets, filter, selection, and files.

The module builds the core list of spec section 2, applies the rules of section
3, runs the selection of section 4, orders the cards of section 6, writes the
cards of section 8, and records the counts of section 9. A language package
supplies the :class:`anki_tools.ranking.scan.CoreProfile` adapters, and this
module reads every language-specific value from it.

The shared gates live in ``anki_tools/ranking/coregates.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.cardfilter import _WORD_PATTERN, filter_cards
from anki_tools.ranking.core import CoreCandidate, CoreOptions, CoreResult
from anki_tools.ranking.core import select as select_core
from anki_tools.ranking.core import target_quotas
from anki_tools.ranking.corelist import CONTENT_POS, CoreTarget, SenseWords, build_core_targets, build_targets
from anki_tools.ranking.reorder import reorder_deck
from anki_tools.ranking.scan import CoreProfile, scan
from anki_tools.sentence_pack.audio import probe_duration
from anki_tools.sentence_pack.audit import sha256_value
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
    :param name: Output deck name, or None for the profile value.
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
    name: str | None = None
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


def default_output(apkg_path: Path) -> Path:
    """Build the default output path of a core run.

    :param apkg_path: Input APKG file.
    :returns: ``<stem>_core.apkg`` beside the input.
    """
    path = Path(apkg_path)
    return path.parent / f"{path.stem}_core.apkg"


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


def side_paths(output_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build the four side file paths of a core run.

    :param output_path: Output APKG file.
    :returns: The core CSV, the coverage CSV, the manifest, and the rank CSV.
    """
    path = Path(output_path)
    stem = path.with_suffix("") if path.suffix else path
    return (
        Path(f"{stem}.core.csv"),
        Path(f"{stem}.coverage.csv"),
        Path(f"{stem}.manifest.json"),
        Path(f"{stem}.rank.csv"),
    )


def write_rank_csv(
    cards: Sequence[Any], ordered: Sequence[int], path: Path
) -> None:
    """Write the ranking CSV that ``reorder_deck`` reads.

    :param cards: Input cards.
    :param ordered: Kept cards in card order.
    :param path: Output CSV file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "note_id", "sentence", "kept"])
        for rank, index in enumerate(ordered, start=1):
            writer.writerow(
                [rank, cards[index].note_id, cards[index].sentence, 1]
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
) -> list[dict[str, Any]]:
    """Build the card CSV rows in card order.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param result: Selection result.
    :param ordered: Kept cards in card order.
    :param candidates: Cards that passed the filter.
    :param targets: Core list.
    :param reasons: Filter reason per input index.
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
        target = result.assigned.get(index, "")
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
        "bolded": bolded,
        "unbolded": unbolded,
        "name_hits": 0,
        "loan_hits": 0,
        "rows": len(rows),
    }


def write_core_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write the card CSV.

    :param rows: Rows in card order.
    :param path: Output CSV file.
    """
    _write_csv(CORE_COLUMNS, rows, path)


def write_coverage_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write the coverage CSV.

    :param rows: One row per target.
    :param path: Output CSV file.
    """
    _write_csv(COVERAGE_COLUMNS, rows, path)


def _write_csv(
    columns: Sequence[str], rows: Sequence[dict[str, Any]], path: Path
) -> None:
    """Write a CSV file with a fixed header."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(
    path: Path,
    *,
    source: dict[str, Any],
    params: dict[str, Any],
    counts: dict[str, int],
    frequency: dict[str, Any],
    files: dict[str, str],
    kept_order: str,
    licenses: Mapping[str, str] | None = None,
    reserve_pack: str = "",
    swadesh_exceptions: Sequence[str] = (),
) -> None:
    """Write the manifest JSON of a deck run.

    :param path: Output JSON file.
    :param source: Input APKG path, deck name, and model name.
    :param params: Options of the run.
    :param counts: Card, lemma, and target counts.
    :param frequency: Frequency source path and the spoken weight.
    :param files: ``sha256:`` values of the output files.
    :param kept_order: Hash of the kept sentences in their final order.
    :param licenses: License text per data source, or None for no block.
    :param reserve_pack: Reserve pack of the run, empty without a reserve.
    :param swadesh_exceptions: Reserve concepts that no source sentence covers.
    """
    payload: dict[str, Any] = {
        "build_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "params": params,
        "counts": counts,
        "frequency": frequency,
        "files": files,
    }
    if licenses is not None:
        payload["licenses"] = dict(licenses)
    payload["kept_order"] = kept_order
    if reserve_pack:
        payload["reserve_pack"] = reserve_pack
        payload["swadesh_exceptions"] = list(swadesh_exceptions)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_model_id(models: dict[str, dict], name: str) -> str:
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


def _connection(package: AnkiPackage) -> sqlite3.Connection:
    """Give the open database connection of a package."""
    if package.conn is None:
        raise RuntimeError("The package database is closed.")
    return package.conn


def _legacy_decks(package: AnkiPackage) -> dict[str, dict]:
    """Read the deck JSON of a legacy collection."""
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


def _trim_templates(package: AnkiPackage, model_id: str) -> None:
    """Remove every card template after the first one."""
    models = package.get_models(include_templates=True)
    have = len(models[model_id].get("tmpls", ()))
    for number in reversed(range(1, have)):
        package.remove_model_template(model_id, number)


RESERVE_TAGS = " swadesh "


def insert_reserve_cards(
    package: AnkiPackage,
    reserve: Sequence[Any],
    reserve_pack_path: Path,
    deck_id: int,
    reserve_fields: Callable[[Any], Sequence[str]],
) -> None:
    """Insert the reserve notes and cards into an open package.

    The note and card rows copy a kept row of the input model, so every column
    of the package schema stays valid.

    :param package: Open output package after the non-kept cards are deleted.
    :param reserve: Reserve cards to insert.
    :param reserve_pack_path: Pack with the reserve media files.
    :param deck_id: Target deck id.
    :param reserve_fields: Callable that gives the field values of one card.
    :raises RuntimeError: The package has no template row, or a reserve audio
        file is not in the pack.
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
            raise RuntimeError("The reserve pack has no temporary directory.")
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
        fields = "\x1f".join(reserve_fields(card))
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


def write_deck(
    input_path: Path,
    cards: Sequence[Any],
    kept: Sequence[int],
    output_path: Path,
    deck_name: str,
    model_name: str,
    *,
    templates: tuple[tuple[str, str], ...] | None = None,
    card_css: Callable[[], str] | None = None,
    reserve: Sequence[Any] = (),
    reserve_pack_path: Path | None = None,
    reserve_fields: Callable[[Any], Sequence[str]] | None = None,
) -> dict[str, int]:
    """Write the kept cards to an APKG file.

    The cards of every note that is not kept are deleted, and the media files
    that no kept card uses are removed with them.

    :param input_path: Input APKG file.
    :param cards: Input cards.
    :param kept: Input indices of the kept cards.
    :param output_path: Output APKG file.
    :param deck_name: Name of the output deck.
    :param model_name: Name of the note model.
    :param templates: Card templates of the output, or None to keep the first
        template of the input model.
    :param card_css: Callable that gives the CSS of the output, or None to
        keep the input CSS.
    :param reserve: Reserve cards to add to the input model.
    :param reserve_pack_path: Pack with the reserve media files.
    :param reserve_fields: Callable that gives the field values of one reserve
        card. A run with reserve cards needs it.
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
        models = package.get_models(include_templates=True)
        model_id = find_model_id(models, model_name)
        if reserve:
            deck_id = int(next(iter(package.get_decks())))
            insert_reserve_cards(
                package, reserve, reserve_pack_path, deck_id, reserve_fields
            )
        if card_css is not None:
            package.update_model_template(model_id=model_id, css=card_css())
        if templates is None:
            _trim_templates(package, model_id)
        else:
            for number, (qfmt, afmt) in enumerate(templates):
                package.update_model_template(
                    model_id=model_id, qfmt=qfmt, afmt=afmt, template_index=number
                )
            have = len(models[model_id].get("tmpls", ()))
            for number in reversed(range(len(templates), have)):
                package.remove_model_template(model_id, number)
        package.save(target)

    expected = len(kept_notes) + len(reserve)
    with AnkiPackage(target) as package:
        notes = len(package.get_notes())
        media = len(package.get_media_mapping())
    if notes != expected:
        raise RuntimeError(f"The output APKG has {notes} notes, not {expected}.")
    return {"notes": notes, "media": media, "items": expected}


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


def read_output(path: Path, profile: CoreProfile) -> dict[str, Any]:
    """Read the notes, the decks, and the media of an output APKG.

    :param path: Output APKG file.
    :param profile: Language hooks.
    :returns: Cards, note count, model names, deck names, media names, the
        card templates and CSS per model, and the audio duration per file.
    """
    cards = profile.load_input_deck(path)
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
        media = package.get_media_mapping()
        names = {card.audio for card in cards if card.audio}
        durations = probe_audio(package.temp_dir, media, names)
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


def _model_id(apkg_path: Path, model_name: str) -> str:
    """Give the model id of the core sentence model."""
    with AnkiPackage(apkg_path) as package:
        return find_model_id(
            package.get_models(include_templates=True), model_name
        )


def _field_index(apkg_path: Path, model_id: str, field: str) -> int:
    """Give the zero-based index of a field of the model."""
    with AnkiPackage(apkg_path) as package:
        models = package.get_models(include_templates=True)
        for index, entry in enumerate(models[model_id].get("flds", ())):
            if entry.get("name") == field:
                return index
    raise RuntimeError(f"The input APKG has no field {field!r}.")


def _write_bold(
    apkg_path: Path,
    output_path: Path,
    cards: Sequence[Any],
    result: CoreResult,
    ordered: Sequence[int],
    targets: Sequence[CoreTarget],
    profile: CoreProfile,
) -> tuple[int, int]:
    """Write the assigned target of each kept card in bold.

    :param apkg_path: Input APKG file, for the model fields.
    :param output_path: Output APKG file.
    :param cards: Input cards.
    :param result: Selection result with the assignment.
    :param ordered: Kept cards in card order.
    :param targets: Core list.
    :param profile: Language hooks.
    :raises RuntimeError: The model has no sentence field.
    :returns: The count of bolded cards and the count of the others.
    """
    if profile.highlight is None:
        return 0, len(ordered)
    by_key = {target.key: target for target in targets}
    model_id = _model_id(apkg_path, profile.model_name)
    field_index = _field_index(apkg_path, model_id, profile.sentence_field)
    bolded = 0
    with AnkiPackage(output_path) as package:
        for index in ordered:
            target = result.assigned.get(index)
            if target is None:
                continue
            marked = profile.highlight(cards[index].sentence, by_key[target].form)
            if marked != cards[index].sentence:
                package.update_note_field(
                    cards[index].note_id, field_index, marked
                )
                bolded += 1
        package._modified = True
        package.save(output_path)
    return bolded, len(ordered) - bolded


def _write_core_model(
    output_path: Path,
    cards: Sequence[Any],
    result: CoreResult,
    ordered: Sequence[int],
    targets: Sequence[CoreTarget],
    candidates: Sequence[CoreCandidate],
    profile: CoreProfile,
) -> int:
    """Set the core card templates and fill the core word field.

    :param output_path: Output APKG file.
    :param cards: Input cards.
    :param result: Selection result with the assignment.
    :param ordered: Kept cards in card order.
    :param targets: Core list.
    :param candidates: Cards that passed the filter, with the targets they hold.
    :param profile: Language hooks.
    :returns: The count of cards that hold a core word.
    """
    by_key = {target.key: target for target in targets}
    held = {candidate.index: candidate.targets for candidate in candidates}
    filled = 0
    with AnkiPackage(output_path) as package:
        models = package.get_models(include_templates=True)
        model_id = find_model_id(models, profile.model_name)
        word_index = package.add_model_field(model_id, profile.core_field)
        gloss_index = package.add_model_field(model_id, profile.core_gloss_field)
        for index in ordered:
            target = result.assigned.get(index)
            if target is None and held.get(index):
                target = min(
                    held[index], key=lambda key: (by_key[key].position, key)
                )
            if target is None:
                continue
            core = by_key[target]
            package.update_note_field(
                cards[index].note_id, word_index, core.form
            )
            package.update_note_field(
                cards[index].note_id, gloss_index, core.gloss
            )
            filled += 1
        package.update_model_template(model_id=model_id, css=profile.card_css())
        for number, (qfmt, afmt) in enumerate(profile.templates):
            package.update_model_template(
                model_id=model_id, qfmt=qfmt, afmt=afmt, template_index=number
            )
        have = len(models[model_id].get("tmpls", ()))
        for number in reversed(range(len(profile.templates), have)):
            package.remove_model_template(model_id, number)
        package._modified = True
        package.save(output_path)
    return filled


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
    name = request.name or profile.deck_name

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
    core_csv, coverage_csv, manifest_path, rank_csv = side_paths(output_path)

    write_rank_csv(cards, ordered, rank_csv)
    temporary = output_path.with_name(output_path.stem + ".tmp.apkg")
    write_deck(apkg_path, cards, ordered, temporary, name, profile.model_name)
    reorder_deck(str(temporary), str(output_path), str(rank_csv))
    temporary.unlink(missing_ok=True)
    _write_core_model(output_path, cards, result, ordered, targets, candidates, profile)
    bolded, unbolded = (
        _write_bold(apkg_path, output_path, cards, result, ordered, targets, profile)
        if request.bold_target
        else (0, 0)
    )

    rows = card_rows(
        cards, tokens, result, ordered, candidates, targets, reasons
    )
    coverage_rows = coverage_rows_of(targets, result, present, quotas)
    counts = manifest_counts(
        rows, coverage_rows, result, candidates, ordered, ranks, request.options,
        bolded=bolded, unbolded=unbolded,
    )
    counts["rejected_c3"] = sum(1 for reason in reasons if reason == "C3")
    counts["rejected_c4"] = sum(1 for reason in reasons if reason == "C4")
    counts["sense_failures"] = sum(sense_failures.values())
    counts["sense_rows"] = sum(1 for target in targets if target.checked)

    write_core_csv(rows, core_csv)
    write_coverage_csv(coverage_rows, coverage_csv)
    write_manifest(
        manifest_path,
        source={
            "input": str(apkg_path),
            "deck": name,
            "model": profile.model_name,
        },
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
            "name": name,
            "max_unchecked_sense": request.max_unchecked_sense,
            "max_unranked": request.max_unranked,
        },
        counts=counts,
        frequency={
            "path": profile.frequency_source,
            "spoken_weight": request.spoken_weight,
        },
        files={
            "core_csv": sha256_value(core_csv),
            "coverage_csv": sha256_value(coverage_csv),
            "input_apkg": sha256_value(apkg_path),
        },
        kept_order=kept_order_hash(cards, ordered),
    )

    print(f"Input:            {apkg_path}")
    print(f"Input cards:      {len(cards)}")
    print(f"Passed:           {len(passed)}")
    for rule in FILTER_KEYS:
        count = sum(1 for reason in reasons if reason == rule)
        if count:
            print(f"  Rejected {rule}:   {count}")
    print(f"Targets:          {len(targets)}")
    print(f"Quota total:      {sum(quotas.values())}")
    print(f"Cards:            {len(ordered)}")
    print(f"Covered once:     {counts['covered1']}")
    print(f"Covered twice:    {counts['covered2']}")
    print(f"Outside lemmas:   {counts['outside']}")
    print(f"Rare outside:     {counts['outside_rare']}")
    print(f"Unchecked:        {counts['unchecked']}")
    print(f"Merged S rows:    {len(merged)}")
    print()
    print(f"wrote {output_path}")
    print(f"wrote {core_csv}")
    print(f"wrote {coverage_csv}")
    print(f"wrote {manifest_path}")
    return CoreRun(
        output_path=output_path,
        core_csv=core_csv,
        coverage_csv=coverage_csv,
        manifest_path=manifest_path,
        rank_csv=rank_csv,
        cards=len(ordered),
        targets=len(targets),
        covered1=counts["covered1"],
        covered2=counts["covered2"],
    )
