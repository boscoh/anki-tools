"""Candidate sentence pool (stage S2).

The pool applies filters F1 to F12, the unknown-token budget, and the English
sense test in order. The guard tables give the order, and each report count is
the number of sentences that pass the guard. A sentence that fails no filter
becomes a candidate for the lowest-index word it covers. With ``fallback``, a
sentence in the wider window that fails only F6 or F8 becomes a scaffold
candidate for a word that has no core candidate.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

from anki_tools.sentence_pack.english_sense import holds_sense
from anki_tools.sentence_pack.profile import LanguageProfile, WordRow
from anki_tools.sentence_pack.tatoeba import (
    read_audio,
    read_authors,
    read_english_sentences,
    read_links,
    read_sentences,
)
from wordfreq import zipf_frequency

FILTER_KEYS = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "unknown",
    "sense",
]
FILTER_NAMES = {
    "F1": "english link",
    "F2": "length window",
    "F3": "script",
    "F4": "no repeated pattern",
    "F5": "final mark",
    "F6": "one final mark",
    "F7": "no quotation mark",
    "F8": "at most one internal mark",
    "F9": "predicate",
    "F10": "no orphan conjunction",
    "F11": "english word count",
    "F12": "unique text",
    "unknown": "unknown token budget",
    "sense": "english sense",
}
MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12


@dataclass(frozen=True)
class SentenceFacts:
    """Measured properties of one accepted Tatoeba sentence.

    :param sentence: Prepared sentence text.
    :param text_original: Sentence text before the language conversion.
    :param tatoeba_id: Tatoeba sentence id.
    :param tatoeba_author: Tatoeba author name, empty when unknown.
    :param english: Paired English translation.
    :param english_id: Tatoeba English sentence id.
    :param audio_id: Native Tatoeba audio id, 0 when absent.
    :param audio_license: License of the native recording, empty when absent.
    :param unknown_words: Token count outside the known set.
    :param unit_len: Word or character count of the sentence.
    :param zipf_mean: Mean Zipf value of the sentence tokens.
    :param hits: Forms that appear exactly one time.
    """

    sentence: str
    text_original: str
    tatoeba_id: int
    tatoeba_author: str
    english: str
    english_id: int
    audio_id: int
    audio_license: str
    unknown_words: int
    unit_len: int
    zipf_mean: float
    hits: tuple[str, ...]


def filter_names(profile: LanguageProfile) -> dict[str, str]:
    """Build the filter names of one language.

    :param profile: Language profile.
    :returns: Mapping from filter key to a short name.
    """
    names = dict(FILTER_NAMES)
    names["F2"] = f"{profile.unit_name} length window"
    return names


def count_marks(sentence: str, marks: str) -> int:
    """Count the characters of a sentence that belong to a mark set.

    :param sentence: Sentence text.
    :param marks: Characters to count.
    :returns: Number of matching characters.
    """
    return sum(1 for char in sentence if char in marks)


@dataclass(frozen=True)
class ShapeInput:
    """Measured values of the F2 to F8 shape guards.

    :param sentence: Prepared sentence text.
    :param profile: Language profile.
    :param unit_len: Word or character count of the sentence.
    :param min_units: Minimum unit count of the window.
    :param max_units: Maximum unit count of the window.
    :param final_mark: Apply filters F5 and F6 (the final-mark rules).
    """

    sentence: str
    profile: LanguageProfile
    unit_len: int
    min_units: int
    max_units: int
    final_mark: bool = True


def _passes_length(item: ShapeInput) -> bool:
    return item.min_units <= item.unit_len <= item.max_units


def _passes_script(item: ShapeInput) -> bool:
    return item.profile.script_ok(item.sentence)


def _passes_repeat(item: ShapeInput) -> bool:
    return not item.profile.is_repeated(item.sentence)


def _passes_final_mark(item: ShapeInput) -> bool:
    return not item.final_mark or (
        bool(item.sentence) and item.sentence[-1] in item.profile.final_marks
    )


def _passes_one_final_mark(item: ShapeInput) -> bool:
    return not item.final_mark or (
        count_marks(item.sentence, item.profile.final_marks) == 1
    )


def _passes_quote(item: ShapeInput) -> bool:
    return not any(char in item.profile.quote_marks for char in item.sentence)


def _passes_internal_marks(item: ShapeInput) -> bool:
    return count_marks(item.sentence, item.profile.internal_marks) <= 1


SHAPE_GUARDS: tuple[tuple[str, Callable[[ShapeInput], bool]], ...] = (
    ("F2", _passes_length),
    ("F3", _passes_script),
    ("F4", _passes_repeat),
    ("F5", _passes_final_mark),
    ("F6", _passes_one_final_mark),
    ("F7", _passes_quote),
    ("F8", _passes_internal_marks),
)
SCAFFOLD_SHAPE_FAILURES = frozenset(("F6", "F8"))
SCAFFOLD_SHAPE_KEYS = frozenset(
    key for key, _ in SHAPE_GUARDS if key not in SCAFFOLD_SHAPE_FAILURES
)


def shape_failure(
    sentence: str,
    profile: LanguageProfile,
    min_units: int,
    max_units: int,
    *,
    final_mark: bool = True,
    keys: Collection[str] = (),
) -> str | None:
    """Give the first failed guard of filters F2 to F8.

    :param sentence: Prepared sentence text.
    :param profile: Language profile.
    :param min_units: Minimum unit count of the window.
    :param max_units: Maximum unit count of the window.
    :param final_mark: Apply filters F5 and F6 (the final-mark rules).
    :param keys: Guard keys to apply, empty for every key.
    :returns: The first failed filter key, or None when the guards pass.
    """
    data = ShapeInput(
        sentence=sentence,
        profile=profile,
        unit_len=profile.unit_count(sentence),
        min_units=min_units,
        max_units=max_units,
        final_mark=final_mark,
    )
    for key, test in SHAPE_GUARDS:
        if (not keys or key in keys) and not test(data):
            return key
    return None




@dataclass(frozen=True)
class FactInput:
    """Measured values of the F9 to sense guards.

    :param predicate: True when the sentence has a predicate tag.
    :param orphan: True when the sentence starts with an orphan conjunction.
    :param english_words: Word count of the English translation.
    :param english_bounds: Accepted English word-count window.
    :param duplicate: True when the dedupe key is already kept.
    :param unknown_words: Token count outside the known set.
    :param max_unknown: Unknown token budget.
    :param hits: Forms that the form filter keeps.
    :param sense_hits: Forms whose paired English holds the card sense.
    """

    predicate: bool
    orphan: bool
    english_words: int
    english_bounds: tuple[int, int]
    duplicate: bool
    unknown_words: int
    max_unknown: int
    hits: tuple[str, ...]
    sense_hits: tuple[str, ...]


def _passes_predicate(item: FactInput) -> bool:
    return item.predicate


def _passes_orphan(item: FactInput) -> bool:
    return not item.orphan


def _passes_english(item: FactInput) -> bool:
    low, high = item.english_bounds
    return low <= item.english_words <= high


def _passes_unique(item: FactInput) -> bool:
    return not item.duplicate


def _passes_unknown(item: FactInput) -> bool:
    return item.unknown_words <= item.max_unknown


def _passes_sense(item: FactInput) -> bool:
    return not item.hits or bool(item.sense_hits)


FACT_GUARDS: tuple[tuple[str, Callable[[FactInput], bool]], ...] = (
    ("F9", _passes_predicate),
    ("F10", _passes_orphan),
    ("F11", _passes_english),
    ("F12", _passes_unique),
    ("unknown", _passes_unknown),
    ("sense", _passes_sense),
)
SCAFFOLD_SKIP_GUARDS = frozenset(("sense",))


def fact_failure(data: FactInput, *, skip: Collection[str] = ()) -> str | None:
    """Give the first failed guard of filters F9 to sense.

    :param data: Measured values of the sentence.
    :param skip: Guard keys to leave out.
    :returns: The first failed guard key, or None when the guards pass.
    """
    for key, test in FACT_GUARDS:
        if key not in skip and not test(data):
            return key
    return None


def exceeds_max_units(sentence: str, max_units: int) -> bool:
    """Check without a model that a sentence is longer than the length window.

    Each whitespace chunk with a letter or digit gives at least one word or
    character unit. A count above the maximum guarantees a filter F2 failure.

    :param sentence: Prepared sentence text.
    :param max_units: Maximum unit count of the window.
    :returns: True when the sentence has more than ``max_units`` units.
    """
    chunks = 0
    for chunk in sentence.split():
        if any(char.isalnum() for char in chunk):
            chunks += 1
            if chunks > max_units:
                return True
    return False


def record_pass(counts: dict[str, int], failure: str | None) -> None:
    """Count one sentence as a pass of every guard before its first failure.

    :param counts: Filter counts of the pool report.
    :param failure: First failed guard key, or None when the guards pass.
    """
    stop = len(FILTER_KEYS) if failure is None else FILTER_KEYS.index(failure)
    for key in FILTER_KEYS[:stop]:
        counts[key] += 1


def zipf_value(word: str, lang: str) -> float:
    """Read the cached `wordfreq` Zipf value of one word.

    :param word: Word to look up.
    :param lang: `wordfreq` language code.
    :returns: Zipf value between 0 and 8.
    """
    return _zipf_cached(word, lang)


@lru_cache(maxsize=200_000)
def _zipf_cached(word: str, lang: str) -> float:
    return zipf_frequency(word, lang)


def mean_zipf(words: Sequence[str], lang: str) -> float:
    """Compute the mean Zipf value of a token list.

    :param words: Words of the sentence.
    :param lang: `wordfreq` language code.
    :returns: Mean value, or 0.0 for an empty list.
    """
    if not words:
        return 0.0
    return sum(zipf_value(word, lang) for word in words) / len(words)


def pair_english(
    english_ids: Sequence[int], english: dict[int, str]
) -> tuple[int, str] | None:
    """Pick the English translation with the fewest words, then the lowest id.

    :param english_ids: Linked English sentence ids.
    :param english: Mapping from English sentence id to its text.
    :returns: ``(english_id, text)`` or None when no link has a text.
    """
    best: tuple[int, int, str] | None = None
    for english_id in english_ids:
        text = english.get(english_id)
        if not text:
            continue
        words = len(text.split())
        if best is None or (words, english_id) < (best[0], best[1]):
            best = (words, english_id, text)
    if best is None:
        return None
    return best[1], best[2]


def pool_paths(profile: LanguageProfile, data_dir: Path) -> dict[str, Path]:
    """Give the Tatoeba input files of one language.

    :param profile: Language profile.
    :param data_dir: Cache directory of the Tatoeba TSV files.
    :returns: Mapping from input kind to its path.
    """
    lang = profile.tatoeba_lang
    paths = {
        "sentences": data_dir / f"{lang}_sentences.tsv",
        "links": data_dir / f"{lang}-{profile.english_lang}_links.tsv",
        "detailed": data_dir / f"{lang}_sentences_detailed.tsv",
        "english": data_dir / f"{profile.english_lang}_sentences.tsv",
    }
    audio = data_dir / f"{lang}_sentences_with_audio.tsv"
    if profile.native_audio:
        paths["audio"] = audio
    return paths


FRAME_FIELDS = ("target", "sentence", "english")


def default_frames_path(
    profile: LanguageProfile, vocab_dir: Path = Path("vocab")
) -> Path:
    """Give the manual frame-sentence file of one language.

    :param profile: Language profile.
    :param vocab_dir: Directory of the vocabulary files.
    :returns: Path of the frame file, which may not exist.
    """
    return vocab_dir / f"{profile.name}_frames.csv"


def read_frames(path: Path) -> list[dict[str, str]]:
    """Read the manual frame sentences of one language.

    The file has the columns target, sentence, and english. A missing file
    gives an empty list, so a language needs no frame file. A row with an
    empty sentence is skipped.

    :param path: Path of the frame CSV file.
    :returns: The rows, in file order.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            row
            for row in reader
            if (row.get("sentence") or "").strip()
        ]


def build_candidate_pool(
    word_table: Sequence[WordRow],
    profile: LanguageProfile,
    *,
    data_dir: Path = Path("data/tatoeba"),
    min_units: int | None = None,
    max_units: int | None = None,
    max_unknown: int = 2,
    fallback: bool = False,
    known_words: Collection[str] = (),
    make_candidate: Callable[..., Any],
    frames_path: Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Build the S2 candidate pool from the Tatoeba exports.

    The function also sets ``candidate_count`` and ``tier`` on the word table.

    :param word_table: Reviewed word table of the language.
    :param profile: Language profile.
    :param data_dir: Directory with the Tatoeba TSV files.
    :param min_units: Minimum unit count for a core candidate.
    :param max_units: Maximum unit count.
    :param max_unknown: Unknown token budget per sentence.
    :param fallback: Add scaffold candidates for forms with no core candidate.
    :param known_words: Known word set of the unknown-token filter.
    :param make_candidate: Callable that builds a card object from the facts.
    :param frames_path: Optional CSV file of manual frame sentences. A frame
        passes the same guards as a Tatoeba sentence.
    :returns: The candidate list and the pool report.
    """
    min_units = profile.min_units if min_units is None else min_units
    max_units = profile.max_units if max_units is None else max_units
    forms = [profile.form_text(form) for form in word_table]
    form_set = set(forms)
    positions = profile.form_positions(word_table)
    gloss_by_form = {profile.form_text(form): form.english for form in word_table}
    paths = pool_paths(profile, Path(data_dir))
    links = read_links(paths["links"])
    sentences = read_sentences(paths["sentences"])
    prepared = {
        sentence_id: profile.prepare_text(sentences[sentence_id]).strip()
        for sentence_id in links
        if sentence_id in sentences
    }
    if profile.warm_tokens is not None:
        profile.warm_tokens(prepared.values())
    wanted_english = {
        english_id for english_ids in links.values() for english_id in english_ids
    }
    english = read_english_sentences(paths["english"], wanted_english)
    authors = read_authors(paths["detailed"])
    audio = read_audio(paths["audio"]) if "audio" in paths else {}

    counts: dict[str, int] = dict.fromkeys(FILTER_KEYS, 0)
    core_facts: list[SentenceFacts] = []
    relax_facts: list[SentenceFacts] = []
    seen: set[str] = set()
    relax_seen: set[str] = set()
    total = 0
    for sentence_id in sorted(links):
        raw = sentences.get(sentence_id)
        if raw is None:
            continue
        total += 1
        pair = pair_english(links[sentence_id], english)
        if pair is None:
            continue
        english_id, english_text = pair
        sentence = prepared[sentence_id]
        if exceeds_max_units(sentence, max_units):
            record_pass(counts, "F2")
            continue
        unit_len = profile.unit_count(sentence)
        shape = shape_failure(sentence, profile, min_units, max_units)
        relax_shape = (
            shape_failure(
                sentence, profile, profile.fallback_min_units, max_units
            )
            if fallback
            else None
        )
        relax_ok = fallback and (
            relax_shape is None or relax_shape in SCAFFOLD_SHAPE_FAILURES
        )
        if shape is not None and not relax_ok:
            record_pass(counts, shape)
            continue

        predicate = profile.has_predicate(sentence)
        orphan = profile.has_orphan(sentence)
        unknown_words = profile.count_unknown(sentence, known_words)
        english_words = len(english_text.split())
        dedupe = profile.dedupe_key(sentence)

        hits = tuple(sorted(profile.hits(sentence, form_set)))
        if profile.sense_filter is not None:
            hits = tuple(
                form
                for form in hits
                if not profile.sense_filter(form, english_text)
            )
        sense_hits = tuple(
            form
            for form in hits
            if holds_sense(gloss_by_form.get(form, ""), english_text)
        )

        measured = FactInput(
            predicate=predicate,
            orphan=orphan,
            english_words=english_words,
            english_bounds=(MIN_ENGLISH_WORDS, MAX_ENGLISH_WORDS),
            duplicate=dedupe in seen,
            unknown_words=unknown_words,
            max_unknown=max_unknown,
            hits=hits,
            sense_hits=sense_hits,
        )
        failure = shape if shape is not None else fact_failure(measured)
        record_pass(counts, failure)
        hits = sense_hits
        if failure is None and hits:
            seen.add(dedupe)
            audio_id, audio_license = audio.get(sentence_id, (0, ""))
            core_facts.append(
                SentenceFacts(
                    sentence=sentence,
                    text_original=raw,
                    tatoeba_id=sentence_id,
                    tatoeba_author=authors.get(sentence_id, ""),
                    english=english_text,
                    english_id=english_id,
                    audio_id=audio_id,
                    audio_license=audio_license,
                    unknown_words=unknown_words,
                    unit_len=unit_len,
                    zipf_mean=mean_zipf(profile.zipf_words(sentence), profile.wordfreq_lang),
                    hits=hits,
                )
            )
        elif (
            relax_ok
            and hits
            and fact_failure(
                replace(
                    measured,
                    duplicate=measured.duplicate or dedupe in relax_seen,
                ),
                skip=SCAFFOLD_SKIP_GUARDS,
            )
            is None
        ):
            relax_seen.add(dedupe)
            audio_id, audio_license = audio.get(sentence_id, (0, ""))
            relax_facts.append(
                SentenceFacts(
                    sentence=sentence,
                    text_original=raw,
                    tatoeba_id=sentence_id,
                    tatoeba_author=authors.get(sentence_id, ""),
                    english=english_text,
                    english_id=english_id,
                    audio_id=audio_id,
                    audio_license=audio_license,
                    unknown_words=unknown_words,
                    unit_len=unit_len,
                    zipf_mean=mean_zipf(profile.zipf_words(sentence), profile.wordfreq_lang),
                    hits=hits,
                )
            )

    frame_counts: dict[str, int] = dict.fromkeys(FILTER_KEYS, 0)
    frames = read_frames(frames_path) if frames_path is not None else []
    frame_added = 0
    for row in frames:
        raw = row["sentence"].strip()
        target_form = (row.get("target") or "").strip()
        frame_english = (row.get("english") or "").strip()
        sentence = profile.prepare_text(raw).strip()
        shape = shape_failure(sentence, profile, min_units, max_units)
        if shape is not None:
            record_pass(frame_counts, shape)
            continue
        hits = tuple(sorted(profile.hits(sentence, form_set)))
        if target_form and target_form not in hits:
            record_pass(frame_counts, "sense")
            continue
        dedupe = profile.dedupe_key(sentence)
        measured = FactInput(
            predicate=profile.has_predicate(sentence),
            orphan=profile.has_orphan(sentence),
            english_words=len(frame_english.split()),
            english_bounds=(MIN_ENGLISH_WORDS, MAX_ENGLISH_WORDS),
            duplicate=dedupe in seen,
            unknown_words=profile.count_unknown(sentence, known_words),
            max_unknown=max_unknown,
            hits=hits,
            sense_hits=hits,
        )
        failure = fact_failure(measured)
        record_pass(frame_counts, failure)
        if failure is not None or not hits:
            continue
        seen.add(dedupe)
        frame_added += 1
        core_facts.append(
            SentenceFacts(
                sentence=sentence,
                text_original=raw,
                tatoeba_id=0,
                tatoeba_author="manual",
                english=frame_english,
                english_id=0,
                audio_id=0,
                audio_license="",
                unknown_words=measured.unknown_words,
                unit_len=profile.unit_count(sentence),
                zipf_mean=mean_zipf(
                    profile.zipf_words(sentence), profile.wordfreq_lang
                ),
                hits=hits,
            )
        )

    core_covered: Counter[str] = Counter()
    pool: list[Any] = []
    for facts in core_facts:
        target = min(facts.hits, key=lambda form: positions[form])
        pool.append(
            make_candidate(facts, target, "core", frozenset(facts.hits))
        )
        core_covered.update(facts.hits)

    uncovered_forms = [form for form in forms if form not in core_covered]
    scaffold_texts: set[str] = set()
    scaffold_count = 0
    if fallback and uncovered_forms:
        uncovered_set = set(uncovered_forms)
        core_texts = {facts.sentence for facts in core_facts}
        for facts in relax_facts:
            if facts.sentence in core_texts:
                continue
            targets = [form for form in facts.hits if form in uncovered_set]
            if not targets:
                continue
            scaffold_texts.add(facts.sentence)
            for form in targets:
                scaffold_count += 1
                pool.append(
                    make_candidate(facts, form, "scaffold", frozenset((form,)))
                )

    candidates_per_form: Counter[str] = Counter()
    for candidate in pool:
        candidates_per_form.update(candidate.covered)
    for form, name in zip(word_table, forms):
        form.candidate_count = candidates_per_form[name]
        if name not in core_covered and form.candidate_count:
            form.tier = "scaffold"

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "min_units": min_units,
        "max_units": max_units,
        "max_unknown": max_unknown,
        "fallback": fallback,
        "english_ids": len(wanted_english),
        "total_linked": total,
        "counts": counts,
        "filter_names": filter_names(profile),
        "pool_candidates": len(pool),
        "core_sentences": len(core_facts),
        "frames_path": str(frames_path) if frames_path is not None else "",
        "frame_rows": len(frames),
        "frame_sentences": frame_added,
        "frame_counts": frame_counts,
        "scaffold_sentences": len(scaffold_texts),
        "scaffold_candidates": scaffold_count,
        "forms_with_core_candidate": len(core_covered),
        "forms_without_core_candidate": len(uncovered_forms),
    }
    return pool, report


def build_pack_report(
    pool_report: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    """Merge the pool report and the selection report.

    :param pool_report: Report from :func:`build_candidate_pool`.
    :param selection: Report from a language selection function.
    :returns: One mapping with the counts, the cards, and the coverage.
    """
    report = dict(pool_report)
    report["cards"] = selection["cards"]
    report["per_word"] = selection["per_word"]
    report["max_cards"] = selection["max_cards"]
    report["tier_counts"] = selection["tier_counts"]
    report["form_coverage"] = selection["forms"]
    report["coverage"] = selection["core_coverage"]
    report["uncovered"] = selection["uncovered"]
    report["zero_candidates"] = selection["zero_candidates"]
    report["cut_by_cap"] = selection["cut_by_cap"]
    return report


def write_pool_report(report: dict[str, Any], path: Path) -> Path:
    """Write the pool report as JSON.

    :param report: Mapping from :func:`build_pack_report`.
    :param path: Output JSON file.
    :returns: The written path.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def print_sentence_report(report: dict[str, Any]) -> None:
    """Print the filter counts, the card count, and the coverage.

    :param report: Mapping from :func:`build_pack_report`.
    """
    print(f"linked sentences: {report['total_linked']}")
    names = report.get("filter_names", FILTER_NAMES)
    for key in FILTER_KEYS:
        print(f"  {key} {names[key]}: {report['counts'][key]}")
    print(
        f"pool candidates: {report['pool_candidates']} "
        f"core={report['core_sentences']} "
        f"scaffold={report['scaffold_candidates']}"
    )
    print(
        f"cards: {report['cards']} "
        f"core={report['tier_counts'].get('core', 0)} "
        f"scaffold={report['tier_counts'].get('scaffold', 0)}"
    )
    forms = report["form_coverage"]
    coverage = report["coverage"]
    print(
        f"forms with card: {forms['with_card']}/{forms['total']} "
        f"({forms['ratio'] * 100:.1f}%)"
    )
    print(
        f"core coverage: {coverage['with_card']}/{coverage['concepts']} "
        f"({coverage['ratio'] * 100:.1f}%)"
    )
    uncovered = report["uncovered"]
    print(f"uncovered words: {len(uncovered)}")
    if uncovered:
        preview = ", ".join(item["target"] for item in uncovered)
        print(f"  {preview}")
