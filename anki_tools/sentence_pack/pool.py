"""Candidate sentence pool (stage S2).

The pool applies filters F1 to F12 in order and reports how many sentences each
filter removes. A sentence that fails no filter becomes a candidate for the
lowest-index word it covers. With ``fallback``, a sentence in the wider window
that fails only F6 or F8 becomes a scaffold candidate for a word that has no
core candidate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

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


def check_sentence_shape(
    sentence: str,
    profile: LanguageProfile,
    min_units: int,
    max_units: int,
    *,
    final_mark: bool = True,
) -> str | None:
    """Apply filters F2 to F8 to one sentence.

    :param sentence: Prepared sentence text.
    :param profile: Language profile.
    :param min_units: Minimum unit count.
    :param max_units: Maximum unit count.
    :param final_mark: Apply filters F5 and F6 (the final-mark rules).
    :returns: The first failed filter key, or None when all filters pass.
    """
    unit_len = profile.unit_count(sentence)
    if not min_units <= unit_len <= max_units:
        return "F2"
    if not profile.script_ok(sentence):
        return "F3"
    if profile.is_repeated(sentence):
        return "F4"
    if final_mark:
        if not sentence or sentence[-1] not in profile.final_marks:
            return "F5"
        if count_marks(sentence, profile.final_marks) != 1:
            return "F6"
    if any(char in profile.quote_marks for char in sentence):
        return "F7"
    if count_marks(sentence, profile.internal_marks) > 1:
        return "F8"
    return None


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


def _stage_number(key: str) -> int:
    if key == "unknown":
        return len(FILTER_KEYS)
    return int(key[1:])


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
    :returns: The candidate list and the pool report.
    """
    min_units = profile.min_units if min_units is None else min_units
    max_units = profile.max_units if max_units is None else max_units
    forms = [profile.form_text(form) for form in word_table]
    form_set = set(forms)
    positions = profile.form_positions(word_table)
    paths = pool_paths(profile, Path(data_dir))
    links = read_links(paths["links"])
    sentences = read_sentences(paths["sentences"])
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
        sentence = profile.prepare_text(raw).strip()
        unit_len = profile.unit_count(sentence)
        shape = check_sentence_shape(sentence, profile, min_units, max_units)
        relax_shape = (
            check_sentence_shape(sentence, profile, profile.fallback_min_units, max_units)
            if fallback
            else "F2"
        )
        relax_ok = relax_shape is None or relax_shape in ("F6", "F8")
        if shape is not None and not (fallback and relax_ok):
            for key in FILTER_KEYS[: _stage_number(shape) - 1]:
                counts[key] += 1
            continue

        predicate = profile.has_predicate(sentence)
        orphan = profile.has_orphan(sentence)
        unknown_words = profile.count_unknown(sentence, known_words)
        english_words = len(english_text.split())
        english_ok = MIN_ENGLISH_WORDS <= english_words <= MAX_ENGLISH_WORDS
        dedupe = profile.dedupe_key(sentence)

        fail: int | None = _stage_number(shape) if shape is not None else None
        if fail is None:
            if not predicate:
                fail = 9
            elif orphan:
                fail = 10
            elif not english_ok:
                fail = 11
            elif dedupe in seen:
                fail = 12
            elif unknown_words > max_unknown:
                fail = 13
        passed = (fail - 1) if fail is not None else len(FILTER_KEYS)
        for key in FILTER_KEYS[:passed]:
            counts[key] += 1

        hits = tuple(sorted(profile.hits(sentence, form_set)))
        if fail is None and hits:
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
            fallback
            and relax_ok
            and predicate
            and not orphan
            and english_ok
            and unknown_words <= max_unknown
            and hits
            and dedupe not in seen
            and dedupe not in relax_seen
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
