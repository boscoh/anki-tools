"""Sentence filter and diversity selection for Cantonese decks.

The filter applies rules Y1 to Y7. The selection is a greedy cover of the most
frequent words of the blended spoken and written list. It limits the uses of
one word, then orders the kept sentences from easy to hard.

See ``specs/yue-sentence-selection.md`` sections 7 and 8.
"""

from __future__ import annotations

import csv
import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.languages.cantonese.profile import CANTONESE
from anki_tools.ranking.rank import (
    Sentence,
    char_similarity_zh_same_len,
    get_chinese_chars,
)
from anki_tools.languages.cantonese.frequency import (
    VOCAB_DIR,
    Tokenizer,
    YueFrequency,
    han_tokens,
)

NAMES_PATH = VOCAB_DIR / "cantonese_names.tsv"
EXEMPT_TOP = 100
UNRANKED = 1_000_000
ORDERS = ("coverage", "score")
RULES = ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8")
FINAL_MARK_WARNING_SHARE = 0.2
SHAPE_RULES = {
    "F2": "Y1",
    "F3": "Y2",
    "F4": "Y3",
    "F5": "Y4",
    "F6": "Y4",
    "F7": "Y5",
    "F8": "Y6",
}
SELECT_REASONS = ("similar", "no_gain", "use_cap", "cap")

SELECT_COLUMNS = ("sentence", "kept", "reason", "targets", "gain")
COVERAGE_COLUMNS = ("word", "rank", "count", "status")


@dataclass(frozen=True)
class SelectionOptions:
    """Settings of the filter and the selection.

    :param max_sentences: Stop the selection at this count. None stops at zero gain.
    :param per_word: Cards wanted for each target word.
    :param max_uses: Most kept sentences for one word outside the top 100. 0 is no limit.
    :param top_words: Size of the target set.
    :param order: ``coverage`` (learning order) or ``score`` (ranker order).
    :param min_han: Minimum Han characters (rule Y1).
    :param max_han: Maximum Han characters (rule Y1).
    :param max_unknown: Unknown tokens allowed in a sentence (rule Y7).
    :param known_top: Size of the known set (rule Y7).
    :param similarity_threshold: Character Jaccard threshold for duplicates.
    :param final_mark: Apply rule Y4.
    :param names: Foreign names for rule Y8. A name is a substring of the sentence.
    """

    max_sentences: int | None = None
    per_word: int = 2
    max_uses: int = 8
    top_words: int = 1500
    order: str = "coverage"
    min_han: int = 3
    max_han: int = 12
    max_unknown: int = 2
    known_top: int = 2000
    similarity_threshold: float = 0.67
    final_mark: bool = True
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {self.order!r}")
        if self.per_word < 1:
            raise ValueError("per_word must be at least 1")
        if self.max_uses and self.max_uses < self.per_word:
            raise ValueError("max_uses must be 0 or at least per_word")
        if self.max_sentences is not None and self.max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")


@dataclass
class Candidate:
    """A sentence that passed the rules.

    :param sentence: The ranked sentence.
    :param tokens: Han tokens, with variant characters mapped to the standard form.
    :param han_len: Number of Han characters.
    :param gain: Gain at the last evaluation.
    """

    sentence: Sentence
    tokens: list[str]
    han_len: int
    gain: float = 0.0

    @property
    def words(self) -> set[str]:
        return set(self.tokens)


@dataclass
class SelectionResult:
    """Output of :func:`greedy_select`.

    :param kept: Candidates in pick order.
    :param reasons: Candidate index to the reason it was not kept.
    :param word_counts: Kept sentences that contain each word.
    :param targets: Target words in rank order.
    """

    kept: list[Candidate]
    reasons: dict[int, str]
    word_counts: Counter[str]
    targets: list[str]


@dataclass
class SelectionReport:
    """Everything that ``fix yue`` writes after a selection.

    :param ordered: Kept sentences in output order, then the others in ranker order.
    :param select_rows: One row per input sentence.
    :param coverage_rows: One row per target word.
    :param summary: Measures for the printed summary.
    """

    ordered: list[Sentence]
    select_rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def read_names(path: Path = NAMES_PATH) -> tuple[str, ...]:
    """Read the foreign names of rule Y8.

    The file has a header line and one name per line, in the first column.

    :param path: TSV file with a ``name`` column.
    :returns: Names, in file order. An absent file gives an empty tuple.
    """
    if not path.exists():
        return ()
    names = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        name = line.split("\t")[0].strip()
        if number and name:
            names.append(name)
    return tuple(names)


def check_sentence(
    text: str,
    tokens: list[str],
    ranks: dict[str, int],
    options: SelectionOptions,
) -> str:
    """Apply rules Y1 to Y8 in order.

    Rules Y1 to Y6 reuse :func:`anki_tools.sentence_pack.guards.check_sentence_shape`.
    Rule Y8 matches the substrings of ``options.names``.

    :param text: Sentence text.
    :param tokens: Han tokens of the sentence.
    :param ranks: Word to frequency rank, variants mapped.
    :param options: Filter settings.
    :returns: The first failed rule, or an empty string when the sentence passes.
    """
    stripped = text.strip()
    key = check_sentence_shape(
        stripped,
        CANTONESE,
        options.min_han,
        options.max_han,
        final_mark=options.final_mark,
    )
    if key is not None:
        return SHAPE_RULES[key]
    unknown = sum(1 for t in tokens if ranks.get(t, math.inf) > options.known_top)
    if unknown > options.max_unknown:
        return "Y7"
    if any(name in stripped for name in options.names):
        return "Y8"
    return ""


def _hardest_rank(tokens: list[str], ranks: dict[str, int]) -> int:
    return max((ranks.get(token, UNRANKED) for token in tokens), default=UNRANKED)


def greedy_select(
    candidates: list[Candidate],
    ranks: dict[str, int],
    options: SelectionOptions,
) -> SelectionResult:
    """Pick sentences with the greedy cover of section 8.2.

    :param candidates: Sentences that passed the rules.
    :param ranks: Word to frequency rank, variants mapped.
    :param options: Selection settings.
    :returns: Kept candidates in pick order, and the reason for each other candidate.
    """
    present = {word for c in candidates for word in c.words if word in ranks}
    targets = sorted(present, key=lambda word: (ranks[word], word))[: options.top_words]
    target_set = set(targets)
    weight = {word: 1 / math.log2(ranks[word] + 1) for word in targets}
    chosen: Counter[str] = Counter()
    uses: Counter[str] = Counter()

    def gain(candidate: Candidate) -> float:
        return sum(
            weight[word] / (chosen[word] + 1)
            for word in candidate.words & target_set
            if chosen[word] < options.per_word
        )

    def at_limit(candidate: Candidate) -> bool:
        if not options.max_uses:
            return False
        return any(
            ranks.get(word, math.inf) > EXEMPT_TOP and uses[word] >= options.max_uses
            for word in candidate.words
        )

    def key(index: int, value: float) -> tuple[float, float, int, int, int]:
        c = candidates[index]
        return (
            -value,
            c.sentence.complexity_score,
            c.han_len,
            c.sentence.original_order,
            index,
        )

    heap = [key(i, gain(c)) for i, c in enumerate(candidates)]
    heapq.heapify(heap)
    kept: list[Candidate] = []
    reasons: dict[int, str] = {}
    by_size: dict[int, list[str]] = {}
    token_sets: set[frozenset[str]] = set()

    def too_similar(candidate: Candidate) -> bool:
        if frozenset(candidate.words) in token_sets:
            return True
        text = candidate.sentence.text
        return any(
            char_similarity_zh_same_len(text, other) >= options.similarity_threshold
            for other in by_size.get(len(set(get_chinese_chars(text))), ())
        )

    limit = options.max_sentences
    while heap and (limit is None or len(kept) < limit):
        stored, *_, index = heapq.heappop(heap)
        candidate = candidates[index]
        value = gain(candidate)
        candidate.gain = value
        if value <= 0:
            reasons[index] = "no_gain"
            continue
        if abs(value + stored) > 1e-12:
            heapq.heappush(heap, key(index, value))
            continue
        if too_similar(candidate):
            reasons[index] = "similar"
            continue
        if at_limit(candidate):
            reasons[index] = "use_cap"
            continue
        token_sets.add(frozenset(candidate.words))
        text = candidate.sentence.text
        by_size.setdefault(len(set(get_chinese_chars(text))), []).append(text)
        for word in candidate.words:
            uses[word] += 1
            if word in target_set:
                chosen[word] += 1
        kept.append(candidate)

    for *_, index in heap:
        candidate = candidates[index]
        candidate.gain = gain(candidate)
        reasons[index] = "cap" if candidate.gain > 0 else "no_gain"
    return SelectionResult(kept, reasons, uses, targets)


def _reach(freq: YueFrequency, counts: Counter[str], top: int) -> int:
    return sum(1 for row in freq.rows[:top] if counts[row.word])


def spoken_mass(freq: YueFrequency, candidates: list[Candidate]) -> float:
    """Share of the Cifu spoken tokens that the kept words cover.

    :param freq: The ranked frequency list.
    :param candidates: Kept candidates.
    :returns: A value from 0 to 100.
    """
    total = sum(row.spoken_count for row in freq.rows)
    if not total:
        return 0.0
    words = {word for candidate in candidates for word in candidate.words}
    covered = sum(row.spoken_count for row in freq.rows if row.word in words)
    return 100 * covered / total


def select_yue_sentences(
    ranked: list[Sentence],
    dropped: list[Sentence],
    freq: YueFrequency,
    tokenizer: Tokenizer,
    options: SelectionOptions,
) -> SelectionReport:
    """Filter and select sentences, then set the selection fields on each.

    :param ranked: Sentences from :func:`anki_tools.ranking.rank.rank_sentences_yue`, best first.
    :param dropped: Sentences that the ranker removed as repeats (reason Y3).
    :param freq: The ranked frequency list.
    :param tokenizer: Cantonese tokenizer.
    :param options: Filter and selection settings.
    :returns: The ordered sentences and the report rows.
    """
    ranks = freq.ranks
    candidates: list[Candidate] = []
    for sentence in ranked:
        tokens = [freq.variants.get(t, t) for t in han_tokens(tokenizer, sentence.text)]
        reason = check_sentence(sentence.text, tokens, ranks, options)
        sentence.reject_reason = reason
        sentence.kept = False
        if not reason:
            han_len = len(get_chinese_chars(sentence.text))
            candidates.append(Candidate(sentence, tokens, han_len))
    for sentence in dropped:
        sentence.reject_reason = "Y3"
        sentence.kept = False

    result = greedy_select(candidates, ranks, options)
    for index, reason in result.reasons.items():
        candidates[index].sentence.reject_reason = reason

    for candidate in result.kept:
        candidate.sentence.kept = True
        candidate.sentence.reject_reason = ""
        candidate.sentence.hardest_rank = _hardest_rank(candidate.tokens, ranks)
    if options.order == "coverage":
        kept_sorted = sorted(
            result.kept,
            key=lambda c: (
                c.sentence.hardest_rank,
                c.sentence.complexity_score,
                c.han_len,
                c.sentence.original_order,
            ),
        )
    else:
        position = {id(s): i for i, s in enumerate(ranked)}
        kept_sorted = sorted(result.kept, key=lambda c: position[id(c.sentence)])
    kept_ids = {id(c.sentence) for c in kept_sorted}
    ordered = [c.sentence for c in kept_sorted] + [s for s in ranked if id(s) not in kept_ids]

    candidate_of = {id(c.sentence): c for c in candidates}
    target_set = set(result.targets)
    select_rows = []
    for sentence in sorted(ranked + dropped, key=lambda s: s.original_order):
        candidate = candidate_of.get(id(sentence))
        targets = (
            sorted(candidate.words & target_set, key=lambda word: ranks[word])
            if candidate
            else []
        )
        select_rows.append({
            "sentence": sentence.text,
            "kept": int(bool(sentence.kept)),
            "reason": sentence.reject_reason,
            "targets": " ".join(targets),
            "gain": f"{candidate.gain:.3f}" if candidate else "",
        })

    coverage_rows = []
    in_pool = target_set
    for word in result.targets:
        count = result.word_counts[word]
        status = "covered" if count >= options.per_word else "partial"
        coverage_rows.append(
            {"word": word, "rank": ranks[word], "count": count, "status": status}
        )
    for row in freq.rows[: options.top_words]:
        if row.word not in in_pool:
            coverage_rows.append(
                {"word": row.word, "rank": row.rank, "count": 0, "status": "absent"}
            )
    coverage_rows.sort(key=lambda r: r["rank"])

    rejected = Counter(s.reject_reason for s in ranked + dropped if s.reject_reason in RULES)
    removed = Counter(
        s.reject_reason for s in ranked + dropped if s.reject_reason in SELECT_REASONS
    )
    unlimited = [
        count for word, count in result.word_counts.items() if ranks.get(word, math.inf) > EXEMPT_TOP
    ]
    summary = {
        "sentences_in": len(ranked) + len(dropped),
        "rejected": dict(rejected),
        "removed": dict(removed),
        "kept": len(result.kept),
        "top500": _reach(freq, result.word_counts, 500),
        "top1000": _reach(freq, result.word_counts, 1000),
        "top1500": _reach(freq, result.word_counts, 1500),
        "spoken_mass": spoken_mass(freq, result.kept),
        "max_uses": max(unlimited, default=0),
        "final_mark_share": rejected["Y4"] / max(len(ranked) + len(dropped), 1),
    }
    return SelectionReport(ordered, select_rows, coverage_rows, summary)


def _write_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    try:
        handle = open(path, "w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {path}.") from exc
    with handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_select_csv(report: SelectionReport, path: Path) -> None:
    """Write one row per input sentence: sentence, kept, reason, targets, gain.

    :param report: Selection report.
    :param path: Output file.
    """
    _write_rows(path, SELECT_COLUMNS, report.select_rows)


def write_coverage_csv(report: SelectionReport, path: Path) -> None:
    """Write one row per target word: word, rank, count, status.

    :param report: Selection report.
    :param path: Output file.
    """
    _write_rows(path, COVERAGE_COLUMNS, report.coverage_rows)


def print_selection_summary(report: SelectionReport, options: SelectionOptions) -> None:
    """Print the measures of a selection.

    :param report: Selection report.
    :param options: The settings of the run.
    """
    s = report.summary
    rejected = ", ".join(f"{rule} {s['rejected'].get(rule, 0)}" for rule in RULES)
    removed = ", ".join(f"{k} {v}" for k, v in sorted(s["removed"].items())) or "none"
    print(f"  Sentences in:        {s['sentences_in']}")
    print(f"  Rejected by rule:    {rejected}")
    print(f"  Removed by select:   {removed}")
    print(f"  Kept:                {s['kept']}")
    for top in (500, 1000, 1500):
        count = s[f"top{top}"]
        print(f"  Top {top} words present:  {count} ({100 * count / top:.1f}%)")
    print(f"  Spoken mass:         {s['spoken_mass']:.1f}%")
    limit = options.max_uses or "off"
    print(
        f"  Most uses of one word outside the top {EXEMPT_TOP}: "
        f"{s['max_uses']} (limit {limit})"
    )
    if options.final_mark and s["final_mark_share"] > FINAL_MARK_WARNING_SHARE:
        print(
            f"  Warning: {100 * s['final_mark_share']:.0f}% of the sentences fail rule Y4. "
            "Use --no-final-mark to skip it."
        )
