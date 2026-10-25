"""Select mode of ``anki fix yue``: rank, filter, select, and write the CSVs.

The mode extracts and ranks the sentences of a deck, filters and selects them,
and writes the rank, select, and coverage CSV files next to the input. The
selection order sets the card order of the output deck.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from anki_tools.languages.cantonese.frequency import (
    Tokenizer,
    YueFrequency,
    build_yue_tokenizer,
    download_cifu,
    han_tokens,
    load_yue_frequency,
    rank_change_rows,
    read_cifu,
    read_overrides,
    write_freq_change_csv,
)
from anki_tools.languages.cantonese.rank import rank_sentences_yue
from anki_tools.languages.cantonese.select import (
    SelectionOptions,
    SelectionReport,
    print_selection_summary,
    read_names,
    select_yue_sentences,
    write_coverage_csv,
    write_select_csv,
)
from anki_tools.languages.mandarin.characters import (
    char_similarity_zh_same_len,
    get_chinese_chars,
)
from anki_tools.ranking.pipeline import RankPaths, rank_paths, run_ranked_deck
from anki_tools.ranking.rank import Sentence
from anki_tools.sentence_pack.audit import print_audit, sample

_DEFAULTS = SelectionOptions()
#: Output name suffix of a select run.
OUTPUT_SUFFIX = "_reordered"

#: Default frequency term of a select run.
DEFAULT_FREQ = "blend"

#: Default weight of the spoken term in the blended frequency term.
DEFAULT_SPOKEN_WEIGHT = 0.7

#: Default use cap of a select run, from the filter options.
DEFAULT_MAX_USES = _DEFAULTS.max_uses

#: Character Jaccard limit of gate G6.
SIMILARITY_LIMIT = 0.67

#: Most kept sentences for one word outside the top 100, of gate G12.
MAX_USES = 8

#: Words above this rank are exempt from the use cap of gate G12.
EXEMPT_TOP = 100

#: A gate row: the gate name, the pass flag, and the detail message.
GateRow = tuple[str, bool, str]


@dataclass(frozen=True)
class SelectRequest:
    """Options of one select run.

    The tuning defaults are the defaults of :class:`SelectionOptions`, so the
    request and the filter cannot drift apart.

    :param input_path: Input APKG file.
    :param output_path: Output APKG file.
    :param model_id: Model id to filter by, or None for every model.
    :param max_sentences: Stop the selection at this count, or None at zero gain.
    :param per_word: Cards wanted for each target word.
    :param max_uses: Most kept sentences for one word outside the top 100.
    :param top_words: Size of the target set.
    :param order: ``coverage`` or ``score``.
    :param min_han: Minimum Han characters in a sentence.
    :param max_han: Maximum Han characters in a sentence.
    :param max_unknown: Unknown tokens allowed in a sentence.
    :param known_top: Size of the known word set.
    :param similarity_threshold: Character Jaccard threshold for duplicates.
    :param final_mark: Apply the final-mark rule Y4.
    :param freq: Frequency term: ``blend``, ``spoken``, or ``written``.
    :param spoken_weight: Weight of the spoken term in the blend.
    :param freq_report: Write the sentences that the frequency change moves.
    """

    input_path: Path
    output_path: Path
    model_id: int | None = None
    max_sentences: int | None = None
    per_word: int = _DEFAULTS.per_word
    max_uses: int = _DEFAULTS.max_uses
    top_words: int = _DEFAULTS.top_words
    order: str = _DEFAULTS.order
    min_han: int = _DEFAULTS.min_han
    max_han: int = _DEFAULTS.max_han
    max_unknown: int = _DEFAULTS.max_unknown
    known_top: int = _DEFAULTS.known_top
    similarity_threshold: float = _DEFAULTS.similarity_threshold
    final_mark: bool = _DEFAULTS.final_mark
    freq: str = DEFAULT_FREQ
    spoken_weight: float = DEFAULT_SPOKEN_WEIGHT
    freq_report: bool = False

    def options(self) -> SelectionOptions:
        """Build the filter and selection options of the run.

        :returns: The options, with the foreign names of rule Y8.
        """
        return SelectionOptions(
            max_sentences=self.max_sentences,
            per_word=self.per_word,
            max_uses=self.max_uses,
            top_words=self.top_words,
            order=self.order,
            min_han=self.min_han,
            max_han=self.max_han,
            max_unknown=self.max_unknown,
            known_top=self.known_top,
            similarity_threshold=self.similarity_threshold,
            final_mark=self.final_mark,
            names=read_names(),
        )


@dataclass(frozen=True)
class SelectRun:
    """Files and counts of one select run.

    :param paths: Input, output, and ranking file of the run.
    :param ranked_count: Sentences that the ranker kept.
    :param report: The ordered sentences and the CSV rows, or None if the deck
        held no sentences.
    :param stats: Card counts of the reorder step.
    """

    paths: RankPaths
    ranked_count: int
    report: SelectionReport | None
    stats: dict[str, Any]


def output_stem(output_path: Path) -> Path:
    """Build the file stem that names the CSV files of a run.

    The CSV files are named after the input, so the output suffix is removed.

    :param output_path: Output APKG file.
    :returns: The input path without its suffix.
    """
    path = Path(output_path)
    stem = path.stem
    if stem.endswith(OUTPUT_SUFFIX):
        stem = stem[: -len(OUTPUT_SUFFIX)]
    return path.with_name(stem)


def default_output(apkg_path: Path) -> Path:
    """Build the default output name of a select run.

    :param apkg_path: Input APKG file.
    :returns: Path next to the input with the suffix ``_reordered.apkg``.
    """
    path = Path(apkg_path)
    return path.with_name(f"{path.stem}{OUTPUT_SUFFIX}.apkg")


def side_paths(output_path: Path) -> tuple[Path, Path]:
    """Build the select and coverage CSV paths of a select run.

    :param output_path: Output APKG file.
    :returns: The select CSV and the coverage CSV paths.
    """
    stem = output_stem(output_path)
    return Path(f"{stem}.select.csv"), Path(f"{stem}.coverage.csv")


def rank_csv_path(output_path: Path) -> Path:
    """Build the rank CSV path of a select run.

    :param output_path: Output APKG file.
    :returns: The rank CSV path.
    """
    return Path(f"{output_stem(output_path)}.rank.csv")


def run_select(request: SelectRequest) -> SelectRun:
    """Rank the sentences, filter and select them, and write the CSV files.

    :param request: Options of the run.
    :returns: The written paths, the sentence counts, and the report.
    """
    paths = rank_paths(request.input_path, request.output_path)
    select_csv, coverage_csv = side_paths(paths.output_path)

    print("=" * 50)
    print("Processing Cantonese Anki Deck")
    print("=" * 50)
    print(f"Input:   {paths.input_path}")
    print(f"Output:  {paths.output_path}")
    print(f"Ranking: {paths.ranking_path}")
    print()
    print("Step 1: Ranking sentences...")

    extracted: list[Sentence] = []
    counts: dict[str, int] = {}
    reports: dict[str, SelectionReport] = {}

    def rank_and_capture(sentences: list[Sentence]) -> list[Sentence]:
        extracted.extend(sentences)
        ranked = rank_sentences_yue(
            sentences, freq=request.freq, spoken_weight=request.spoken_weight
        )
        counts["ranked"] = len(ranked)
        return ranked

    def select_ranked(ranked: list[Sentence]) -> list[Sentence]:
        if request.freq_report and request.freq != "written":
            comparison = rank_sentences_yue(
                [
                    Sentence(
                        note_id=sentence.note_id,
                        text=sentence.text,
                        original_order=sentence.original_order,
                    )
                    for sentence in extracted
                ],
                freq="written",
            )
            rows = rank_change_rows(comparison, ranked)
            change_csv = Path(f"{output_stem(paths.output_path)}.freq_change.csv")
            write_freq_change_csv(rows, change_csv)
            print(f"  {len(rows)} sentences move by more than 100 places: {change_csv}")
        options = request.options()
        print("\nStep 2: Selecting sentences...")
        frequency = load_yue_frequency(
            freq=request.freq, spoken_weight=request.spoken_weight
        )
        tokenizer = build_yue_tokenizer(
            read_cifu(download_cifu()), extra_words=read_overrides()
        )
        ranked_ids = {id(sentence) for sentence in ranked}
        dropped = [s for s in extracted if id(s) not in ranked_ids]
        report = select_yue_sentences(ranked, dropped, frequency, tokenizer, options)
        write_select_csv(report, select_csv)
        write_coverage_csv(report, coverage_csv)
        print_selection_summary(report, options)
        print("\nStep 3: Reordering deck...")
        reports["report"] = report
        return report.ordered

    deck = run_ranked_deck(
        paths,
        rank=rank_and_capture,
        model_id=request.model_id,
        romanization_candidates=("Jyutping",),
        reduce=select_ranked,
        remove_unranked=True,
    )
    return SelectRun(
        paths,
        counts.get("ranked", len(deck.ranked)),
        reports.get("report"),
        deck.stats,
    )


def check_select(output_path: Path) -> int:
    """Run gates G5, G6, G7, G9, and G12 on the files of a select run.

    G10 and G11 measure the LTL deck, so they stay in the test suite. The
    gate limits are the limits of the spec.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    select_csv, _coverage_csv = side_paths(output_path)
    rank_csv = rank_csv_path(output_path)
    rank_rows = [row for row in read_rows(rank_csv) if row.get("kept") == "1"]
    select_rows = read_rows(select_csv)
    frequency = load_yue_frequency()
    tokenizer = build_yue_tokenizer(
        read_cifu(download_cifu()), extra_words=read_overrides()
    )
    return print_audit(
        [
            gate_kept_reasons(select_rows),
            gate_no_duplicates(rank_rows),
            gate_select_rows(rank_rows, select_rows),
            gate_rank_order(rank_rows),
            gate_use_cap(rank_rows, frequency, tokenizer),
        ]
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings.

    :param path: CSV file with a header line.
    :returns: One mapping per data row.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def gate_kept_reasons(select_rows: Sequence[dict[str, str]]) -> GateRow:
    """Gate G5: no kept sentence carries a filter reason.

    :param select_rows: Rows of the select CSV.
    :returns: The gate row.
    """
    failures = [
        row["sentence"]
        for row in select_rows
        if row.get("kept") == "1" and row.get("reason")
    ]
    kept = sum(1 for row in select_rows if row.get("kept") == "1")
    if failures:
        return (
            "G5",
            False,
            f"{len(failures)} kept sentence(s) fail a rule: {sample(failures)}",
        )
    return "G5", True, f"{kept} kept sentences fail no rule" if kept else "no kept rows"


def gate_no_duplicates(rank_rows: Sequence[dict[str, str]]) -> GateRow:
    """Gate G6: no kept pair of one length reaches the Jaccard limit.

    :param rank_rows: Kept rows of the rank CSV.
    :returns: The gate row.
    """
    by_size: dict[int, list[str]] = {}
    offenders: list[str] = []
    for row in rank_rows:
        text = row["sentence"]
        size = len(set(get_chinese_chars(text)))
        near = [
            other
            for other in by_size.get(size, [])
            if char_similarity_zh_same_len(text, other) >= SIMILARITY_LIMIT
        ]
        if near:
            offenders.append(f"{text} / {near[0]}")
        by_size.setdefault(size, []).append(text)
    if offenders:
        return (
            "G6",
            False,
            f"{len(offenders)} kept pair(s) reach the {SIMILARITY_LIMIT:.2f} limit: "
            f"{sample(offenders)}",
        )
    return (
        "G6",
        True,
        f"{len(rank_rows)} kept sentences stay below {SIMILARITY_LIMIT:.2f}",
    )


def gate_select_rows(
    rank_rows: Sequence[dict[str, str]], select_rows: Sequence[dict[str, str]]
) -> GateRow:
    """Gate G7: the select CSV holds one row per kept sentence.

    :param rank_rows: Kept rows of the rank CSV.
    :param select_rows: Rows of the select CSV.
    :returns: The gate row.
    """
    kept = {row["sentence"] for row in select_rows if row.get("kept") == "1"}
    missing = sorted({row["sentence"] for row in rank_rows} - kept)
    empty = sum(1 for row in select_rows if not row["sentence"])
    distinct = len({row["sentence"] for row in select_rows})
    repeated = len(select_rows) - distinct - empty
    problems: list[str] = []
    if missing:
        problems.append(
            f"{len(missing)} kept sentence(s) not kept in the CSV: {sample(missing)}"
        )
    if repeated:
        problems.append(f"{repeated} repeated select row(s)")
    if empty:
        problems.append(f"{empty} empty select row(s)")
    if problems:
        return "G7", False, "; ".join(problems)
    return (
        "G7",
        True,
        f"{len(rank_rows)} kept sentences, {distinct} distinct select rows",
    )


def gate_rank_order(rank_rows: Sequence[dict[str, str]]) -> GateRow:
    """Gate G9: the hardest rank of the kept rows does not decrease.

    :param rank_rows: Kept rows of the rank CSV.
    :returns: The gate row.
    """
    ranks = [int(row["hardest_rank"]) for row in rank_rows]
    drops = [
        f"{ranks[index - 1]} then {ranks[index]}"
        for index in range(1, len(ranks))
        if ranks[index] < ranks[index - 1]
    ]
    if drops:
        return (
            "G9",
            False,
            f"{len(drops)} kept row(s) go back in rank: {sample(drops)}",
        )
    return "G9", True, f"{len(ranks)} kept rows do not decrease in rank"


def gate_use_cap(
    rank_rows: Sequence[dict[str, str]],
    frequency: YueFrequency,
    tokenizer: Tokenizer,
) -> GateRow:
    """Gate G12: no word outside the top 100 passes the use cap.

    :param rank_rows: Kept rows of the rank CSV.
    :param frequency: The frequency list and its variant map.
    :param tokenizer: Tokenizer from :func:`build_yue_tokenizer`.
    :returns: The gate row.
    """
    uses: Counter[str] = Counter()
    for row in rank_rows:
        tokens = {
            frequency.variants.get(token, token)
            for token in han_tokens(tokenizer, row["sentence"])
        }
        uses.update(tokens)
    ranks = frequency.ranks
    limited = {
        word: count
        for word, count in uses.items()
        if ranks.get(word, EXEMPT_TOP + 1) > EXEMPT_TOP
    }
    peak = max(limited.values(), default=0)
    if peak > MAX_USES:
        offenders = [
            f"{word} x{count}"
            for word, count in sorted(limited.items(), key=lambda item: -item[1])
            if count > MAX_USES
        ]
        return (
            "G12",
            False,
            f"{len(offenders)} word(s) pass the {MAX_USES} cap: {sample(offenders)}",
        )
    return (
        "G12",
        True,
        f"no word outside the top {EXEMPT_TOP} passes {MAX_USES} uses, max {peak}",
    )
