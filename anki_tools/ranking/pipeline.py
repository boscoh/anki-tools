"""Shared rank and reorder pipeline for the ``fix`` commands.

The pipeline extracts the sentences of a deck, ranks them with the ranker of
the language, optionally reduces them, writes the ranking CSV, and reorders
the deck. The caller prints the progress and the summary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anki_tools.ranking.rank import (
    Sentence,
    extract_sentences,
    write_ranking_csv,
)
from anki_tools.ranking.reorder import reorder_deck

#: A reducer takes the ranked sentences and gives the kept sentences.
Reducer = Callable[[list[Sentence]], list[Sentence]]


@dataclass(frozen=True)
class RankPaths:
    """Input, output, and ranking file of one rank run.

    :param input_path: Input APKG file.
    :param output_path: Output APKG file.
    :param ranking_path: Ranking CSV file.
    """

    input_path: Path
    output_path: Path
    ranking_path: Path


@dataclass(frozen=True)
class RankedDeck:
    """Result of one rank run.

    :param paths: Input, output, and ranking file of the run.
    :param sentences: Extracted sentences.
    :param ranked: Ranked sentences, after the optional reduction.
    :param stats: Card counts of the reorder step.
    """

    paths: RankPaths
    sentences: list[Sentence]
    ranked: list[Sentence]
    stats: dict[str, Any]


def rank_paths(apkg_path: Path, output: Path | None = None) -> RankPaths:
    """Build the three file paths of a rank run from the input name.

    :param apkg_path: Input APKG file.
    :param output: Output APKG file, or None for the ``_reordered`` name.
    :returns: The input path, the output path, and the ranking CSV path.
    """
    path = Path(apkg_path)
    output_path = (
        Path(output) if output is not None else path.parent / f"{path.stem}_reordered.apkg"
    )
    ranking_path = path.parent / f"{path.stem}.rank.csv"
    return RankPaths(path, output_path, ranking_path)


def run_ranked_deck(
    paths: RankPaths,
    *,
    rank: Callable[[list[Sentence]], list[Sentence]],
    model_id: int | None = None,
    text_field: str | None = None,
    text_candidates: Sequence[str] = (),
    romanization_candidates: Sequence[str] = (),
    keep_filtered: bool = True,
    reduce: Reducer | None = None,
    remove_unranked: bool = False,
) -> RankedDeck:
    """Extract, rank, write the ranking CSV, and reorder the deck.

    An empty extraction stops before the ranking and the reorder.

    :param paths: Input, output, and ranking file of the run.
    :param rank: Ranker of the language, sentences to ranked sentences.
    :param model_id: Model id to filter by, or None for every model.
    :param text_field: Preferred sentence field, or None to infer the field.
    :param text_candidates: Sentence field names of the language, in order.
    :param romanization_candidates: Romanization field names, in order.
    :param keep_filtered: Keep the cards that the ranker marks for removal.
    :param reduce: Reducer of the ranked sentences, or None to keep them all.
    :param remove_unranked: Remove the cards that the ranking CSV does not hold.
    :returns: The extracted sentences, the ranked sentences, and the card counts.
    """
    sentences = extract_sentences(
        str(paths.input_path),
        model_id,
        text_field=text_field,
        text_candidates=text_candidates,
        romanization_candidates=romanization_candidates,
    )
    if not sentences:
        return RankedDeck(paths, [], [], {})

    ranked = rank(sentences)
    if reduce is not None:
        ranked = reduce(ranked)

    write_ranking_csv(ranked, str(paths.ranking_path))
    stats = reorder_deck(
        str(paths.input_path),
        str(paths.output_path),
        str(paths.ranking_path),
        remove_filtered=not keep_filtered,
        remove_unranked=remove_unranked,
    )
    return RankedDeck(paths, sentences, ranked, stats)
