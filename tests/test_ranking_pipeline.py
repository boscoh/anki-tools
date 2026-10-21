"""Tests for the shared rank and reorder pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from anki_tools.ranking.pipeline import RankPaths, rank_paths, run_ranked_deck

APKG = Path(__file__).parent / "A_Course_in_Contemporary_Chinese_1.apkg"


def _identity(sentences: list) -> list:
    return sentences


@pytest.fixture
def paths(tmp_path: Path) -> RankPaths:
    return RankPaths(APKG, tmp_path / "out.apkg", tmp_path / "out.rank.csv")


class TestRankPaths:
    def test_the_output_default_gets_the_reordered_name(self) -> None:
        paths = rank_paths(APKG)

        assert paths.input_path == APKG
        assert paths.output_path == APKG.parent / f"{APKG.stem}_reordered.apkg"
        assert paths.ranking_path == APKG.parent / f"{APKG.stem}.rank.csv"

    def test_a_given_output_keeps_the_ranking_path(self, tmp_path: Path) -> None:
        output = tmp_path / "custom.apkg"

        paths = rank_paths(APKG, output)

        assert paths.output_path == output
        assert paths.ranking_path == APKG.parent / f"{APKG.stem}.rank.csv"


class TestRunRankedDeck:
    def test_the_pipeline_ranks_and_reorders(
        self, paths: RankPaths, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deck = run_ranked_deck(paths, rank=_identity)
        capsys.readouterr()

        assert deck.sentences
        assert deck.ranked == deck.sentences
        assert paths.output_path.exists()
        assert paths.ranking_path.exists()
        with paths.ranking_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(deck.ranked)
        assert deck.stats["cards_reordered"] > 0
        assert deck.stats["cards_removed"] == 0

    def test_the_reducer_limits_the_ranked_sentences(
        self, paths: RankPaths, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deck = run_ranked_deck(
            paths,
            rank=_identity,
            reduce=lambda ranked: ranked[:100],
            remove_unranked=True,
        )
        capsys.readouterr()

        assert len(deck.ranked) == 100
        assert deck.stats["cards_removed"] > 0
