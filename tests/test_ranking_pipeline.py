"""Tests for the shared rank and reorder pipeline.

The deck comes from :mod:`tests.deck_fixture`, so the tests need no committed
APKG file. The rank and the reducer are stubs, so the tests stay offline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from anki_tools.ranking.pipeline import RankPaths, rank_paths, run_ranked_deck
from tests.deck_fixture import build_ranking_deck

#: A name only. ``rank_paths`` reads no file.
NAME_ONLY = Path("apkg/deck.apkg")
KEPT = 100


def _identity(sentences: list) -> list:
    return sentences


@pytest.fixture(scope="module")
def deck_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_ranking_deck(tmp_path_factory.mktemp("ranking"))


@pytest.fixture
def paths(deck_path: Path, tmp_path: Path) -> RankPaths:
    return RankPaths(deck_path, tmp_path / "out.apkg", tmp_path / "out.rank.csv")


class TestRankPaths:
    def test_the_output_name_sets_the_ranking_path(self) -> None:
        output = Path("apkg/french_reordered.apkg")

        paths = rank_paths(NAME_ONLY, output)

        assert paths.input_path == NAME_ONLY
        assert paths.output_path == output
        assert paths.ranking_path == Path("apkg/french_reordered.rank.csv")

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
            reduce=lambda ranked: ranked[:KEPT],
            remove_unranked=True,
        )
        capsys.readouterr()

        assert len(deck.ranked) == KEPT
        assert deck.stats["cards_removed"] == len(deck.sentences) - KEPT
