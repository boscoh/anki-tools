"""Tests for the shared lemma Jaccard helper."""

from __future__ import annotations

import pytest

from anki_tools.ranking.duplicates import at_limit, jaccard


def test_empty_sets_give_zero() -> None:
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"chat"}, set()) == 0.0
    assert jaccard(set(), {"chat"}) == 0.0


def test_disjoint_sets_give_zero() -> None:
    assert jaccard({"chat", "chien"}, {"boire", "lait"}) == 0.0


def test_equal_sets_give_one() -> None:
    assert jaccard({"chat", "chien"}, {"chien", "chat"}) == 1.0


def test_shared_lemma_over_the_union() -> None:
    assert jaccard({"chat", "chien"}, {"chat", "lait"}) == pytest.approx(1 / 3)


def test_at_limit_includes_the_boundary() -> None:
    assert at_limit({"chat", "chien"}, {"chat", "lait"}, 1 / 3)
    assert at_limit({"chat", "chien"}, {"chat", "chien"}, 1.0)
    assert not at_limit({"chat", "chien"}, {"chat", "chien"}, 1.1)
    assert not at_limit({"chat", "chien"}, {"boire", "lait"}, 0.1)
    assert not at_limit(set(), set(), 0.1)
