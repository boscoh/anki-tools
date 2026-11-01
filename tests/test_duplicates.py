"""Tests for the shared lemma Jaccard helper."""

from __future__ import annotations

import pytest

from anki_tools.ranking.duplicates import at_limit

CAT_DOG = {"chat", "chien"}


@pytest.mark.parametrize(
    "left, right, limit, expected",
    [
        pytest.param(CAT_DOG, {"chat", "lait"}, 1 / 3, True, id="boundary-is-included"),
        pytest.param(CAT_DOG, CAT_DOG, 1.0, True, id="equal-at-one"),
        pytest.param(CAT_DOG, CAT_DOG, 1.1, False, id="above-the-limit"),
        pytest.param(CAT_DOG, {"boire", "lait"}, 0.1, False, id="disjoint"),
        pytest.param(set(), set(), 0.1, False, id="empty"),
    ],
)
def test_at_limit_includes_the_boundary(
    left: set[str], right: set[str], limit: float, expected: bool
) -> None:
    assert at_limit(left, right, limit) is expected
