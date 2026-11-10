"""Tests of the English ``name`` filter rule of the cull and the core.

The rule reads the English field of a card. The table holds the pass and the
reject cases as data. Each row uses the same deck sentence, so only the English
text can give the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from anki_tools.ranking.cardfilter import filter_reason


@dataclass(frozen=True)
class _Card:
    sentence: str
    english: str


@dataclass(frozen=True)
class _Shape:
    """A shape profile that passes every guard but the ``name`` rule."""

    final_marks: str = ".?!"
    quote_marks: str = '"«»“”'
    internal_marks: str = ",;:"
    script_ok: Callable[[str], bool] = lambda _text: True
    is_repeated: Callable[[str], bool] = lambda _text: False
    unit_count: Callable[[str], int] = lambda text: len(text.split())
    dedupe_key: Callable[[str], str] = str
    min_units: int = 1
    max_units: int = 10
    register_ok: Callable[[str], bool] | None = None


@pytest.mark.parametrize(
    ("english", "expected"),
    [
        pytest.param("Tom likes to", "name", id="tom"),
        pytest.param(
            "It was already dark when Tom got home.", "name", id="tom-inside"
        ),
        pytest.param("Tom likes Mary's long black hair.", "name", id="possessive"),
        pytest.param("May I lie on the sofa?", "", id="may-modal"),
        pytest.param("May fifth is Children's Day.", "", id="may-month"),
        pytest.param("I will go.", "", id="will"),
        pytest.param("I drink water.", "", id="no-name"),
    ],
)
def test_the_name_rule_reads_the_english_field(english: str, expected: str) -> None:
    card = _Card(sentence="La carte est ici.", english=english)

    assert filter_reason(card, _Shape()) == expected
