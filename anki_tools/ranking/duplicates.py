"""Lemma Jaccard test that the duplicate guards share."""

from __future__ import annotations

from typing import AbstractSet


def jaccard(first: AbstractSet[str], second: AbstractSet[str]) -> float:
    """Give the Jaccard value of two lemma sets.

    :param first: The lemmas of one card.
    :param second: The lemmas of the other card.
    :returns: The part of the union that both sets hold, or 0.0 when the union
        is empty.
    """
    union = len(first | second)
    if not union:
        return 0.0
    return len(first & second) / union


def at_limit(first: AbstractSet[str], second: AbstractSet[str], limit: float) -> bool:
    """Tell whether two lemma sets reach the Jaccard limit.

    :param first: The lemmas of one card.
    :param second: The lemmas of the other card.
    :param limit: The Jaccard value that marks a duplicate pair.
    :returns: True when the value reaches the limit.
    """
    return jaccard(first, second) >= limit
