"""Shared card filter rules of the sentence decks.

The cull and the core read their length window, text rules, and mark rules from
the sentence shape guard with the values of the language profile. This module
holds :func:`filter_reason`, which runs the shape guard as the sentence pack does.
:func:`filter_cards` adds the ``unique`` dedupe.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol, Sequence

from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.profile import SentenceShape

MARK_RULES = ("marks",)
LENGTH_RULES = ("length_window",)
TEXT_RULES = ("digit_marks", "script", "repeat")

_WORD_PATTERN = re.compile(r"[^\W\d_]+")


def filter_cards(
    cards: Sequence[Any],
    *,
    card_rule: Callable[..., str],
    dedupe_key: Callable[[str], str],
) -> tuple[list[int], list[str]]:
    """Apply the shared filter rules.

    :param cards: Input cards.
    :param card_rule: Rule function of one card, :func:`filter_reason` with the
        profile of the language.
    :param dedupe_key: Key of filter ``unique``, from the language profile.
    :returns: The input indices of the passed cards, and the reason per input
        index. A passed card has the empty string as its reason.
    """

    reasons = [""] * len(cards)
    passed: list[int] = []
    seen: set[str] = set()
    for index, card in enumerate(cards):
        reason = card_rule(card)
        if reason:
            reasons[index] = reason
            continue
        key = dedupe_key(card.sentence)
        if key in seen:
            reasons[index] = "unique"
            continue
        seen.add(key)
        passed.append(index)
    return passed, reasons


class ShapeProfile(SentenceShape, Protocol):
    """The values of a language that the cull and the core read, as their profiles hold them."""

    @property
    def dedupe_key(self) -> Callable[[str], str]: ...

    @property
    def min_units(self) -> int: ...

    @property
    def max_units(self) -> int: ...


def filter_reason(
    card: Any,
    profile: ShapeProfile,
) -> str:
    """Give the first failed filter rule of one card of the cull or the core.

    The length window, the text rules, and the mark rules come from the sentence
    shape guard, so the unit count, the marks, and the alphabet test are those of
    the language profile.

    :param card: Input card.
    :param profile: Shape values of the language.
    :returns: The first failed rule id, or the empty string on a pass.
    """
    # The rank flows pass only the length, text, and mark keys.
    return (
        check_sentence_shape(
            card.sentence,
            profile,
            profile.min_units,
            profile.max_units,
            keys=LENGTH_RULES + TEXT_RULES + MARK_RULES,
        )
        or ""
    )
