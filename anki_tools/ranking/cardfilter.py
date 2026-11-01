"""Shared card filter rules of the sentence decks.

The cull and the core read their length window, text rules, and mark rules from
the sentence shape guard with the values of the language profile. This module
holds :func:`filter_reason`, which runs the shape guard first, as the sentence
pack does, and then the English window ``english_words`` of the language
profile. :func:`filter_cards` adds the ``unique`` dedupe.
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
    min_words: int = 3,
    max_words: int = 12,
    *,
    card_rule: Callable[..., str],
    dedupe_key: Callable[[str], str],
) -> tuple[list[int], list[str]]:
    """Apply the shared filter rules.

    :param cards: Input cards.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
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
        reason = card_rule(card, min_words, max_words)
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
    def english_bounds(self) -> tuple[int, int]: ...


def filter_reason(
    card: Any,
    min_words: int,
    max_words: int,
    profile: ShapeProfile,
) -> str:
    """Give the first failed filter rule of one card of the cull or the core.

    The shape guard runs first, as in the sentence pack. The length window, the
    text rules, and the mark rules come from the sentence shape guard, so the
    unit count, the marks, and the alphabet test are those of the language
    profile. The English window of the language profile runs last.

    :param card: Input card.
    :param min_words: Minimum unit count of a card.
    :param max_words: Maximum unit count of a card.
    :param profile: Shape values and English window of the language.
    :returns: The rule id from ``length_window`` to ``english_words``, or the
        empty string on a pass.
    """
    # The rank flows pass only the length, text, and mark keys.
    reason = check_sentence_shape(
        card.sentence,
        profile,
        min_words,
        max_words,
        keys=LENGTH_RULES + TEXT_RULES + MARK_RULES,
    )
    return reason or english_rule(card, profile.english_bounds)


def english_rule(card: Any, bounds: tuple[int, int]) -> str:
    """Give filter ``english_words``: the English word window.

    :param card: Input card.
    :param bounds: Minimum and maximum English word count, from the language
        profile.
    :returns: ``english_words``, or the empty string on a pass.
    """
    low, high = bounds
    if low <= len(card.english.split()) <= high:
        return ""
    return "english_words"
