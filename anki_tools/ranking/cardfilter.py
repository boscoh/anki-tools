"""Shared card filter rules of the sentence decks.

The cull and the core read their length window, text rules, and mark rules from
the sentence shape guard with the values of the language profile. This module
holds :func:`filter_reason`, which runs the shape guard as the sentence pack does.
:func:`filter_cards` adds the ``unique`` dedupe.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol, Sequence

from anki_tools.sentence_pack.english_sense import load_names, name_failure
from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.profile import SentenceShape

MARK_RULES = ("marks",)
LENGTH_RULES = ("length_window",)
TEXT_RULES = ("digit_marks", "script", "repeat")

_WORD_PATTERN = re.compile(r"[^\W\d_]+")


def english_key(english: str) -> str:
    """Give the dedupe key of an English text.

    :param english: English text of a card.
    :returns: The casefolded text with single spaces. It is empty for an empty text.
    """
    return " ".join(english.casefold().split())


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
    :param dedupe_key: Sentence key of filter ``unique``, from the language
        profile. A card also fails ``unique`` when its :func:`english_key` is
        not empty and an earlier passed card has the same one.
    :returns: The input indices of the passed cards, and the reason per input
        index. A passed card has the empty string as its reason.
    """

    reasons = [""] * len(cards)
    passed: list[int] = []
    seen: set[str] = set()
    seen_english: set[str] = set()
    for index, card in enumerate(cards):
        reason = card_rule(card)
        if reason:
            reasons[index] = reason
            continue
        key = dedupe_key(card.sentence)
        english = english_key(card.english)
        if key in seen or (english and english in seen_english):
            reasons[index] = "unique"
            continue
        seen.add(key)
        seen_english.add(english)
        passed.append(index)
    return passed, reasons


class ShapeProfile(SentenceShape, Protocol):
    """The values of a language that the cull and the core read, as their profiles hold them."""

    @property
    def dedupe_key(self) -> Callable[[str], str]: ...

    @property
    def register_ok(self) -> Callable[[str], bool] | None: ...

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
    the language profile. The ``register`` rule follows them, when the profile
    has a test for it. The ``name`` rule follows ``register`` and reads the
    English field, so the cull and the core reject a card whose translation
    holds a personal name. The sentence pack does not run either rule.

    :param card: Input card.
    :param profile: Shape values of the language.
    :returns: The first failed rule id, or the empty string on a pass.
    """
    # The rank flows pass only the length, text, and mark keys.
    reason = check_sentence_shape(
        card.sentence,
        profile,
        profile.min_units,
        profile.max_units,
        keys=LENGTH_RULES + TEXT_RULES + MARK_RULES,
    )
    if reason:
        return reason
    register_ok = profile.register_ok
    if register_ok and not register_ok(card.sentence):
        return "register"
    names = load_names()
    return "name" if name_failure(card.english, names) else ""
