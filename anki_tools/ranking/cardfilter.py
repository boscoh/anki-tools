"""Shared card filter rules F1 to F9 of the sentence decks.

The rules are the Latin default. The cull reads its mark rules and its length
window from the sentence shape guard with the values of the language profile;
this module holds the pieces that the cull shares: the English window F1, the
script rule F3, the proper-noun rule F8, and the F9 dedupe of
:func:`filter_cards`. The core flow applies the Latin rules as they stand.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12

FINAL_MARKS = frozenset(".?!")
BRACKET_MARKS = frozenset("[]{}<>")
QUOTE_MARKS = frozenset('"«»“”')
DIALOGUE_DASHES = frozenset("-–—")
INTERNAL_MARKS = ",;:"
ELLIPSIS_MARKS = ("...", "…")
URL_MARKS = ("http", "www.")

_DIGIT_PATTERN = re.compile(r"\d")
_WORD_PATTERN = re.compile(r"[^\W\d_]+")


def filter_cards(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    min_words: int = 3,
    max_words: int = 12,
    *,
    card_rule: Callable[..., str] | None = None,
) -> tuple[list[int], list[str]]:
    """Apply the filter rules F1 to F9.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :param card_rule: Rule function of one card, the Latin rules F1 to F8 of
        :func:`_filter_reason` when absent.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The input indices of the passed cards, and the reason per input
        index. A passed card has the empty string as its reason.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    rule = _filter_reason if card_rule is None else card_rule

    reasons = [""] * len(cards)
    passed: list[int] = []
    seen: set[str] = set()
    for index, card in enumerate(cards):
        reason = rule(card, tokens[index], min_words, max_words)
        if reason:
            reasons[index] = reason
            continue
        if card.sentence in seen:
            reasons[index] = "F9"
            continue
        seen.add(card.sentence)
        passed.append(index)
    return passed, reasons


def _filter_reason(
    card: Any,
    card_tokens: Sequence[tuple[str, str]],
    min_words: int,
    max_words: int,
) -> str:
    """Give the first failed filter rule of one card.

    :param card: Input card.
    :param card_tokens: Lemma and part of speech of the card sentence.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :returns: The rule name ``F1`` to ``F8``, or the empty string on a pass.
    """
    reason = english_rule(card)
    if reason:
        return reason
    if not min_words <= len(card_tokens) <= max_words:
        return "F2"
    sentence = card.sentence
    reason = script_rule(sentence)
    if reason:
        return reason
    if not sentence.endswith(tuple(FINAL_MARKS)):
        return "F4"
    if sum(1 for char in sentence if char in FINAL_MARKS) != 1:
        return "F5"
    if set(sentence) & QUOTE_MARKS or sentence[:1] in DIALOGUE_DASHES:
        return "F6"
    if sum(1 for char in sentence if char in INTERNAL_MARKS) > 1:
        return "F7"
    return pos_rule(card_tokens)


def english_rule(card: Any) -> str:
    """Give filter F1: the English word window.

    :param card: Input card.
    :returns: ``F1``, or the empty string on a pass.
    """
    if MIN_ENGLISH_WORDS <= len(card.english.split()) <= MAX_ENGLISH_WORDS:
        return ""
    return "F1"


def script_rule(sentence: str) -> str:
    """Give filter F3: no digit, bracket, link, or ellipsis.

    :param sentence: Prepared sentence text.
    :returns: ``F3``, or the empty string on a pass.
    """
    lower = sentence.lower()
    if (
        _DIGIT_PATTERN.search(sentence)
        or any(mark in sentence for mark in BRACKET_MARKS)
        or any(mark in lower for mark in URL_MARKS)
        or any(mark in sentence for mark in ELLIPSIS_MARKS)
    ):
        return "F3"
    return ""


def pos_rule(card_tokens: Sequence[tuple[str, str]]) -> str:
    """Give filter F8: no proper noun.

    :param card_tokens: Lemma and part of speech of the card sentence.
    :returns: ``F8``, or the empty string on a pass.
    """
    if any(pos == "PROPN" for _lemma, pos in card_tokens):
        return "F8"
    return ""
