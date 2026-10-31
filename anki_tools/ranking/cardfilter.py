"""Shared card filter rules of the sentence decks.

The cull and the core read their mark rules and their length window from the
sentence shape guard with the values of the language profile. This module
holds :func:`filter_reason`, which joins the two with the English window
``english_words``, the ``digit_marks`` rule, the ``propn`` rule, and the
``unique`` dedupe of :func:`filter_cards`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, cast

from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.profile import LanguageProfile

MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12

MARK_RULES = ("final_mark", "one_final_mark", "quote", "internal_mark")
LENGTH_RULES = ("length_window",)
BRACKET_MARKS = frozenset("[]{}<>")
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
    card_rule: Callable[..., str],
) -> tuple[list[int], list[str]]:
    """Apply the shared filter rules.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param min_words: Minimum word count of a card.
    :param max_words: Maximum word count of a card.
    :param card_rule: Rule function of one card, :func:`filter_reason` with the
        marks of the language.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The input indices of the passed cards, and the reason per input
        index. A passed card has the empty string as its reason.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")

    reasons = [""] * len(cards)
    passed: list[int] = []
    seen: set[str] = set()
    for index, card in enumerate(cards):
        reason = card_rule(card, tokens[index], min_words, max_words)
        if reason:
            reasons[index] = reason
            continue
        if card.sentence in seen:
            reasons[index] = "unique"
            continue
        seen.add(card.sentence)
        passed.append(index)
    return passed, reasons


class Marks(Protocol):
    """The mark sets of a language, as the cull profile and the core profile hold them."""

    @property
    def final_marks(self) -> str: ...

    @property
    def quote_marks(self) -> str: ...

    @property
    def internal_marks(self) -> str: ...


@dataclass(frozen=True)
class CardShape:
    """Shape-guard values of one card.

    The rank flows measure the length window with the tokenizer output, so the
    unit count comes from the card and not from the sentence text.

    :param units: Token count of the card sentence.
    :param final_marks: Accepted sentence-final marks of the language.
    :param quote_marks: Quotation marks and dialogue dashes of the language.
    :param internal_marks: Internal punctuation marks of the language.
    """

    units: int
    final_marks: str
    quote_marks: str
    internal_marks: str

    def unit_count(self, sentence: str) -> int:
        """Give the token count of the card.

        :param sentence: Prepared sentence text.
        :returns: The token count of the card.
        """
        return self.units


def filter_reason(
    card: Any,
    card_tokens: Sequence[tuple[str, str]],
    min_words: int,
    max_words: int,
    profile: Marks,
) -> str:
    """Give the first failed filter rule of one card of the cull or the core.

    The English window, the script rule, and the proper-noun rule come from this
    module. The length window and the mark rules come from the sentence shape
    guard, so the marks are the marks of the language profile.

    :param card: Input card.
    :param card_tokens: Lemma and part of speech of the card sentence.
    :param min_words: Minimum token count of a card.
    :param max_words: Maximum token count of a card.
    :param profile: Mark sets of the language.
    :returns: The rule id from ``english_words`` to ``propn``, or the empty
        string on a pass.
    """
    reason = english_rule(card)
    if reason:
        return reason
    shape = CardShape(
        len(card_tokens),
        profile.final_marks,
        profile.quote_marks,
        profile.internal_marks,
    )
    # The rank flows pass only the length key and the mark keys, so the adapter
    # gives those values and not the rest of a language profile.
    guard_profile = cast(LanguageProfile, shape)
    reason = check_sentence_shape(
        card.sentence, guard_profile, min_words, max_words, keys=LENGTH_RULES
    )
    if reason:
        return reason
    reason = script_rule(card.sentence)
    if reason:
        return reason
    reason = check_sentence_shape(
        card.sentence, guard_profile, min_words, max_words, keys=MARK_RULES
    )
    if reason:
        return reason
    return pos_rule(card_tokens)


def english_rule(card: Any) -> str:
    """Give filter ``english_words``: the English word window.

    :param card: Input card.
    :returns: ``english_words``, or the empty string on a pass.
    """
    if MIN_ENGLISH_WORDS <= len(card.english.split()) <= MAX_ENGLISH_WORDS:
        return ""
    return "english_words"


def script_rule(sentence: str) -> str:
    """Give filter ``digit_marks``: no digit, bracket, link, or ellipsis.

    :param sentence: Prepared sentence text.
    :returns: ``digit_marks``, or the empty string on a pass.
    """
    lower = sentence.lower()
    if (
        _DIGIT_PATTERN.search(sentence)
        or any(mark in sentence for mark in BRACKET_MARKS)
        or any(mark in lower for mark in URL_MARKS)
        or any(mark in sentence for mark in ELLIPSIS_MARKS)
    ):
        return "digit_marks"
    return ""


def pos_rule(card_tokens: Sequence[tuple[str, str]]) -> str:
    """Give filter ``propn``: no proper noun.

    :param card_tokens: Lemma and part of speech of the card sentence.
    :returns: ``propn``, or the empty string on a pass.
    """
    if any(pos == "PROPN" for _lemma, pos in card_tokens):
        return "propn"
    return ""
