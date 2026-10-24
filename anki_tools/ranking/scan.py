"""Shared scan: extract, tokenize, count, and blend the frequency ranks.

The scan is the first stage of the core flow of spec section 3. A language
package supplies the five adapters of a :class:`CoreProfile`, and :func:`scan`
runs them in order. The result feeds the shared target-list builders of
``anki_tools/ranking/corelist.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CoreProfile:
    """Language hooks of the shared scan.

    Each field is one adapter that the language package supplies. The scan calls
    them in the order below.

    :param load_input_deck: Read the cards of the input deck from an APKG file.
    :param tokenize_deck: Tokenize every card into ``(lemma, part of speech)``
        pairs.
    :param load_frequency: Read the frequency rows of the language.
    :param frequency_ranks: Rank the lemmas by the blended frequency.
    :param read_names: Read the name stoplist.
    """

    load_input_deck: Callable[[Path], Any]
    tokenize_deck: Callable[[Any], Any]
    load_frequency: Callable[[], Any]
    frequency_ranks: Callable[[Any, float], dict[str, int]]
    read_names: Callable[[], set[str]]


@dataclass(frozen=True)
class ScanResult:
    """Output of one scan.

    :param cards: Input cards, as ``load_input_deck`` returns them.
    :param tokens: Lemma and part of speech per card, as ``tokenize_deck``
        returns them.
    :param ranks: Lemma to blended frequency rank.
    :param names: Name stoplist.
    """

    cards: Any
    tokens: Any
    ranks: dict[str, int]
    names: set[str]


def scan(profile: CoreProfile, apkg_path: Path, spoken_weight: float) -> ScanResult:
    """Run the five adapters of the profile in order.

    :param profile: Language hooks of the scan.
    :param apkg_path: Input APKG file.
    :param spoken_weight: Weight of the spoken term in the frequency blend.
    :returns: The cards, the tokens, the ranks, and the names.
    """
    cards = profile.load_input_deck(apkg_path)
    tokens = profile.tokenize_deck(cards)
    frequency = profile.load_frequency()
    ranks = profile.frequency_ranks(frequency, spoken_weight)
    names = profile.read_names()
    return ScanResult(cards=cards, tokens=tokens, ranks=ranks, names=names)
