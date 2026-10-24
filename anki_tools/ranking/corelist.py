"""Core target list: the tier F candidates and the tier S + tier F core list.

The module holds the language-agnostic machinery of spec section 4. A language
package supplies the tokenized deck, the frequency ranks, the name stoplist,
the reviewed Swadesh rows, and the tier F sense rows. This module filters the
content lemmas, ranks them, merges the Swadesh rows, and fills the core list.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Collection, Sequence

from anki_tools.sentence_pack.english_sense import english_tokens, load_senses

CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})
CONTENT_SHARE = 0.5
MIN_DECK_COUNT = 3
MIN_LEMMA_LENGTH = 2
UNRANKED_RANK = 100_000


@dataclass(frozen=True)
class TargetWord:
    """One target word of the frequency fill.

    :param lemma: Lemma text.
    :param rank: Rank in the blended frequency list.
    :param deck_count: Occurrence count in the input deck.
    :param content_share: Share of the occurrences with a content part of speech.
    :param weight: Selection weight, ``1 / log2(rank + 1)``.
    """

    lemma: str
    rank: int
    deck_count: int
    content_share: float
    weight: float


@dataclass(frozen=True)
class SenseWords:
    """English sense data of one tier F target.

    :param senses: English sense words of the target.
    :param false_friends: English false-friend words of the target.
    :param first: First sense word of the file, used as the card gloss.
    """

    senses: frozenset[str]
    false_friends: frozenset[str]
    first: str


@dataclass(frozen=True)
class CoreTarget:
    """One word and one sense that the deck teaches.

    :param key: Unique key of the target, for example ``S1:je``.
    :param tier: ``S`` for a Swadesh form, ``F`` for a frequency lemma.
    :param position: 1-based position in the core list.
    :param form: Surface form of a tier S target, or the lemma of tier F.
    :param gloss: English gloss of the target: the Swadesh gloss of a tier S
        target, the first sense word of a tier F target, or empty when the
        target is unchecked.
    :param senses: English sense words, empty when the target is unchecked.
    :param false_friends: English false-friend words of the target.
    """

    key: str
    tier: str
    position: int
    form: str
    gloss: str
    senses: frozenset[str] = frozenset()
    false_friends: frozenset[str] = frozenset()

    @property
    def word(self) -> str:
        """Give the display word of the target."""
        return self.form

    @property
    def checked(self) -> bool:
        """Tell whether the target has a sense test."""
        return self.tier == "S" or bool(self.senses)


def build_targets(
    cards: Sequence[Any],
    tokens: Sequence[Sequence[tuple[str, str]]],
    ranks: dict[str, int],
    names: Collection[str],
    target_words: int | None = 300,
    min_deck_count: int = MIN_DECK_COUNT,
) -> tuple[list[TargetWord], int]:
    """Build the target word list of the frequency fill.

    The candidates are the content lemmas of the scan. The ranks sort them; a
    lemma that the rank table does not hold gets :data:`UNRANKED_RANK`, and an
    unranked tie breaks on the deck count. The filter drops a lemma with too
    few content tokens, a low content share, a short form, or a name.

    :param cards: Input cards. Only the count is used.
    :param tokens: Lemma and part of speech per card.
    :param ranks: Lemma to blended frequency rank.
    :param names: Name stoplist.
    :param target_words: Number of targets to keep, or None for every candidate.
    :param min_deck_count: Minimum content-token count of a candidate.
    :raises ValueError: The card list and the token list have different sizes.
    :returns: The targets in rank order, and the count of content lemmas before
        the target cut.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")

    totals: Counter[str] = Counter()
    content: Counter[str] = Counter()
    for card_tokens in tokens:
        for lemma, pos in card_tokens:
            totals[lemma] += 1
            if pos in CONTENT_POS:
                content[lemma] += 1

    candidates: list[TargetWord] = []
    for lemma, count in totals.items():
        if content[lemma] < min_deck_count:
            continue
        share = content[lemma] / count
        if share < CONTENT_SHARE:
            continue
        if len(lemma) < MIN_LEMMA_LENGTH or lemma in names:
            continue
        rank = ranks.get(lemma, UNRANKED_RANK)
        candidates.append(
            TargetWord(
                lemma=lemma,
                rank=rank,
                deck_count=content[lemma],
                content_share=share,
                weight=1 / math.log2(rank + 1),
            )
        )

    candidates.sort(key=lambda target: (target.rank, -target.deck_count, target.lemma))
    if target_words is None:
        return candidates, len(candidates)
    return candidates[:target_words], len(candidates)


def glosses_share_sense(english_a: str, english_b: str) -> bool:
    """Tell whether two Swadesh glosses show one sense.

    A shared word of the two sense sets is the test. A gloss with no sense row
    uses the words of its own text.

    :param english_a: First Swadesh gloss.
    :param english_b: Second Swadesh gloss.
    :returns: True when the two glosses share a word.
    """
    table = load_senses()
    first = table.get(english_a.casefold())
    second = table.get(english_b.casefold())
    words_a = (
        first.words | {word for phrase in first.phrases for word in phrase}
        if first
        else frozenset(english_tokens(english_a))
    )
    words_b = (
        second.words | {word for phrase in second.phrases for word in phrase}
        if second
        else frozenset(english_tokens(english_b))
    )
    return bool(words_a & words_b)


def build_core_targets(
    table: Sequence[tuple[int, str, str]],
    frequency_targets: Sequence[TargetWord],
    senses: dict[str, SenseWords],
    core_words: int,
) -> tuple[list[CoreTarget], list[CoreTarget]]:
    """Build the core list of tier S and tier F targets.

    Two Swadesh rows with the same form give one target when the glosses share
    a sense. Otherwise they give two targets, and the sense guard separates
    them.

    :param table: Reviewed Swadesh rows as ``(index, form, english gloss)``.
    :param frequency_targets: Tier F candidates in rank order.
    :param senses: Sense data of the tier F targets.
    :param core_words: Count of tier F targets to add.
    :returns: The core list in order, and the tier S rows that a merge dropped.
    """
    merged: list[CoreTarget] = []
    dropped: list[CoreTarget] = []
    by_form: dict[str, list[CoreTarget]] = {}
    for index, form, english in table:
        target = CoreTarget(
            key=f"S{index}:{form}",
            tier="S",
            position=0,
            form=form,
            gloss=english,
        )
        kept = by_form.get(form)
        if kept:
            same = [
                existing
                for existing in kept
                if glosses_share_sense(existing.gloss, target.gloss)
            ]
            if same:
                dropped.append(target)
                continue
            kept.append(target)
        else:
            by_form[form] = [target]
        merged.append(target)

    lemmas = {target.form.casefold() for target in merged}
    items: list[CoreTarget] = list(merged)
    added = 0
    for word in frequency_targets:
        if added >= core_words:
            break
        if word.lemma in lemmas:
            continue
        sense = senses.get(word.lemma, SenseWords(frozenset(), frozenset(), ""))
        items.append(
            CoreTarget(
                key=f"F:{word.lemma}",
                tier="F",
                position=0,
                form=word.lemma,
                gloss=sense.first,
                senses=sense.senses,
                false_friends=sense.false_friends,
            )
        )
        added += 1
    positioned = [
        CoreTarget(
            key=target.key,
            tier=target.tier,
            position=index,
            form=target.form,
            gloss=target.gloss,
            senses=target.senses,
            false_friends=target.false_friends,
        )
        for index, target in enumerate(items, start=1)
    ]
    return positioned, dropped
