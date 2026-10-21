"""Shared English sense test of the Swadesh sentence packs.

The test compares the paired English of a candidate sentence with the English
gloss of the card word. A card passes when the English holds a form that shows
the gloss sense. The table ``vocab/english_senses.json`` holds the forms of all
207 glosses, and ``scripts/build_english_senses.py`` writes it from WordNet.

The test is a reject filter, not a ranking hint. A candidate whose only target
word fails the test drops out of the pool, so the card gets another candidate or
none at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

SENSE_PATH = Path("vocab/english_senses.json")

CONTRACTIONS: dict[str, str] = {
    "let's": "let us",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'m": " am",
    "'d": " would",
    "'s": " is",
}
CONTRACTION_PATTERN = re.compile(
    "|".join(re.escape(key) for key in CONTRACTIONS)
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GlossSense:
    """English forms that show the sense of one gloss.

    :param words: Single-word surface forms.
    :param phrases: One token set per word of each multiword form.
    """

    words: frozenset[str]
    phrases: tuple[tuple[frozenset[str], ...], ...]


def english_tokens(text: str) -> tuple[str, ...]:
    """Split an English sentence into lowercase tokens.

    The function expands a contraction, so ``I'm`` gives ``i am`` and ``don't``
    gives ``do not``.

    :param text: English sentence text.
    :returns: Lowercase word and number tokens.
    """
    folded = text.casefold().replace("\u2019", "'")
    expanded = CONTRACTION_PATTERN.sub(
        lambda match: CONTRACTIONS[match.group(0)], folded
    )
    return tuple(TOKEN_PATTERN.findall(expanded))


@lru_cache(maxsize=8)
def load_senses(path: Path = SENSE_PATH) -> dict[str, GlossSense]:
    """Read the English sense table.

    :param path: JSON file written by ``scripts/build_english_senses.py``.
    :returns: Mapping from a casefolded gloss to its sense forms.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read the English sense table {path}"
        ) from error
    table: dict[str, GlossSense] = {}
    for gloss, entry in data["glosses"].items():
        phrases = tuple(
            tuple(frozenset(token) for token in phrase)
            for phrase in entry["phrases"]
        )
        table[gloss] = GlossSense(frozenset(entry["words"]), phrases)
    return table


def _phrase_matches(tokens: Sequence[str], phrase: Sequence[frozenset[str]]) -> bool:
    length = len(phrase)
    return any(
        all(tokens[position + index] in phrase[index] for index in range(length))
        for position in range(len(tokens) - length + 1)
    )


def holds_sense(
    gloss: str,
    english: str,
    senses: Mapping[str, GlossSense] | None = None,
) -> bool:
    """Tell whether the English shows the sense of the gloss.

    A gloss with no table entry does not reject a card.

    :param gloss: English Swadesh gloss of the card word.
    :param english: Paired English sentence.
    :param senses: Sense table, loaded from the data file when empty.
    :returns: True when the English holds a form of the gloss sense.
    """
    table = load_senses() if senses is None else senses
    sense = table.get(gloss.casefold())
    if sense is None:
        return True
    tokens = english_tokens(english)
    if not tokens:
        return False
    if sense.words.intersection(tokens):
        return True
    return any(_phrase_matches(tokens, phrase) for phrase in sense.phrases)


def sense_mismatches(
    rows: Iterable[Mapping[str, str]],
    gloss_by_form: Mapping[str, str],
    false_friend: Callable[[str, str], bool] | None = None,
    senses: Mapping[str, GlossSense] | None = None,
) -> list[str]:
    """List the cards whose English does not show the word sense.

    :param rows: Card rows with ``target``, ``sentence``, and ``english``.
    :param gloss_by_form: Mapping from a card target to its Swadesh gloss.
    :param false_friend: Language filter for a form-level false friend.
    :param senses: Sense table, loaded from the data file when empty.
    :returns: One ``target in sentence`` text per bad card.
    """
    table = load_senses() if senses is None else senses
    bad: list[str] = []
    for row in rows:
        target = row.get("target", "")
        english = row.get("english", "")
        if false_friend is not None and false_friend(target, english):
            bad.append(f"{target} in {row.get('sentence', '')}")
        elif not holds_sense(gloss_by_form.get(target, ""), english, table):
            bad.append(f"{target} in {row.get('sentence', '')}")
    return bad
