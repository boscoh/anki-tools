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

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

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


FALSE_FRIEND = "false_friend"
SENSE = "sense"


@dataclass(frozen=True)
class SenseRow:
    """The sense data of one target word.

    :param senses: English words that show the sense.
    :param false_friends: English words that show another sense.
    :param first: First sense word of the file, used as the card gloss.
    """

    senses: frozenset[str] = frozenset()
    false_friends: frozenset[str] = frozenset()
    first: str = ""


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


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


@lru_cache(maxsize=16)
def read_senses(path: Path) -> dict[tuple[str, str], SenseRow]:
    """Read the sense file of a language.

    The columns are ``word``, ``senses``, ``false_friends``, and ``tier``. The
    word lists hold comma-separated values. A blank ``tier`` is ``F``, a frequency lemma. ``S`` is a Swadesh form. A missing file gives no
    rows. The result is cached, so the caller must not change it.

    :param path: TSV file.
    :returns: ``(tier, casefolded word)`` to its row.
    """
    table: dict[tuple[str, str], SenseRow] = {}
    source = Path(path)
    if not source.exists():
        return table
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            word = (row.get("word") or "").strip().casefold()
            if not word:
                continue
            senses = _split((row.get("senses") or "").casefold())
            tier = (row.get("tier") or "").strip().upper() or "F"
            table[tier, word] = SenseRow(
                senses=frozenset(senses),
                false_friends=frozenset(
                    _split((row.get("false_friends") or "").casefold())
                ),
                first=senses[0] if senses else "",
            )
    return table


def sense_row(path: Path, tier: str, word: str) -> SenseRow | None:
    """Give the row of one word.

    :param path: Sense file.
    :param tier: ``S`` for a Swadesh form, ``F`` for a frequency lemma.
    :param word: Form or lemma.
    :returns: The row, or None when the file has none.
    """
    return read_senses(path).get((tier, word.casefold()))


def sense_failure(
    english: str,
    row: SenseRow | None = None,
    gloss: str | None = None,
    senses: Mapping[str, GlossSense] | None = None,
) -> str | None:
    """Test the English of a card against the sense of its target.

    A tier S form has a Swadesh gloss. The English passes when it holds a sense
    word of the row. Otherwise the row rejects it when it holds a false-friend
    word. Then the English must show the gloss sense; a gloss with no table
    entry does not reject.

    A tier F lemma has no gloss. A row with no sense words accepts every
    English. Otherwise the English must hold a sense word and no false-friend
    word.

    :param english: Paired English sentence.
    :param row: Sense row of the target, or None.
    :param gloss: Swadesh gloss of a tier S form, None for a tier F lemma.
    :param senses: Gloss table, loaded from the data file when empty.
    :returns: ``"false_friend"``, ``"sense"``, or None when the sense holds.
    """
    tokens = set(english_tokens(english))
    if gloss is None:
        if row is None or not row.senses:
            return None
        if row.false_friends & tokens:
            return FALSE_FRIEND
        return None if row.senses & tokens else SENSE
    if row is not None:
        if row.senses & tokens:
            return None
        if row.false_friends & tokens:
            return FALSE_FRIEND
    return None if _holds_gloss(gloss, english, senses) else SENSE


def _holds_gloss(
    gloss: str,
    english: str,
    senses: Mapping[str, GlossSense] | None,
) -> bool:
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

