"""Build the shared English sense table of the Swadesh sentence packs.

Each English Swadesh gloss gets the English surface forms that show its sense.
A sentence pack uses the table to reject a card when the paired English does
not hold the sense of the card word.

Run the script with the WordNet corpus available::

    uv run --with nltk python scripts/build_english_senses.py

The script downloads the WordNet corpus when it is absent. The output is
``vocab/english_senses.json``. The sentence-pack code reads only that file, so a
pack build does not need nltk.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Iterable

from nltk.corpus import wordnet as wn

LABELS_PATH = Path("vocab/swadesh.txt")
OUTPUT_PATH = Path("vocab/english_senses.json")

# Closed-class glosses. WordNet gives a poor or empty sense set for these, so
# the table lists the English forms that show the gloss sense.
FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "i": ("i", "me", "my", "mine", "myself"),
    "you (singular)": ("you", "your", "yours", "yourself"),
    "he/she/it/they (singular)": (
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    ),
    "we": ("we", "us", "our", "ours", "ourselves"),
    "you (plural)": ("you", "your", "yours", "yourselves"),
    "they (plural)": ("they", "them", "their", "theirs", "themselves"),
    "this": ("this", "these"),
    "that": ("that", "those"),
    "here": ("here",),
    "there": ("there",),
    "who": ("who", "whom", "whose"),
    "what": ("what",),
    "where": ("where",),
    "when": ("when",),
    "how": ("how",),
    "not": ("not", "never", "no"),
    "all": (
        "all", "everything", "everyone", "everybody", "whole", "entire",
        "every", "each",
    ),
    "many": (
        "many", "much", "lots", "lot", "plenty", "several", "numerous",
        "more", "most",
    ),
    "some": (
        "some", "any", "several", "certain", "someone", "somebody",
        "something",
    ),
    "few": ("few", "little", "less", "least", "hardly", "scarcely"),
    "other": ("other", "another", "others", "else", "different"),
    "one": ("one", "1", "single"),
    "two": ("two", "2", "both", "pair", "twice"),
    "three": ("three", "3", "thrice"),
    "four": ("four", "4"),
    "five": ("five", "5"),
    "at": ("at",),
    "in": ("in", "inside", "into", "within"),
    "with": ("with", "together", "alongside"),
    "and": ("and", "plus"),
    "if": ("if", "whether", "unless"),
    "because": ("because", "since"),
}

# Content glosses where WordNet alone gives no useful or too narrow sense set.
# Keys are the base gloss: no note, no leading "to ", casefolded.
EXTRA_WORDS: dict[str, tuple[str, ...]] = {
    "big": (
        "huge", "huger", "hugest", "large", "larger", "largest",
        "enormous", "massive", "giant", "gigantic",
    ),
    "blood": ("bleed", "bleeds", "bled", "bleeding"),
    "breathe": (
        "breath", "breathes", "breathed", "breathing",
        "inhale", "inhales", "inhaled", "inhaling",
        "exhale", "exhales", "exhaled", "exhaling",
        "respire", "respires", "respired", "respiring",
    ),
    "correct": ("true", "truer", "truest"),
    "cut": (
        "chop", "chops", "chopped", "chopping",
        "slice", "slices", "sliced", "slicing",
    ),
    "die": ("dead", "death", "deaths", "dying"),
    "earth": ("mud", "muddy", "soil", "soils", "dirt"),
    "far": ("distant", "farther", "farthest", "further", "furthest"),
    "father": ("dad", "daddy", "papa", "pa", "pop"),
    "fear": (
        "afraid", "scared", "frightened", "fearful", "terrified",
    ),
    "fight": (
        "wrestle", "wrestles", "wrestled", "wrestling",
    ),
    "fingernail": ("nail", "nails"),
    "hand": ("manual", "manually"),
    "live": ("life", "lives", "alive"),
    "mother": ("mom", "mum", "mommy", "mummy", "mama", "mamma"),
    "near": ("nearby", "close", "closer", "closest"),
    "sun": ("solar",),
}

# Multiword glosses that WordNet does not list as a lemma.
GLOSS_FALLBACKS: dict[str, tuple[str, ...]] = {
    "tree bark": ("bark",),
    "to lie down": ("lie", "lie down"),
    "human being": ("human being", "human"),
}


def read_labels(path: Path) -> list[str]:
    """Read the English Swadesh labels.

    :param path: File with one label per line.
    :returns: The non-empty labels in file order.
    """
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exception_surfaces(fname: str) -> dict[str, set[str]]:
    """Read a WordNet exception file and invert it.

    :param fname: File name, for example ``verb.exc``.
    :returns: Mapping from a base lemma to its irregular surface forms.
    """
    surfaces: dict[str, set[str]] = collections.defaultdict(set)
    for line in wn.open(fname):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        for base in parts[1:]:
            surfaces[base].add(parts[0])
    return surfaces

def inflections(
    base: str, pos: str, exceptions: dict[str, set[str]]
) -> set[str]:
    """Give the regular and irregular forms of one base word.

    :param base: Base word.
    :param pos: WordNet part of speech.
    :param exceptions: Mapping from a base lemma to its irregular forms.
    :returns: Lowercase surface forms.
    """
    forms = {base}
    if pos == wn.NOUN:
        forms |= {base + "s", base + "es", base + "'s"}
        if base.endswith("y") and base[-2:-1] not in "aeiou":
            forms.add(base[:-1] + "ies")
        if base.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(base + "es")
    elif pos == wn.VERB:
        forms |= {base + "s", base + "es", base + "ed", base + "ing"}
        if base.endswith("e"):
            forms |= {base + "d", base[:-1] + "ing"}
        if base.endswith("y") and base[-2:-1] not in "aeiou":
            forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    elif pos == wn.ADJ:
        forms |= {base + "er", base + "est"}
        if base.endswith("y"):
            forms |= {base[:-1] + "ier", base[:-1] + "iest"}
        if base.endswith("e"):
            forms |= {base + "r", base + "st"}
    return forms | exceptions.get(base, set())


def lemma_forms(
    lemma, exceptions: dict[str, set[str]]
) -> tuple[set[str], str | None]:
    """Give the surface forms of one WordNet lemma.

    :param lemma: WordNet lemma.
    :param exceptions: Irregular forms by base lemma.
    :returns: Single-word forms, and the lemma text when it holds a space.
    """
    text = lemma.name().lower().replace("_", " ")
    if " " in text:
        return set(), text
    return inflections(text, lemma.synset().pos(), exceptions), None

def expand(
    synset,
    exceptions: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """Expand one synset to surface words and multiword lemmas.

    The expansion adds the synonym lemmas of the synset and their
    derivationally related forms. It does not add the hypernyms, the
    hyponyms, or the similar synsets: those relations spread too wide.

    :param synset: WordNet synset.
    :param exceptions: Irregular forms by base lemma.
    :returns: Single-word forms and multiword lemmas.
    """
    words: set[str] = set()
    phrases: set[str] = set()
    for lemma in synset.lemmas():
        forms, phrase = lemma_forms(lemma, exceptions)
        words |= forms
        if phrase is not None:
            phrases.add(phrase)
        for related in lemma.derivationally_related_forms():
            forms, phrase = lemma_forms(related, exceptions)
            words |= forms
            if phrase is not None:
                phrases.add(phrase)
    return words, phrases


def gloss_forms(
    gloss: str, exceptions: dict[str, set[str]]
) -> tuple[set[str], set[str]]:
    """Give the English forms that show the sense of one gloss.

    :param gloss: English Swadesh gloss.
    :param exceptions: Irregular forms by base lemma.
    :returns: Single-word surface forms and multiword lemmas.
    """
    key = gloss.casefold()
    if key in FUNCTION_WORDS:
        return set(FUNCTION_WORDS[key]), set()
    base = re.sub(r"\(.*?\)", "", gloss).strip().casefold()
    if base.startswith("to "):
        base = base[3:]
        pos = wn.VERB
    elif "(noun)" in gloss:
        pos = wn.NOUN
    else:
        pos = None
    words: set[str] = set()
    phrases: set[str] = set()
    for part in base.split("/"):
        part = part.strip()
        if not part:
            continue
        synsets = []
        for text in (part, part.replace(" ", "_")):
            synsets += wn.synsets(text, pos=pos) if pos else wn.synsets(text)
        for synset in synsets:
            found_words, found_phrases = expand(synset, exceptions)
            words |= found_words
            phrases |= found_phrases
    if " " in base:
        phrases.add(base)
    else:
        words.add(base)
    words |= set(EXTRA_WORDS.get(base, ()))
    for fallback in GLOSS_FALLBACKS.get(base, ()):
        if " " in fallback:
            phrases.add(fallback)
        else:
            words.add(fallback)
    return words, phrases


def phrase_tokens(
    phrase: str, exceptions: dict[str, set[str]]
) -> tuple[frozenset[str], ...]:
    """Give the inflection set of each token of a multiword lemma.

    :param phrase: Lowercase lemma text with spaces.
    :param exceptions: Irregular forms by base lemma.
    :returns: One token set per word of the phrase.
    """
    return tuple(
        frozenset(inflections(token, wn.VERB, exceptions) | inflections(token, wn.NOUN, exceptions))
        for token in phrase.split()
    )


def build_senses(
    labels: Iterable[str], exceptions: dict[str, set[str]]
) -> dict[str, dict[str, list]]:
    """Build the sense entry of every label.

    :param labels: English Swadesh labels.
    :param exceptions: Irregular forms by base lemma.
    :returns: Mapping from a casefolded gloss to words and phrase tokens.
    """
    table: dict[str, dict[str, list]] = {}
    for label in labels:
        words, phrases = gloss_forms(label, exceptions)
        token_phrases = [phrase_tokens(phrase, exceptions) for phrase in sorted(phrases)]
        token_phrases = [phrase for phrase in token_phrases if len(phrase) > 1]
        table[label.casefold()] = {
            "words": sorted(words),
            "phrases": [[sorted(token) for token in phrase] for phrase in token_phrases],
        }
    return table


def main() -> None:
    """Write the English sense table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    try:
        wn.get_version()
    except LookupError:
        import nltk

        nltk.download("wordnet")

    exceptions = {
        "noun": exception_surfaces("noun.exc"),
        "verb": exception_surfaces("verb.exc"),
        "adj": exception_surfaces("adj.exc"),
    }
    merged: dict[str, set[str]] = collections.defaultdict(set)
    for forms in exceptions.values():
        for base, surfaces in forms.items():
            merged[base] |= surfaces

    labels = read_labels(args.labels)
    senses = build_senses(labels, merged)
    empty = [label for label in labels if not senses[label.casefold()]["words"]]
    data = {
        "source": "nltk WordNet",
        "wordnet_version": wn.get_version(),
        "labels": len(labels),
        "glosses": senses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    forms = sum(len(entry["words"]) + len(entry["phrases"]) for entry in senses.values())
    print(f"wrote {args.out}: {len(labels)} glosses, {forms} forms")
    if empty:
        print(f"glosses with no word: {', '.join(empty)}")


if __name__ == "__main__":
    main()
