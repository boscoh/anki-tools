"""Lemma repair of the Spanish spaCy model.

The model joins a clitic pronoun to the verb lemma, for example ``hacer él``.
The cull and the highlight match the verb without the pronoun.
"""

from __future__ import annotations

CLITIC_LEMMAS = frozenset(
    {
        "yo",
        "tú",
        "él",
        "ella",
        "nosotros",
        "nosotras",
        "vosotros",
        "vosotras",
        "ellos",
        "ellas",
        "se",
        "me",
        "te",
        "nos",
        "os",
        "le",
        "les",
        "lo",
        "la",
        "los",
        "las",
    }
)


def base_lemma(lemma: str) -> str:
    """Drop the clitic pronoun that the Spanish model appends to a verb lemma.

    :param lemma: Lemma from the Spanish spaCy model.
    :returns: The lemma without its clitic pronoun.
    """
    head, _, tail = lemma.partition(" ")
    return head if tail in CLITIC_LEMMAS else lemma
