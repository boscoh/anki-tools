"""spaCy token helpers of the sentence pipeline.

The model name comes from the language profile. A missing model stops the build
with a message that names the install step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class Token:
    """One word token of a sentence.

    :param text: Token text.
    :param lemma: Lemma from the spaCy lemmatizer.
    :param pos: Universal part-of-speech tag.
    :param morph: Morphology string, for example ``VerbForm=Fin``.
    :param punctuation: True for a punctuation token.
    :param idx: Character offset of the token in the source text.
    """

    text: str
    lemma: str
    pos: str
    morph: str
    punctuation: bool
    idx: int = 0


@lru_cache(maxsize=4)
def load_model(name: str) -> Any:
    """Load a spaCy model once, without the parser and the entity tagger.

    :param name: spaCy model name, for example ``el_core_news_sm``.
    :raises RuntimeError: The model is not installed.
    :returns: The loaded pipeline.
    """
    import spacy

    try:
        return spacy.load(name, exclude=["parser", "ner"])
    except OSError as error:
        raise RuntimeError(
            f"spaCy model {name} is not installed. Install it from the spaCy "
            "model release (see README) and run the build again."
        ) from error


def tokenize(name: str, text: str) -> list[Token]:
    """Tokenize a sentence with a spaCy model.

    :param name: spaCy model name.
    :param text: Sentence text.
    :returns: One token per word and punctuation mark, in sentence order.
    """
    doc = load_model(name)(text)
    return [
        Token(
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            morph=str(token.morph),
            punctuation=token.is_punct,
            idx=token.idx,
        )
        for token in doc
    ]


def words(name: str, text: str) -> list[str]:
    """Give the word tokens of a sentence.

    :param name: spaCy model name.
    :param text: Sentence text.
    :returns: Token texts with the punctuation tokens removed.
    """
    return [token.text for token in tokenize(name, text) if not token.punctuation]


def has_finite_verb(tokens: list[Token]) -> bool:
    """Check the predicate rule: a finite verb or auxiliary.

    :param tokens: Tokens of the sentence.
    :returns: True when one token is a finite verb.
    """
    return any(
        token.pos in ("VERB", "AUX") and "VerbForm=Fin" in token.morph
        for token in tokens
    )
