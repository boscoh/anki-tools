"""spaCy token helpers of the sentence pipeline.

The model name comes from the language profile. A missing model stops the build
with a message that names the install step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

PIPE_BATCH = 256


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
    return [_convert(token) for token in load_model(name)(text)]


def tokenize_many(name: str, texts: Iterable[str]) -> list[list[Token]]:
    """Tokenize sentences in one pipeline pass.

    The pass pays the pipeline overhead once. A call per sentence costs 1 to
    3 ms each on top of the parse.

    :param name: spaCy model name.
    :param texts: Sentence texts.
    :returns: One token list per text, in the given order.
    """
    nlp = load_model(name)
    return [
        [_convert(token) for token in doc]
        for doc in nlp.pipe(texts, batch_size=PIPE_BATCH)
    ]


def _convert(token: Any) -> Token:
    """Convert a spaCy token to the shared record.

    :param token: spaCy token.
    :returns: The token record.
    """
    return Token(
        text=token.text,
        lemma=token.lemma_,
        pos=token.pos_,
        morph=str(token.morph),
        punctuation=token.is_punct,
        idx=token.idx,
    )


class TokenCache:
    """Per-sentence token cache that a batch pass can fill.

    The guards read one sentence at a time, so the cache keeps the per-sentence
    result. :meth:`warm` fills it from a whole pool with one pipeline pass.

    :param name: spaCy model name.
    :param limit: Maximum count of cached sentences.
    """

    def __init__(self, name: str, limit: int = 200_000) -> None:
        self.name = name
        self.limit = limit
        self._tokens: dict[str, tuple[Token, ...]] = {}

    def get(self, sentence: str) -> tuple[Token, ...]:
        """Give the tokens of one sentence, and cache them.

        :param sentence: Sentence text.
        :returns: Tokens of the sentence.
        """
        tokens = self._tokens.get(sentence)
        if tokens is None:
            tokens = tuple(tokenize(self.name, sentence))
            self._remember(sentence, tokens)
        return tokens

    def warm(self, sentences: Iterable[str]) -> None:
        """Tokenize the sentences that the cache does not hold, in one pass.

        :param sentences: Sentence texts of the pool.
        """
        unique = dict.fromkeys(sentences)
        missing = [text for text in unique if text not in self._tokens]
        for sentence, tokens in zip(missing, tokenize_many(self.name, missing)):
            self._remember(sentence, tuple(tokens))

    def _remember(self, sentence: str, tokens: tuple[Token, ...]) -> None:
        """Cache one result while the cache holds room.

        :param sentence: Sentence text.
        :param tokens: Tokens of the sentence.
        """
        if len(self._tokens) < self.limit:
            self._tokens[sentence] = tokens


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
