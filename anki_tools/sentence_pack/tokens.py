"""spaCy token helpers of the sentence pipeline.

The model name comes from the language profile. A missing model stops the build
with a message that names the install step.
"""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

PIPE_BATCH = 256
CACHE_VERSION = 1
COMMIT_ROWS = 2000

_CONNECTIONS: dict[Path, sqlite3.Connection] = {}


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

    The disk connection is shared by the path in one process, so a second
    cache over the same file joins the same transaction.

    :param name: spaCy model name.
    :param limit: Maximum count of cached sentences.
    :param path: SQLite file of the disk cache, or None for memory only.
    """

    def __init__(
        self,
        name: str,
        limit: int = 200_000,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.limit = limit
        self.path = Path(path) if path is not None else None
        self._tokens: dict[str, tuple[Token, ...]] = {}
        self._disk: sqlite3.Connection | None = None
        self._pending = 0

    def get(self, sentence: str) -> tuple[Token, ...]:
        """Give the tokens of one sentence, from the memory or the disk cache."""
        tokens = self._tokens.get(sentence)
        if tokens is not None:
            return tokens
        key = _key(sentence)
        tokens = self._read(key)
        if tokens is None:
            tokens = tuple(tokenize(self.name, sentence))
            self._write(key, tokens)
        self._remember(sentence, tokens)
        return tokens

    def warm(self, sentences: Iterable[str]) -> None:
        """Fill the cache for a whole pool.

        A sentence with a disk row is read, not tokenized, so a second build
        skips the pipeline for every sentence of the first.

        :param sentences: Sentence texts of the pool.
        """
        missing: list[str] = []
        for text in dict.fromkeys(sentences):
            if text in self._tokens:
                continue
            tokens = self._read(_key(text))
            if tokens is None:
                missing.append(text)
            else:
                self._remember(text, tokens)
        for text, tokens in zip(missing, tokenize_many(self.name, missing)):
            tokens = tuple(tokens)
            self._write(_key(text), tokens)
            self._remember(text, tokens)
        self.commit()

    def commit(self) -> None:
        """Write the pending rows to the disk cache."""
        if self._disk is not None:
            self._disk.commit()
            self._pending = 0

    def _remember(self, sentence: str, tokens: tuple[Token, ...]) -> None:
        """Cache one result while the cache holds room.

        :param sentence: Sentence text.
        :param tokens: Tokens of the sentence.
        """
        if len(self._tokens) < self.limit:
            self._tokens[sentence] = tokens

    def _connection(self) -> sqlite3.Connection | None:
        """Open the disk cache, and drop its rows when the model changed."""
        if self.path is None:
            return None
        if self._disk is not None:
            return self._disk
        resolved = self.path.resolve()
        connection = _CONNECTIONS.get(resolved)
        if connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(resolved)
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tokens (key TEXT PRIMARY KEY, value BLOB)"
            )
            _CONNECTIONS[resolved] = connection
        header = f"{self.name}|{CACHE_VERSION}"
        row = connection.execute(
            "SELECT value FROM meta WHERE name = 'header'"
        ).fetchone()
        if row is None:
            connection.execute("INSERT INTO meta VALUES ('header', ?)", (header,))
        elif row[0] != header:
            connection.execute("DELETE FROM tokens")
            connection.execute(
                "UPDATE meta SET value = ? WHERE name = 'header'", (header,)
            )
        connection.commit()
        self._disk = connection
        return connection

    def _read(self, key: str) -> tuple[Token, ...] | None:
        """Read one cached token list.

        :param key: Sentence key.
        :returns: The tokens, or None when the row is absent.
        """
        connection = self._connection()
        if connection is None:
            return None
        row = connection.execute(
            "SELECT value FROM tokens WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return tuple(pickle.loads(zlib.decompress(row[0])))

    def _write(self, key: str, tokens: tuple[Token, ...]) -> None:
        """Store one token list, and commit every :data:`COMMIT_ROWS` rows.

        :param key: Sentence key.
        :param tokens: Tokens of the sentence.
        """
        connection = self._connection()
        if connection is None:
            return
        value = zlib.compress(
            pickle.dumps(list(tokens), protocol=pickle.HIGHEST_PROTOCOL), 6
        )
        connection.execute("INSERT OR REPLACE INTO tokens VALUES (?, ?)", (key, value))
        self._pending += 1
        if self._pending >= COMMIT_ROWS:
            self.commit()


def _key(sentence: str) -> str:
    """Give the disk key of a sentence.

    :param sentence: Sentence text.
    :returns: SHA-256 hex digest of the text.
    """
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


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
