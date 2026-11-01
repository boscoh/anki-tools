"""A fake spaCy model, the one edge of the token helpers.

``nlp`` replaces ``tokens.load_model`` and records the texts that the real
``tokenize`` and ``tokenize_many`` send to the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import SimpleNamespace

import pytest
import spacy

from anki_tools.sentence_pack import tokens

MARKS = frozenset({",", ".", "?"})


def split_words(text: str) -> list[SimpleNamespace]:
    """Split a text on spaces into spaCy-shaped tokens."""
    found = []
    idx = 0
    for word in text.split():
        found.append(
            SimpleNamespace(
                text=word,
                lemma_=word.casefold(),
                pos_="PUNCT" if word in MARKS else "VERB",
                morph="" if word in MARKS else "VerbForm=Fin",
                is_punct=word in MARKS,
                is_alpha=word.isalpha(),
                idx=idx,
            )
        )
        idx += len(word) + 1
    return found


class FakeNlp:
    def __init__(self) -> None:
        self.single_calls: list[str] = []
        self.batch_calls: list[tuple[list[str], int]] = []

    @property
    def tokenized(self) -> list[str]:
        """Every text that went through the pipeline, single or batched."""
        return self.single_calls + [t for texts, _ in self.batch_calls for t in texts]

    def __call__(self, text: str) -> list[SimpleNamespace]:
        self.single_calls.append(text)
        return split_words(text)

    def pipe(self, texts: Iterable[str], batch_size: int) -> Iterator[list[SimpleNamespace]]:
        texts = list(texts)
        self.batch_calls.append((texts, batch_size))
        for text in texts:
            yield split_words(text)


@pytest.fixture
def fake_spacy_load(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeNlp]:
    """Replace ``spacy.load``, for the callers that bind ``load_model`` at import."""
    fake = FakeNlp()
    tokens.load_model.cache_clear()
    monkeypatch.setattr(spacy, "load", lambda _name, **_kwargs: fake)
    yield fake
    tokens.load_model.cache_clear()


@pytest.fixture
def nlp(monkeypatch: pytest.MonkeyPatch) -> FakeNlp:
    fake = FakeNlp()
    monkeypatch.setattr(tokens, "load_model", lambda _name: fake)
    return fake
