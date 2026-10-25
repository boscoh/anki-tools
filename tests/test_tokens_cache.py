"""Tests of the token cache.

A spaCy model takes about 2 seconds to load and about 1 ms per sentence, so a
second build must read the tokens from the disk cache instead of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.sentence_pack import tokens as token_module


def _fake_tokens(text: str) -> list[token_module.Token]:
    return [
        token_module.Token(
            text=word,
            lemma=word.lower(),
            pos="NOUN",
            morph="",
            punctuation=False,
            idx=index,
        )
        for index, word in enumerate(text.split())
    ]


class _Counter:
    """Fake tokenizers that count their calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, _name: str, text: str) -> list[token_module.Token]:
        self.calls.append(text)
        return _fake_tokens(text)

    def tokenize_many(
        self, _name: str, texts
    ) -> list[list[token_module.Token]]:
        texts = list(texts)
        self.calls.extend(texts)
        return [_fake_tokens(text) for text in texts]


@pytest.fixture
def counter(monkeypatch: pytest.MonkeyPatch) -> _Counter:
    """Replace the spaCy calls with a counter."""
    fake = _Counter()
    monkeypatch.setattr(token_module, "tokenize", fake.tokenize)
    monkeypatch.setattr(token_module, "tokenize_many", fake.tokenize_many)
    return fake


def test_the_second_build_reads_the_disk(
    tmp_path: Path, counter: _Counter
) -> None:
    """A warm run tokenizes, a second run reads the rows."""
    path = tmp_path / "tokens.sqlite"
    sentences = ["le chat dort", "le chien court"]

    first = token_module.TokenCache("fr_test", path=path)
    first.warm(sentences)
    assert sorted(counter.calls) == sorted(sentences)

    second = token_module.TokenCache("fr_test", path=path)
    second.warm(sentences)
    assert len(counter.calls) == len(sentences)
    assert second.get("le chat dort") == tuple(_fake_tokens("le chat dort"))


def test_a_new_model_drops_the_rows(tmp_path: Path, counter: _Counter) -> None:
    """A different model name invalidates the cache."""
    path = tmp_path / "tokens.sqlite"
    sentences = ["le chat dort"]

    token_module.TokenCache("fr_test", path=path).warm(sentences)
    assert len(counter.calls) == 1

    token_module.TokenCache("fr_test_v2", path=path).warm(sentences)
    assert len(counter.calls) == 2


def test_get_falls_back_to_the_pipeline(tmp_path: Path, counter: _Counter) -> None:
    """A sentence that the disk does not hold goes to the tokenizer once."""
    path = tmp_path / "tokens.sqlite"
    cache = token_module.TokenCache("fr_test", path=path)

    assert cache.get("une phrase") == tuple(_fake_tokens("une phrase"))
    assert cache.get("une phrase") == tuple(_fake_tokens("une phrase"))
    assert counter.calls == ["une phrase"]


def test_a_cache_without_a_path_keeps_the_memory_only(counter: _Counter) -> None:
    """A cache without a path never writes a file."""
    cache = token_module.TokenCache("fr_test")

    cache.warm(["le chat dort"])
    cache.warm(["le chat dort"])

    assert counter.calls == ["le chat dort"]
