"""Tests for the shared token helpers: the single and the batched path."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from anki_tools.sentence_pack import tokens

MODEL = "fake_model"
MARKS = frozenset({",", ".", "?"})


class FakeMorph:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return "" if self.text in MARKS else "VerbForm=Fin"


class FakeToken:
    def __init__(self, text: str, idx: int) -> None:
        self.text = text
        self.lemma_ = text.casefold()
        self.pos_ = "PUNCT" if text in MARKS else "VERB"
        self.morph = FakeMorph(text)
        self.is_punct = text in MARKS
        self.idx = idx


class FakeDoc:
    def __init__(self, text: str) -> None:
        self.tokens = _split(text)

    def __iter__(self) -> Iterator[FakeToken]:
        return iter(self.tokens)


class FakeNlp:
    def __init__(self) -> None:
        self.single_calls: list[str] = []
        self.batch_calls: list[tuple[list[str], int]] = []

    def __call__(self, text: str) -> FakeDoc:
        self.single_calls.append(text)
        return FakeDoc(text)

    def pipe(self, texts, batch_size: int):
        texts = list(texts)
        self.batch_calls.append((texts, batch_size))
        for text in texts:
            yield FakeDoc(text)


def _split(text: str) -> list[FakeToken]:
    found = []
    idx = 0
    for word in text.split():
        found.append(FakeToken(word, idx))
        idx += len(word) + 1
    return found


@pytest.fixture
def nlp(monkeypatch: pytest.MonkeyPatch) -> FakeNlp:
    fake = FakeNlp()
    monkeypatch.setattr(tokens, "load_model", lambda _name: fake)
    return fake


class TestTokenizeMany:
    def test_batched_tokens_match_the_single_path(self, nlp: FakeNlp) -> None:
        texts = ["Papa ist hier.", "Wir, die Kinder.", "Das ist gut?"]

        singles = [tokens.tokenize(MODEL, text) for text in texts]
        batched = tokens.tokenize_many(MODEL, texts)

        assert batched == singles

    def test_one_pipeline_pass_holds_every_text(self, nlp: FakeNlp) -> None:
        texts = ["A b.", "C d."]

        tokens.tokenize_many(MODEL, texts)

        assert nlp.batch_calls == [(texts, tokens.PIPE_BATCH)]
        assert nlp.single_calls == []

    def test_an_empty_run_makes_no_pass(self, nlp: FakeNlp) -> None:
        assert tokens.tokenize_many(MODEL, []) == []
        assert nlp.single_calls == []


class TestTokenCache:
    def test_get_gives_the_tokens_of_one_sentence(self, nlp: FakeNlp) -> None:
        cache = tokens.TokenCache(MODEL)

        result = cache.get("Papa ist hier.")

        assert [token.text for token in result] == ["Papa", "ist", "hier."]
        assert nlp.single_calls == ["Papa ist hier."]

    def test_get_reads_a_warm_sentence_from_the_cache(self, nlp: FakeNlp) -> None:
        cache = tokens.TokenCache(MODEL)
        cache.warm(["Papa ist hier."])

        result = cache.get("Papa ist hier.")

        assert [token.text for token in result] == ["Papa", "ist", "hier."]
        assert nlp.single_calls == []

    def test_warm_tokenizes_a_repeated_sentence_one_time(self, nlp: FakeNlp) -> None:
        cache = tokens.TokenCache(MODEL)

        cache.warm(["a b", "a b", "c d"])

        assert nlp.batch_calls == [(["a b", "c d"], tokens.PIPE_BATCH)]

    def test_warm_skips_the_sentences_the_cache_holds(self, nlp: FakeNlp) -> None:
        cache = tokens.TokenCache(MODEL)
        cache.get("a b")

        cache.warm(["a b", "c d"])

        assert nlp.batch_calls == [(["c d"], tokens.PIPE_BATCH)]

    def test_the_cache_keeps_the_limit(self, nlp: FakeNlp) -> None:
        cache = tokens.TokenCache(MODEL, limit=1)

        cache.warm(["a b", "c d"])

        cache.get("c d")

        assert nlp.single_calls == ["c d"]
