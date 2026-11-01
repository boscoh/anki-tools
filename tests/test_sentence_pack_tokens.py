"""Tests for the shared token helpers: the single and the batched path."""

from __future__ import annotations

from anki_tools.sentence_pack import tokens
from tests.token_fixture import FakeNlp, nlp  # noqa: F401

MODEL = "fake_model"


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
