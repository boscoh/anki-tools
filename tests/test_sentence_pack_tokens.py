"""Tests for the shared token helpers: the single and the batched path."""

from __future__ import annotations

from anki_tools.sentence_pack import tokens
from tests.token_fixture import FakeNlp, nlp  # noqa: F401

MODEL = "fake_model"


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
