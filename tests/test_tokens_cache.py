"""Tests of the disk token cache.

A spaCy model takes about 2 seconds to load and about 1 ms per sentence, so a
second build must read the tokens from the disk cache instead of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from anki_tools.sentence_pack import tokens
from tests.token_fixture import FakeNlp, nlp  # noqa: F401

SENTENCES = ["le chat dort", "le chien court"]


def test_the_second_build_reads_the_disk(tmp_path: Path, nlp: FakeNlp) -> None:
    path = tmp_path / "tokens.sqlite"
    tokens.TokenCache("fr_test", path=path).warm(SENTENCES)
    assert sorted(nlp.tokenized) == sorted(SENTENCES)

    second = tokens.TokenCache("fr_test", path=path)
    second.warm(SENTENCES)

    assert len(nlp.tokenized) == len(SENTENCES)
    assert [token.text for token in second.get(SENTENCES[0])] == ["le", "chat", "dort"]


def test_a_new_model_drops_the_rows(tmp_path: Path, nlp: FakeNlp) -> None:
    path = tmp_path / "tokens.sqlite"

    tokens.TokenCache("fr_test", path=path).warm(SENTENCES[:1])
    tokens.TokenCache("fr_test_v2", path=path).warm(SENTENCES[:1])

    assert len(nlp.tokenized) == 2


def test_two_caches_over_one_file_can_write(tmp_path: Path, nlp: FakeNlp) -> None:
    path = tmp_path / "tokens.sqlite"
    first = tokens.TokenCache("fr_test", path=path)
    second = tokens.TokenCache("fr_test", path=path)

    first.get("le chat dort")
    second.get("le chien court")
    first.get("une phrase")
    second.get("une autre")

    assert nlp.tokenized == ["le chat dort", "le chien court", "une phrase", "une autre"]
