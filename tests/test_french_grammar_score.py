"""Tests of the French grammar score.

The spaCy model load takes about 2 seconds, so the score must load it one time
for a whole deck and not one time for each sentence.
"""

from __future__ import annotations

import pytest

from anki_tools.languages.french import rank


class _Token:
    """A token that is its own head, with the tags that the score reads."""

    def __init__(self) -> None:
        self.head = self
        self.pos_ = "VERB"
        self.dep_ = "nsubj"


class _Doc:
    """A minimal document that holds one token."""

    def __iter__(self):
        yield _Token()

    def __bool__(self) -> bool:
        return True


def test_grammar_score_loads_the_model_one_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sentences give one model load."""
    spacy = pytest.importorskip("spacy")
    calls: list[str] = []

    def fake_load(name: str, **_kwargs) -> object:
        calls.append(name)
        return lambda _text: _Doc()

    monkeypatch.setattr(spacy, "load", fake_load)
    rank._load_grammar_nlp.cache_clear()

    assert rank.grammar_score_fr("Le chat dort.") > 0
    assert rank.grammar_score_fr("Le chien court.") > 0
    assert calls == ["fr_core_news_sm"]

    rank._load_grammar_nlp.cache_clear()


def test_grammar_score_keeps_the_marker_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing model gives the marker score, not an error."""
    rank._load_grammar_nlp.cache_clear()
    monkeypatch.setattr(rank, "_load_grammar_nlp", lambda: None)

    assert rank.grammar_score_fr("Il faut que tu partes, qui que tu sois.") > 0
