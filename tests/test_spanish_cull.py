"""Test for the Spanish cull output note type."""

from __future__ import annotations

import pytest

from anki_tools.languages.spanish import cull as cull_es
from anki_tools.languages.spanish.lemma import base_lemma
from anki_tools.ranking import cullflow


def test_the_reserve_fields_fit_the_output_note_type() -> None:
    """The reserve card fills the five output fields in the output order."""
    card = cullflow.ReserveCard(
        sentence="el gato come",
        html="el <b>gato</b> come",
        english="the cat eats",
        target="gato",
        gloss="cat",
        audio="a.mp3",
    )
    fields = cull_es.SPANISH_CULL_PROFILE.reserve_fields(card)

    assert fields == [
        "[sound:a.mp3]",
        "el <b>gato</b> come",
        "the cat eats",
        "gato",
        "cat",
    ]


@pytest.mark.parametrize(
    ("lemma", "base"),
    [("hacer \u00e9l", "hacer"), ("hacer", "hacer"), ("hacer algo", "hacer algo")],
)
def test_a_trailing_clitic_pronoun_leaves_the_lemma(lemma: str, base: str) -> None:
    assert base_lemma(lemma) == base
