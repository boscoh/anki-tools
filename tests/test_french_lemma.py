"""Tests for the French lemma overrides of the cull tokenizer."""

from __future__ import annotations

import pytest

from anki_tools.languages.french import cull as cull_fr


@pytest.mark.parametrize(
    ("sentence", "surface", "lemma"),
    [
        ("Je veux partir.", "veux", "vouloir"),
        ("Tu peux venir.", "peux", "pouvoir"),
        ("Il prend le train.", "prend", "prendre"),
        ("C'est un fait.", "fait", "fait"),
    ],
)
def test_a_surface_form_gets_its_lemma(sentence: str, surface: str, lemma: str) -> None:
    lemmas = [token[0] for token in cull_fr.tokenize_texts([sentence])[0]]

    assert lemma in lemmas
    assert surface == lemma or surface not in lemmas
