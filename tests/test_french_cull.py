"""Tests for the French 1000-card cull (spec sections 10 and 11).

Every test is offline. The token lists are built by hand, so no test loads
ffprobe or the Tatoeba corpus. The frequency and gate tests run the real spaCy
French model on a few sentences.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.languages.french import cull as cull_fr
from anki_tools.ranking import cullflow
from anki_tools.ranking.deckreader import InputCard
from tests.token_fixture import fake_spacy_load  # noqa: F401

PROFILE = cull_fr.FRENCH_CULL_PROFILE


def test_the_loader_builds_the_file_once_from_the_corpus(tmp_path: Path) -> None:
    frequency_path = tmp_path / "french_frequency.tsv"
    tatoeba_path = tmp_path / "fra_sentences.tsv"
    tatoeba_path.write_text(
        "1\tfra\tLe chat mange.\n2\tfra\tLe chat dort.\n", encoding="utf-8"
    )

    rows = cull_fr.load_or_build_frequency(frequency_path, tatoeba_path)
    tatoeba_path.unlink()
    again = cull_fr.load_or_build_frequency(frequency_path, tatoeba_path)

    assert rows
    assert again == rows


def test_the_loader_needs_the_corpus_when_the_file_is_absent(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        cull_fr.load_or_build_frequency(tmp_path / "f.tsv", tmp_path / "t.tsv")


def test_read_names_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "names.txt"
    path.write_text("# a comment\n\nParis\n  zoe \n", encoding="utf-8")

    assert cullflow.read_names(path) == {"paris", "zoe"}


def test_the_reserve_fields_fit_the_output_note_type() -> None:
    card = cullflow.ReserveCard(
        sentence="le chat mange",
        html="le <b>chat</b> mange",
        english="the cat eats",
        target="chat",
        gloss="cat",
        audio="a.mp3",
    )

    assert PROFILE.reserve_fields(card) == [
        "[sound:a.mp3]",
        "le <b>chat</b> mange",
        "the cat eats",
        "chat",
        "cat",
    ]


def test_swadesh_coverage_counts_a_word_from_another_card() -> None:
    profile = replace(PROFILE, concept_forms=lambda concept: {concept})
    sentence = "Un nuage flottait dans le ciel."
    tokens = [[("nuage", "NOUN"), ("flotter", "VERB")]]

    covered, _total = cullflow.swadesh_coverage(
        ["nuage", "flotter", "intestins"], [sentence], tokens, profile
    )

    assert covered == 2


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        pytest.param("Elle mit le linge sale dans la machine à laver.", "register", id="mit"),
        pytest.param("Leur voyage fut annulé en raison de la pluie.", "register", id="fut"),
        pytest.param("Elle a mis le linge sale.", "", id="passé composé"),
    ],
)
def test_the_register_rule_rejects_a_passe_simple(sentence: str, expected: str) -> None:
    card = InputCard(
        note_id=0, sentence=sentence, english="", audio="", index=0
    )

    assert cullflow.filter_reason(card, PROFILE) == expected


@pytest.mark.parametrize(
    ("word", "lemma"), [("veux", "vouloir"), ("fais", "faire"), ("chat", "chat")]
)
@pytest.mark.usefixtures("fake_spacy_load")
def test_the_tokenizer_repairs_a_lemma_that_the_model_leaves_alone(
    word: str, lemma: str
) -> None:
    assert cull_fr.tokenize_texts([word]) == [[(lemma, "VERB")]]
