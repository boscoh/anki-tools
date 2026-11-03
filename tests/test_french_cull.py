"""Tests for the French 1000-card cull (spec sections 10 and 11).

Every test is offline. The token lists are built by hand, so no test loads
ffprobe or the Tatoeba corpus. The frequency and gate tests run the real spaCy
French model on a few sentences.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.languages.french import cull as cull_fr
from anki_tools.ranking import (
    assemble,
    coreflow,
    cullflow,
)
from anki_tools.ranking.identity import DeckIdentity
from tests.cull_fixture import cull_deck, tiny_deck
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


def _write_cull_deck(
    source: Path,
    cards: list[cull_fr.InputCard],
    kept: list[int],
    output: Path,
    *,
    reserve: list[cullflow.ReserveCard] = [],
    reserve_pack_path: Path | None = None,
) -> dict[str, int]:
    identity = DeckIdentity(output)
    values, _bolded, _unbolded = coreflow.note_values(source, cards, kept, PROFILE, {})
    return write_package(
        source,
        output,
        OutputShape(
            deck_name=identity.title,
            model_name=identity.model_name,
            input_model_name=PROFILE.input_model_name,
            fields=assemble.output_fields(PROFILE),
            templates=assemble.deck_templates(PROFILE),
            css=PROFILE.card_css(),
        ),
        values,
        reserve=reserve,
        reserve_pack_path=reserve_pack_path,
        reserve_values=PROFILE.reserve_fields,
    )


SWADESH_FIELDS = [
    "Audio",
    "Sentence",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
]


def _tiny_swadesh_deck(tmp_path: Path) -> Path:
    rows = [
        {
            "Audio": "[sound:tts_reserve.mp3]",
            "Sentence": "<b>Je</b> suis ici.",
            "English": "I am here.",
            "Target": "je",
            "Gloss": "I",
            "WordAudio": "[sound:word_reserve.mp3]",
            "Source": "Tatoeba #1",
        }
    ]
    return tiny_deck(
        tmp_path,
        SWADESH_FIELDS,
        rows,
        cull_fr.SWADESH_MODEL_NAME,
        ["tts_reserve.mp3", "word_reserve.mp3"],
        filename="swadesh.apkg",
    )


def test_write_deck_inserts_the_reserve_card(tmp_path: Path) -> None:
    source = cull_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    swadesh = _tiny_swadesh_deck(tmp_path)
    reserve = cullflow.load_reserve(swadesh, cull_fr.SWADESH_MODEL_NAME)
    output = tmp_path / "merged.apkg"

    counts = _write_cull_deck(
        source, cards, [0, 1], output, reserve=reserve, reserve_pack_path=swadesh
    )

    with AnkiPackage(output) as package:
        notes = [dict(note) for note in package.get_notes()]
        media = set(package.get_media_mapping().values())
        models = {model.get("name") for model in package.get_models().values()}
    fields = [str(note["flds"]).split("\x1f") for note in notes]
    reserve_row = next(row for row in fields if "<b>Je</b>" in row[1])
    assert counts == {"notes": 3, "media": 3, "items": 3}
    assert media == {"a0.mp3", "a1.mp3", "tts_reserve.mp3"}
    assert models == {DeckIdentity(output).title}
    assert reserve_row[:1] + reserve_row[2:5] == [
        "[sound:tts_reserve.mp3]",
        "I am here.",
        "je",
        "I",
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
    card = cull_fr.InputCard(
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
