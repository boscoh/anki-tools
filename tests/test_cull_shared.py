"""Test that the shared cull flow carries no French value.

The test builds a second :class:`anki_tools.ranking.cullflow.CullProfile` with
a different model name, field names, and deck name, and runs the full cull on a
tiny package. The output deck must use those values.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import cullflow

SENTENCES = [
    ("kat ser stor.", "cat is big."),
    ("kat ser liten.", "cat is small."),
    ("kat ser glad.", "cat is glad."),
]
FIELDS = ["Text", "Audio", "Meaning"]
MODEL_NAME = "Test Sentences"
DECK_NAME = "Test Two Sentences"


def _tokens(texts: list[str]) -> list[list[tuple[str, str]]]:
    return [
        [(word.strip("."), "NOUN") for word in text.split() if word.strip(".")]
        for text in texts
    ]


def _profile() -> cullflow.CullProfile:
    return cullflow.CullProfile(
        tokenize_texts=_tokens,
        load_frequency=lambda: [
            cullflow.FrequencyRow(
                word="kat", tatoeba_count=10, tatoeba_pm=10.0, written_pm=10.0, zipf=4.0
            ),
            cullflow.FrequencyRow(
                word="ser", tatoeba_count=10, tatoeba_pm=5.0, written_pm=5.0, zipf=3.0
            ),
        ],
        frequency_ranks=cullflow.frequency_ranks,
        read_names=set,
        concept_forms=lambda concept: {concept},
        content_pos=frozenset({"NOUN"}),
        min_lemma_length=2,
        final_marks=".?!",
        quote_marks='"«»“”',
        internal_marks=",;:",
        model_name=MODEL_NAME,
        sentence_field="Text",
        english_field="Meaning",
        audio_field="Audio",
        clean_sentence=str.strip,
        reserve_fields=lambda card: [
            card.html,
            f"[sound:{card.audio}]",
            card.english,
            "",
        ],
        templates=(("{{Text}}", "{{Meaning}}"),),
        card_css=lambda: ".sentence { color: black; }",
        reserve_pack=None,
        reserve_model_name="Reserve Sentences",
        swadesh_total=0,
        swadesh_exceptions=(),
        frequency_path=Path("data/test_frequency.tsv"),
        names_path=Path("vocab/test_names.txt"),
        tatoeba_path=Path("data/tatoeba/test_sentences.tsv"),
        wordfreq_lang="xx",
        default_deck_name=DECK_NAME,
        licenses={"data": "test license"},
        target_coverage_gate=0.98,
        double_coverage_gate=0.95,
    )


def _tiny_deck(tmp_path: Path) -> Path:
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=FIELDS,
        cards=[
            {"Text": sentence, "Audio": "", "Meaning": english}
            for sentence, english in SENTENCES
        ],
        model_name=MODEL_NAME,
    )
    return source


def test_a_second_profile_drives_every_output_value(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    output = tmp_path / "culled.apkg"

    run = cullflow.run_cull(
        cullflow.CullRequest(
            input_path=source,
            output_path=output,
            max_cards=2,
            target_words=2,
            min_words=2,
            max_words=4,
        ),
        _profile(),
    )

    with AnkiPackage(output) as package:
        models = package.get_models()
        decks = {deck["name"] for deck in package.get_decks().values()}
        fields = [field["name"] for model in models.values() for field in model["flds"]]
        notes = [str(note["flds"]).split("\x1f") for note in package.get_notes()]

    assert {model["name"] for model in models.values()} == {MODEL_NAME}
    assert decks == {DECK_NAME}
    assert fields == FIELDS
    assert run.cards == 2
    assert notes[0][0].startswith("kat ser")
    assert notes[0][2] == "cat is big."

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["model"] == MODEL_NAME
    assert manifest["source"]["deck"] == DECK_NAME
    assert manifest["licenses"] == {"data": "test license"}
    assert "reserve_pack" not in manifest


def _han_profile() -> cullflow.CullProfile:
    return replace(
        _profile(),
        content_pos=frozenset(),
        min_lemma_length=1,
        final_marks="。？！",
        quote_marks="「」『』",
        internal_marks="，：；",
    )


def test_a_han_profile_keeps_one_character_lemmas_and_a_han_final_mark() -> None:
    profile = _han_profile()
    tokens = [[("水", "n")] for _index in range(3)]
    cards = [
        cullflow.InputCard(
            note_id=index, sentence="我 喝 水 。", english="I drink water.", audio="", index=index
        )
        for index in range(3)
    ]

    targets, content = cullflow.build_targets(
        cards,
        tokens,
        {"水": 1},
        set(),
        min_lemma_length=profile.min_lemma_length,
        content_pos=profile.content_pos,
    )

    assert [target.lemma for target in targets] == ["水"]
    assert content == 1
    assert cullflow.filter_reason(cards[0], tokens[0], 1, 12, profile) == ""
    assert cullflow.filter_reason(cards[0], tokens[0], 1, 12, _profile()) == "F4"
