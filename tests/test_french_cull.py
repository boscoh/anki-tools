"""Tests for the French 1000-card cull (spec sections 10 and 11).

Every test is offline. The token lists are built by hand, so no test loads
ffprobe or the Tatoeba corpus. The frequency and gate tests run the real spaCy
French model on a few sentences.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.languages.french import cull as cull_fr
from anki_tools.languages.french.profile import FRENCH
from anki_tools.ranking import (
    assemble,
    cardfilter,
    coreflow,
    cullflow,
    cullgates,
)
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.sentence_pack import pool
from tests.cull_fixture import card, freq_row, tiny_deck
from tests.deck_fixture import _add_second_template

PROFILE = cull_fr.FRENCH_CULL_PROFILE
DECK_FIELDS = ["French", "sound", "Back", "freq ranks"]
DECK_SENTENCES = [
    "Le chat aime la fille.",
    "Le chat voit le chien.",
    "Le chat aime la maison.",
]
DECK_ENGLISH = [
    "The cat likes the girl.",
    "The cat sees the dog.",
    "The cat likes the house.",
]


def test_the_spoken_weight_changes_the_order() -> None:
    rows = [
        freq_row("parle", pm=100.0, written=1.0),
        freq_row("ecrit", pm=1.0, written=100.0),
    ]

    assert cullflow.frequency_ranks(rows, spoken_weight=0.0) == {"ecrit": 1, "parle": 2}
    assert cullflow.frequency_ranks(rows, spoken_weight=1.0) == {"parle": 1, "ecrit": 2}


def test_the_loader_builds_the_file_once_from_the_corpus(tmp_path: Path) -> None:
    frequency_path = tmp_path / "french_frequency.tsv"
    tatoeba_path = tmp_path / "fra_sentences.tsv"
    tatoeba_path.write_text(
        "1\tfra\tLe chat mange.\n2\tfra\tLe chat dort.\n", encoding="utf-8"
    )

    rows = cull_fr.load_or_build_frequency(frequency_path, tatoeba_path)
    tatoeba_path.unlink()
    again = cull_fr.load_or_build_frequency(frequency_path, tatoeba_path)

    assert {row.word: row.tatoeba_count for row in rows}["chat"] == 2
    assert again == rows


def test_the_loader_needs_the_corpus_when_the_file_is_absent(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        cull_fr.load_or_build_frequency(tmp_path / "f.tsv", tmp_path / "t.tsv")


def test_read_names_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "names.txt"
    path.write_text("# a comment\n\nParis\n  zoe \n", encoding="utf-8")

    assert cullflow.read_names(path) == {"paris", "zoe"}


@pytest.mark.parametrize(
    ("rule", "sentence"),
    [
        pytest.param("", "Le chat mange vite.", id="a clean card passes"),
        pytest.param("length_window", "Chat mange.", id="too short"),
        pytest.param("digit_marks", "Le chat a 3 ans.", id="digit"),
        pytest.param("digit_marks", "Le chat [noir] mange.", id="bracket"),
        pytest.param("digit_marks", "Voir le site www.exemple.fr.", id="url"),
        pytest.param("digit_marks", "Le chat mange vite...", id="ellipsis"),
        pytest.param("repeat", "Oui oui oui oui.", id="repeated word"),
        pytest.param("marks", "Le chat mange vite", id="no final mark"),
        pytest.param("marks", "Le chat mange. Le chien dort.", id="two sentences"),
        pytest.param("marks", "Le chat « mange » vite.", id="quotes"),
        pytest.param("marks", "— Le chat mange vite.", id="dialogue dash"),
        pytest.param("marks", "Le chat, le chien, et l'oiseau mangent.", id="commas"),
    ],
)
def test_one_card_per_filter_rule(rule: str, sentence: str) -> None:
    assert cullflow.filter_reason(card(sentence), PROFILE) == rule


@pytest.mark.parametrize(
    "sentence",
    ["Le chat mange vite.", "Chat mange.", "Le chat mange vite", "Oui oui oui."],
)
def test_the_pack_and_the_cull_agree_on_the_shape_rules(sentence: str) -> None:
    keys = cardfilter.LENGTH_RULES + cardfilter.TEXT_RULES + cardfilter.MARK_RULES

    pack_reason = pool.shape_failure(
        sentence, FRENCH, FRENCH.min_units, FRENCH.max_units, keys=keys
    )

    assert (pack_reason or "") == cullflow.filter_reason(card(sentence), PROFILE)


@pytest.mark.parametrize(
    ("sentences", "reasons"),
    [
        pytest.param(
            ["Le chat mange vite.", "Le chien dort bien.", "Le chat mange vite."],
            ["", "", "unique"],
            id="the same text",
        ),
        pytest.param(
            ["Le chat mange vite.", "Le chat mange vite!"],
            ["", "unique"],
            id="a different mark",
        ),
    ],
)
def test_filter_cards_rejects_a_duplicate_text(
    sentences: list[str], reasons: list[str]
) -> None:
    cards = [card(sentence, index=index) for index, sentence in enumerate(sentences)]

    passed, found = cullflow.filter_cards(
        cards,
        card_rule=partial(cullflow.filter_reason, profile=PROFILE),
        dedupe_key=PROFILE.dedupe_key,
    )

    assert found == reasons
    assert passed == [index for index, reason in enumerate(found) if not reason]


def _tiny_deck(tmp_path: Path) -> Path:
    rows = [
        {"French": sentence, "sound": f"[sound:a{index}.mp3]", "Back": english, "freq ranks": ""}
        for index, (sentence, english) in enumerate(zip(DECK_SENTENCES, DECK_ENGLISH))
    ]
    audio = [f"a{index}.mp3" for index in range(len(rows))]
    return tiny_deck(tmp_path, DECK_FIELDS, rows, cull_fr.MODEL_NAME, audio)


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


def _output_model(output: Path) -> dict:
    with AnkiPackage(output) as package:
        models = package.get_models(include_templates=True)
    return next(m for m in models.values() if m.get("name") == DeckIdentity(output).title)


def test_write_deck_keeps_the_notes_media_and_profile_templates(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "output.apkg"

    counts = _write_cull_deck(source, cards, [0, 2], output)

    with AnkiPackage(output) as package:
        media = set(package.get_media_mapping().values())
        decks = {deck.get("name") for deck in package.get_decks().values()}
    model = _output_model(output)
    templates = [(template["qfmt"], template["afmt"]) for template in model["tmpls"]]
    assert counts == {"notes": 2, "media": 2, "items": 2}
    assert media == {"a0.mp3", "a2.mp3"}
    assert decks == {DeckIdentity(output).title}
    assert [card.sentence for card in coreflow.read_output_deck(output)] == [
        DECK_SENTENCES[0],
        DECK_SENTENCES[2],
    ]
    assert templates == list(assemble.deck_templates(PROFILE))
    assert all(selector in model["css"] for selector in (".sentence", ".english", ".hint"))


def test_write_deck_removes_the_second_card_template(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    _add_second_template(source, 1)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "one-card.apkg"

    _write_cull_deck(source, cards, [0, 1], output)

    with AnkiPackage(output) as package:
        rows = package.get_cards()
    assert [template["name"] for template in _output_model(output)["tmpls"]] == ["Card 1"]
    assert {row["ord"] for row in rows} == {0}
    assert len(rows) == 2


def test_the_gates_pass_on_a_tiny_cull(tmp_path: Path) -> None:
    senses = tmp_path / "senses.tsv"
    senses.write_text("word\tsenses\tfalse_friends\nchat\tcat\t\n", encoding="utf-8")
    profile = replace(
        PROFILE, load_frequency=lambda: [freq_row("chat")], sense_path=senses
    )

    run = cullflow.run_cull(
        cullflow.CullRequest(
            input_path=_tiny_deck(tmp_path),
            output_path=tmp_path / "gated.apkg",
            max_cards=3,
            target_words=1,
            no_reserve=True,
        ),
        profile,
    )
    rows = cullgates.run_gates(
        cullgates.GATES, cullgates.CullFiles(run.output_path, profile)
    )

    assert [gate for gate, _passed, _message in rows] == ["G1", "G2", "G5", "G6"]
    for gate, passed, message in rows:
        assert passed, f"{gate}: {message}"


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


def test_load_reserve_reads_the_plain_and_bold_sentence(tmp_path: Path) -> None:
    reserve = cullflow.load_reserve(
        _tiny_swadesh_deck(tmp_path), cull_fr.SWADESH_MODEL_NAME
    )

    assert len(reserve) == 1
    assert reserve[0].sentence == "Je suis ici."
    assert reserve[0].html == "<b>Je</b> suis ici."
    assert reserve[0].target == "je"
    assert reserve[0].audio == "tts_reserve.mp3"


def test_write_deck_inserts_the_reserve_card(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
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
    sentence = "Un nuage flottait dans le ciel."
    tokens = cullflow.tokenize_texts([sentence], model=cull_fr.TOKENIZER)

    covered, total = cullflow.swadesh_coverage(
        ["nuage", "intestins"], [sentence], tokens, PROFILE
    )

    assert covered == 1
    assert total == cull_fr.SWADESH_TOTAL
