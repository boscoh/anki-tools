"""Test that the shared cull flow carries no French value.

The test builds a second :class:`anki_tools.ranking.cullflow.CullProfile` with
a different model name, field names, and deck name, and runs the full cull on a
tiny package. The output deck must use those values. The module also holds the
unit tests of the shared flow.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest

from anki_tools.apkg.deck_writer import RunFiles
from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import corelist, coreflow, cullflow, cullgates
from anki_tools.ranking.identity import DeckIdentity
from tests.cull_fixture import (
    CONTENT_POS,
    ENGLISH,
    card,
    freq_row,
    gate_reader,
    select,
    target,
    tiny_deck,
)
from tests.deck_fixture import write_mp3s

SENTENCES = [
    ("kat ser stor.", "cat is big."),
    ("kat ser liten.", "cat is small."),
    ("kat ser glad.", "cat is glad."),
]
FIELDS = ["Text", "Audio", "Meaning"]
MODEL_NAME = "Test Sentences"
ORDER_TOKENS = [
    [("maison", "NOUN"), ("grande", "ADJ")],
    [("chat", "NOUN"), ("noir", "ADJ")],
    [("et", "DET")],
]
ORDER_RANKS = {"maison": 5, "grande": 40, "chat": 2, "noir": 900}


def _tokens(texts: Iterable[str]) -> list[list[tuple[str, str]]]:
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
        script_ok=lambda sentence: True,
        is_repeated=lambda sentence: False,
        dedupe_key=str,
        unit_count=lambda sentence: len(sentence.split()),
        min_units=2,
        max_units=4,
        input_model_name=MODEL_NAME,
        input_sentence_field="Text",
        input_english_field="Meaning",
        input_audio_field="Audio",
        sense_path=Path("vocab/test_core_senses.tsv"),
        clean_sentence=str.strip,
        reserve_fields=lambda card: [
            f"[sound:{card.audio}]",
            card.html,
            card.english,
            card.target,
            card.gloss,
        ],
        templates=(("{{Audio}}", "{{Sentence}}"),),
        card_css=lambda: ".sentence { color: black; }",
        reserve_pack=None,
        reserve_model_name="Reserve Sentences",
        swadesh_total=0,
        swadesh_exceptions=(),
        frequency_path=Path("data/test_frequency.tsv"),
        wordfreq_lang="xx",
        licenses={"data": "test license"},
        highlight=lambda sentence, word: sentence.replace(
            word, f"<b>{word}</b>", 1
        ),
    )


def _tiny_deck(tmp_path: Path) -> Path:
    rows = [
        {"Text": sentence, "Audio": "[sound:in.mp3]", "Meaning": english}
        for sentence, english in SENTENCES
    ]
    return tiny_deck(tmp_path, FIELDS, rows, MODEL_NAME, ["in.mp3"])


def _senses(tmp_path: Path) -> Path:
    path = tmp_path / "senses.tsv"
    path.write_text("word\tsenses\tfalse_friends\nkat\tcat\t\n", encoding="utf-8")
    return path


def _cull(
    tmp_path: Path,
    profile: cullflow.CullProfile,
    *,
    max_cards: int = 2,
    source: Path | None = None,
    name: str = "culled.apkg",
) -> RunFiles:
    return cullflow.run_cull(
        cullflow.CullRequest(
            input_path=source or _tiny_deck(tmp_path),
            output_path=tmp_path / name,
            max_cards=max_cards,
            target_words=2,
        ),
        profile,
    )


def _notes(path: Path) -> list[list[str]]:
    with AnkiPackage(path) as package:
        return [str(row["flds"]).split("\x1f") for row in package.get_notes()]


def test_a_second_profile_drives_every_output_value(tmp_path: Path) -> None:
    run = _cull(tmp_path, _profile())
    output = run.output_path

    title = DeckIdentity(output).title
    with AnkiPackage(output) as package:
        models = package.get_models()
        decks = {deck["name"] for deck in package.get_decks().values()}
        fields = [field["name"] for model in models.values() for field in model["flds"]]
        cards = package.get_cards()
    notes = _notes(output)

    assert {model["name"] for model in models.values()} == {title}
    assert decks == {title}
    assert fields == ["Audio", "Sentence", "English", "Core", "CoreGloss"]
    assert len(cards) == len(notes) == 2
    assert {int(row["ord"]) for row in cards} == {0}
    assert notes[0][1].startswith("<b>kat</b> ser")
    assert notes[0][2] == "cat is big."
    assert notes[0][3] == "kat"

    assert run.manifest_path is not None
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["model"] == title
    assert manifest["source"]["input_model"] == MODEL_NAME
    assert manifest["source"]["deck"] == title
    assert manifest["licenses"] == {"data": "test license"}
    assert "reserve_pack" not in manifest


def test_a_han_profile_keeps_one_character_lemmas_and_a_han_final_mark() -> None:
    profile = replace(
        _profile(),
        content_pos=frozenset(),
        min_lemma_length=1,
        final_marks="。？！",
        quote_marks="「」『』",
        internal_marks="，：；",
    )
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
    assert cullflow.filter_reason(cards[0], profile) == ""
    assert cullflow.filter_reason(cards[0], _profile()) == "marks"


def _target_fixture() -> tuple[
    list[coreflow.InputCard], list[list[tuple[str, str]]], dict[str, int]
]:
    pairs = (
        [("mange", "NOUN")] * 3
        + [("le", "DET")] * 5
        + [("table", "NOUN")] * 2
        + [("fait", "VERB")] * 3
        + [("fait", "AUX")] * 4
        + [("a", "VERB")] * 3
        + [("paris", "PROPN")] * 3
        + [("chien", "NOUN")] * 4
    )
    tokens = [pairs[start : start + 3] for start in range(0, len(pairs), 3)]
    cards = [card("Le chat mange ici.", index=index) for index in range(len(tokens))]
    return cards, tokens, {"mange": 10, "chien": 2, "le": 1}


def _cap_fixture() -> tuple[
    list[coreflow.InputCard],
    list[list[tuple[str, str]]],
    list[corelist.TargetWord],
    dict[str, int],
]:
    sentences = [
        "Le chat est ici.",
        "Le chat est la.",
        "Le chat est seul.",
        "Le chat est noir.",
        "Le chat est grand.",
    ]
    cards = [card(sentence, index=index) for index, sentence in enumerate(sentences)]
    tokens = [
        [("chat", "NOUN"), (f"f{index}", "NOUN")] for index in range(len(sentences))
    ]
    return cards, tokens, [target("chat", 1)], {"chat": 1}


@pytest.mark.parametrize(
    ("count", "pm", "written", "spoken_weight", "expected"),
    [
        pytest.param(4, 0.0, 12.5, 1.0, 12.5, id="below five, spoken weight 1"),
        pytest.param(4, 0.0, 12.5, 0.0, 12.5, id="below five, spoken weight 0"),
        pytest.param(10, 20.0, 10.0, 0.5, 15.0, id="middle weight uses both terms"),
    ],
)
def test_the_blended_pm(
    count: int, pm: float, written: float, spoken_weight: float, expected: float
) -> None:
    row = freq_row("chat", count=count, pm=pm, written=written)

    assert cullflow.blend_pm(row, spoken_weight=spoken_weight) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("target_words", "expected", "content"),
    [
        pytest.param(300, [("chien", 2), ("mange", 10)], 2, id="every candidate"),
        pytest.param(1, [("chien", 2)], 2, id="the cut keeps the best rank"),
    ],
)
def test_the_target_rules(
    target_words: int, expected: list[tuple[str, int]], content: int
) -> None:
    cards, tokens, ranks = _target_fixture()

    targets, counted = corelist.build_targets(
        cards, tokens, ranks, {"paris"}, target_words=target_words
    )

    assert [(item.lemma, item.rank) for item in targets] == expected
    assert counted == content


def test_a_mismatched_token_list_raises() -> None:
    cards, tokens, ranks = _target_fixture()

    with pytest.raises(ValueError):
        corelist.build_targets(cards[:-1], tokens, ranks, set(), target_words=3)


def test_the_length_spread_shows_the_pool_against_the_kept_cards() -> None:
    assert cullflow.length_spread([4, 4, 5, 6], [4, 6]) == "4:50/50 5:25/0 6:25/50"
    assert cullflow.length_spread([4], []) == "4:100/0"


@pytest.mark.parametrize(
    ("exempt_top", "kept", "rejected"),
    [
        pytest.param(0, 2, ["use_cap"] * 3, id="a non-exempt word stops at the cap"),
        pytest.param(1, 5, [], id="an exempt word passes the cap"),
    ],
)
def test_the_use_cap(exempt_top: int, kept: int, rejected: list[str]) -> None:
    cards, tokens, targets, ranks = _cap_fixture()

    result = select(cards, tokens, targets, ranks, max_uses=2, exempt_top=exempt_top)

    assert len(result.kept) == kept
    assert result.uses["chat"] == kept
    left = [index for index in range(len(cards)) if index not in result.kept]
    assert [result.reasons[index] for index in left] == rejected


def test_a_lemma_jaccard_of_067_is_duplicate() -> None:
    shared = [f"s{index}" for index in range(67)]
    first = shared + [f"a{index}" for index in range(20)]
    second = shared + [f"b{index}" for index in range(13)]
    cards = [card("Le chat est ici.", index=0), card("Le chat est la.", index=1)]
    tokens = [
        [(lemma, "NOUN") for lemma in first],
        [(lemma, "NOUN") for lemma in second],
    ]

    result = select(cards, tokens, [target("s0", 100)], {lemma: 100 for lemma in shared})

    assert len(result.kept) == 1
    assert result.reasons == {1: "similar"}


def test_input_order_keeps_the_given_order() -> None:
    ordered = cullflow.order_cards(
        [2, 0, 1], ORDER_TOKENS, ORDER_RANKS, order="input", content_pos=CONTENT_POS
    )

    assert ordered == [2, 0, 1]


def test_an_unknown_order_raises() -> None:
    with pytest.raises(ValueError):
        cullflow.order_cards(
            [0], ORDER_TOKENS, ORDER_RANKS, order="random", content_pos=CONTENT_POS
        )


def test_cull_rows_columns_and_reasons() -> None:
    cards = [
        card("Le chat mange vite.", index=0),
        card("Le chien dort bien.", index=1),
        card("Le chat mange vite.", index=2),
    ]
    selection = cullflow.SelectionResult(
        kept=[1],
        uses=Counter({"chien": 1}),
        gain={1: 0.5},
        exempt=set(),
        reasons={2: "similar"},
        card_targets={1: ("chien",)},
        card_hardest={1: 3},
    )

    rows = cullflow.cull_rows(cards, ["internal_mark", "", ""], selection, [1])

    assert all(set(row) == set(cullflow.CULL_COLUMNS) for row in rows)
    assert [row["sentence"] for row in rows] == [
        cards[1].sentence,
        cards[0].sentence,
        cards[2].sentence,
    ]
    assert [row["kept"] for row in rows] == ["1", "0", "0"]
    assert [row["reason"] for row in rows] == ["", "internal_mark", "similar"]
    assert rows[0]["gain"] == "0.500000"
    assert rows[0]["targets"] == "chien"
    assert rows[0]["hardest_rank"] == "3"
    assert rows[1]["gain"] == ""


def test_coverage_rows_statuses() -> None:
    targets = [target("chat", 1), target("boire", 2), target("lait", 3)]
    selection = cullflow.SelectionResult(
        kept=[0, 1],
        uses=Counter({"chat": 2, "boire": 1}),
        gain={0: 0.5, 1: 0.4},
        exempt=set(),
        reasons={},
    )

    rows = cullflow.coverage_rows(targets, selection, {"chat": "cat"})

    assert all(set(row) == set(cullflow.COVERAGE_COLUMNS) for row in rows)
    assert [row["word"] for row in rows] == ["chat", "boire", "lait"]
    assert [row["count"] for row in rows] == [2, 1, 0]
    assert [row["status"] for row in rows] == ["covered", "partial", "absent"]
    assert [row["gloss"] for row in rows] == ["cat", "", ""]


def _reserve_card(sentence: str) -> cullflow.ReserveCard:
    return cullflow.ReserveCard(
        sentence=sentence,
        html=sentence,
        english=ENGLISH,
        target="chat",
        gloss="cat",
        audio="a.mp3",
    )


def test_dedupe_reserve_drops_an_internal_duplicate() -> None:
    reserve = [_reserve_card("Le chat est ici."), _reserve_card("Le chat est ici aussi.")]
    sets = [{"chat", "ici", "aussi"}, {"chat", "ici", "aussi", "le"}]

    kept, kept_sets = cullflow.dedupe_reserve_sets(
        reserve, sets, [], [], content_pos=CONTENT_POS
    )

    assert kept == reserve[:1]
    assert kept_sets == sets[:1]


@pytest.mark.parametrize(
    ("sentence", "lemmas"),
    [
        pytest.param("Le chat est ici.", {"autre"}, id="the same sentence"),
        pytest.param("Un chat est ici.", {"chat", "ici"}, id="the same lemmas"),
    ],
)
def test_dedupe_reserve_drops_a_card_that_the_input_holds(
    sentence: str, lemmas: set[str]
) -> None:
    kept, _kept_sets = cullflow.dedupe_reserve_sets(
        [_reserve_card(sentence)],
        [lemmas],
        [card("Le chat est ici.")],
        [[("chat", "NOUN"), ("ici", "NOUN")]],
        content_pos=CONTENT_POS,
    )

    assert kept == []


RESERVE_PACK_FIELDS = [
    "Audio",
    "Sentence",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
]


def _reserve_pack(tmp_path: Path) -> Path:
    (audio,) = write_mp3s(tmp_path, ["reserve.mp3"])
    path = tmp_path / "reserve.apkg"
    AnkiPackage.create(
        path,
        "Reserve Source",
        fields=RESERVE_PACK_FIELDS,
        cards=[
            {
                "Audio": "[sound:reserve.mp3]",
                "Sentence": "<b>je</b> suis ici.",
                "English": "I am here.",
                "Target": "je",
                "Gloss": "I",
                "WordAudio": "",
                "Source": "Tatoeba #1",
            }
        ],
        media_files=[str(audio)],
        model_name="Reserve Sentences",
    )
    return path


def test_the_cull_fills_the_core_word_and_the_gloss(tmp_path: Path) -> None:
    profile = replace(
        _profile(),
        sense_path=_senses(tmp_path),
        reserve_pack=_reserve_pack(tmp_path),
    )

    values = _notes(_cull(tmp_path, profile).output_path)

    reserve = next(row for row in values if "<b>je</b>" in row[1])
    frequency = next(row for row in values if "<b>je</b>" not in row[1])
    assert reserve[3:] == ["je", "I"]
    assert frequency[3:] == ["kat", "cat"]


def test_the_cull_keeps_the_due_of_a_kept_card(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    with AnkiPackage(source) as package:
        package.conn.execute("UPDATE cards SET due = 42")
        package.conn.commit()
        package.touch()
        package.save()

    output = _cull(tmp_path, _profile(), max_cards=1, source=source).output_path

    with AnkiPackage(output) as package:
        dues = [
            int(row[0])
            for row in package.conn.execute("SELECT due FROM cards").fetchall()
        ]
    assert dues == [42]
    assert len(_notes(output)) == 1


CAP = cullflow.DEFAULT_MAX_USES


@pytest.mark.parametrize(
    ("english", "sources", "passes"),
    [
        pytest.param(
            ["dog is big."] * (CAP + 1),
            [cullflow.FREQUENCY_SOURCE] * (CAP + 1),
            True,
            id="a lemma that fails the sense test is not a use",
        ),
        pytest.param(
            ["cat is big."] * (CAP + 1),
            [cullflow.FREQUENCY_SOURCE] * (CAP + 1),
            False,
            id="one use over the cap fails",
        ),
        pytest.param(
            ["cat is big."] * (CAP + 1),
            [cullflow.FREQUENCY_SOURCE] * CAP + ["reserve"],
            True,
            id="a reserve card is not a use",
        ),
    ],
)
def test_the_use_cap_gate(
    tmp_path: Path,
    english: list[str],
    sources: list[str],
    passes: bool,
) -> None:
    cards = [
        SimpleNamespace(sentence=f"kat {index}", english=text)
        for index, text in enumerate(english)
    ]
    files = gate_reader(
        replace(_profile(), sense_path=_senses(tmp_path)),
        coverage=[],
        rows=[
            {"sentence": card.sentence, "source": source, "kept": "1"}
            for card, source in zip(cards, sources)
        ],
        output={"cards": cards},
        tokens=[[("kat", "NOUN")]] * len(cards),
    )

    result, message = cullgates.gate_use_cap(files)

    assert result is passes, message


@pytest.mark.parametrize(
    ("source", "passes"),
    [
        pytest.param(cullflow.FREQUENCY_SOURCE, False, id="a short frequency card fails"),
        pytest.param(cullflow.SWADESH_SOURCE, True, id="a short reserve card is skipped"),
    ],
)
def test_the_filter_gate_skips_a_reserve_card(source: str, passes: bool) -> None:
    files = gate_reader(
        _profile(),
        rows=[{"sentence": "kat", "english": "cat", "source": source, "kept": "1"}],
    )

    result, message = cullgates.gate_filter_recheck(files)

    assert result is passes, message


@pytest.mark.parametrize(
    ("kept", "dropped", "left", "passes"),
    [
        pytest.param(924, 76, 50, True, id="the no-core drop fills the cap"),
        pytest.param(500, 0, 0, True, id="the pool has no more cards"),
        pytest.param(923, 76, 50, False, id="cards are left in the pool"),
    ],
)
def test_the_card_cap_gate(kept: int, dropped: int, left: int, passes: bool) -> None:
    files = gate_reader(
        None,
        rows=[{"kept": "1"}] * kept
        + [{"kept": "0", "reason": "max_cards"}] * left
        + [{"kept": "0", "reason": "similar"}],
        manifest={"params": {"max_cards": 1000}, "counts": {"dropped_no_core": dropped}},
    )

    result, message = cullgates.gate_card_cap(files)

    assert result is passes, message
    assert f"{kept} kept + {dropped} dropped" in message
