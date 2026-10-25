"""Test that the shared cull flow carries no French value.

The test builds a second :class:`anki_tools.ranking.cullflow.CullProfile` with
a different model name, field names, and deck name, and runs the full cull on a
tiny package. The output deck must use those values. The module also holds the
unit tests of the shared flow.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import coreflow, corelist, cullflow

SENTENCES = [
    ("kat ser stor.", "cat is big."),
    ("kat ser liten.", "cat is small."),
    ("kat ser glad.", "cat is glad."),
]
FIELDS = ["Text", "Audio", "Meaning"]
MODEL_NAME = "Test Sentences"
DECK_NAME = "Test Two Sentences"
ENGLISH = "A short English line."
CONTENT_POS = corelist.CONTENT_POS
ORDER_TOKENS = [
    [("maison", "NOUN"), ("grande", "ADJ")],
    [("chat", "NOUN"), ("noir", "ADJ")],
    [("et", "DET")],
]
ORDER_RANKS = {"maison": 5, "grande": 40, "chat": 2, "noir": 900}


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


def _card(
    sentence: str,
    index: int = 0,
    english: str = ENGLISH,
    audio: str = "a.mp3",
) -> coreflow.InputCard:
    return coreflow.InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def _target(lemma: str, rank: int) -> corelist.TargetWord:
    return corelist.TargetWord(
        lemma=lemma,
        rank=rank,
        deck_count=3,
        content_share=1.0,
        weight=1 / math.log2(rank + 1),
    )


def _freq_row(
    word: str,
    count: int = 10,
    pm: float = 5.0,
    written: float = 5.0,
) -> cullflow.FrequencyRow:
    return cullflow.FrequencyRow(
        word=word,
        tatoeba_count=count,
        tatoeba_pm=pm,
        written_pm=written,
        zipf=4.0,
    )


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
    cards = [_card("Le chat mange ici.", index=index) for index in range(len(tokens))]
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
    cards = [_card(sentence, index=index) for index, sentence in enumerate(sentences)]
    tokens = [
        [("chat", "NOUN"), (f"f{index}", "NOUN")] for index in range(len(sentences))
    ]
    return cards, tokens, [_target("chat", 1)], {"chat": 1}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def test_a_count_below_five_uses_the_written_term() -> None:
    row = _freq_row("rare", count=4, pm=0.0, written=12.5)

    assert cullflow.blend_pm(row, spoken_weight=1.0) == pytest.approx(12.5)
    assert cullflow.blend_pm(row, spoken_weight=0.0) == pytest.approx(12.5)


def test_the_middle_weight_uses_both_terms() -> None:
    row = _freq_row("chat", count=10, pm=20.0, written=10.0)

    assert cullflow.blend_pm(row, spoken_weight=0.5) == pytest.approx(15.0)


class TestTargets:
    def test_the_target_rules_keep_two_of_the_candidates(self) -> None:
        cards, tokens, ranks = _target_fixture()

        targets, content = corelist.build_targets(
            cards, tokens, ranks, {"paris"}, target_words=300
        )

        assert [target.lemma for target in targets] == ["chien", "mange"]
        assert content == 2
        assert targets[0].rank == 2
        assert targets[1].rank == 10

    def test_the_target_cut_keeps_the_best_rank(self) -> None:
        cards, tokens, ranks = _target_fixture()

        targets, content = corelist.build_targets(
            cards, tokens, ranks, {"paris"}, target_words=1
        )

        assert [target.lemma for target in targets] == ["chien"]
        assert content == 2

    def test_a_mismatched_token_list_raises(self) -> None:
        cards, tokens, ranks = _target_fixture()

        with pytest.raises(ValueError):
            corelist.build_targets(cards[:-1], tokens, ranks, set(), target_words=3)


class TestSelection:
    def test_a_rare_word_wins_over_a_repeated_word(self) -> None:
        cards = [
            _card("Une phrase rare ici.", index=0),
            _card("Une phrase commune ici.", index=1),
            _card("Une autre phrase commune.", index=2),
        ]
        tokens = [
            [("rare", "NOUN"), ("f1", "NOUN")],
            [("commune", "NOUN"), ("f2", "NOUN")],
            [("commune", "NOUN"), ("f3", "NOUN")],
        ]
        targets = [_target("rare", 1), _target("commune", 50)]
        ranks = {"rare": 1, "commune": 50}

        result = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=4,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )

        assert result.kept == [0, 1, 2]
        assert result.gain[1] > result.gain[2]

    def test_a_non_exempt_word_stops_at_the_cap(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=2,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )

        assert len(result.kept) == 2
        assert result.uses["chat"] == 2

    def test_a_card_with_a_capped_word_is_skipped(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=2,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )

        rejected = [index for index in range(len(cards)) if index not in result.kept]
        assert [result.reasons[index] for index in rejected] == ["use_cap"] * 3

    def test_an_exempt_word_passes_the_cap(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=2,
            exempt_top=1,
            content_pos=CONTENT_POS,
        )

        assert len(result.kept) == len(cards)
        assert result.uses["chat"] == len(cards)

    def test_a_lemma_jaccard_of_067_is_duplicate(self) -> None:
        shared = [f"s{index}" for index in range(67)]
        first = shared + [f"a{index}" for index in range(20)]
        second = shared + [f"b{index}" for index in range(13)]
        cards = [
            _card("Le chat est ici.", index=0),
            _card("Le chat est la.", index=1),
        ]
        tokens = [
            [(lemma, "NOUN") for lemma in first],
            [(lemma, "NOUN") for lemma in second],
        ]
        ranks = {lemma: 100 for lemma in shared}

        result = cullflow.select_cards(
            cards,
            tokens,
            [],
            ranks,
            max_cards=10,
            max_uses=4,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )

        assert len(result.kept) == 1
        assert result.reasons == {1: "similar"}

    def test_two_runs_keep_the_same_order(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        first = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=2,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )
        second = cullflow.select_cards(
            cards,
            tokens,
            targets,
            ranks,
            max_cards=10,
            max_uses=2,
            exempt_top=0,
            content_pos=CONTENT_POS,
        )

        assert first.kept == second.kept


class TestOrder:
    def test_coverage_gives_a_non_decreasing_hardest_rank(self) -> None:
        ordered = cullflow.order_cards(
            [0, 1, 2], ORDER_TOKENS, ORDER_RANKS, content_pos=CONTENT_POS
        )

        hardest = [
            cullflow.hardest_rank(ORDER_TOKENS[index], ORDER_RANKS, CONTENT_POS)
            for index in ordered
        ]
        assert ordered == [0, 1, 2]
        assert hardest == [40, 900, cullflow.UNRANKED_ORDER]

    def test_input_keeps_the_given_order(self) -> None:
        ordered = cullflow.order_cards(
            [2, 0, 1],
            ORDER_TOKENS,
            ORDER_RANKS,
            order="input",
            content_pos=CONTENT_POS,
        )

        assert ordered == [2, 0, 1]

    def test_an_unknown_order_raises(self) -> None:
        with pytest.raises(ValueError):
            cullflow.order_cards(
                [0], ORDER_TOKENS, ORDER_RANKS, order="random", content_pos=CONTENT_POS
            )


def test_cull_csv_columns_and_reasons(tmp_path: Path) -> None:
    cards = [
        _card("Le chat mange vite.", index=0),
        _card("Le chien dort bien.", index=1),
        _card("Le chat mange vite.", index=2),
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
    path = tmp_path / "run.cull.csv"

    cullflow.write_cull_csv(cards, ["F7", "", ""], selection, [1], path)
    header, rows = _read_csv(path)
    raw = path.read_bytes()
    assert raw.split(b"\n")[0] == ",".join(cullflow.CULL_COLUMNS).encode()
    assert b"\r" not in raw

    assert header == list(cullflow.CULL_COLUMNS)
    assert len(rows) == 3
    assert [row["sentence"] for row in rows] == [
        cards[1].sentence,
        cards[0].sentence,
        cards[2].sentence,
    ]
    assert [row["kept"] for row in rows] == ["1", "0", "0"]
    assert [row["reason"] for row in rows] == ["", "F7", "similar"]
    assert rows[0]["gain"] == "0.500000"
    assert rows[0]["targets"] == "chien"
    assert rows[0]["hardest_rank"] == "3"
    assert rows[1]["gain"] == ""


def test_coverage_csv_statuses(tmp_path: Path) -> None:
    targets = [_target("chat", 1), _target("boire", 2), _target("lait", 3)]
    selection = cullflow.SelectionResult(
        kept=[0, 1],
        uses=Counter({"chat": 2, "boire": 1}),
        gain={0: 0.5, 1: 0.4},
        exempt=set(),
        reasons={},
    )
    path = tmp_path / "run.coverage.csv"

    cullflow.write_coverage_csv(targets, selection, path)
    header, rows = _read_csv(path)

    assert header == list(cullflow.COVERAGE_COLUMNS)
    assert [row["word"] for row in rows] == ["chat", "boire", "lait"]
    assert [row["count"] for row in rows] == ["2", "1", "0"]
    assert [row["status"] for row in rows] == ["covered", "partial", "absent"]


def test_dedupe_reserve_drops_an_internal_duplicate() -> None:
    reserve = [
        cullflow.ReserveCard(
            sentence="Le chat est ici.",
            html="Le <b>chat</b> est ici.",
            english="The cat is here.",
            target="chat",
            gloss="cat",
            audio="a.mp3",
        ),
        cullflow.ReserveCard(
            sentence="Le chat est ici aussi.",
            html="Le <b>chat</b> est ici aussi.",
            english="The cat is here too.",
            target="chat",
            gloss="cat",
            audio="b.mp3",
        ),
    ]
    sets = [
        {"chat", "ici", "aussi"},
        {"chat", "ici", "aussi", "le"},
    ]

    kept, kept_sets = cullflow.dedupe_reserve_sets(
        reserve, sets, [], [], content_pos=CONTENT_POS
    )

    assert len(kept) == 1
    assert kept_sets == [{"chat", "ici", "aussi"}]
