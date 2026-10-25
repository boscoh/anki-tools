"""Tests for the core-word small deck (spec sections 2 to 10).

Every test is offline. The token lists are built by hand, so no test loads
spaCy, ffprobe, or the frequency list.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.french import core, cull
from anki_tools.ranking import coreflow, corelist
from anki_tools.ranking.coreflow import (
    coverage_rows_of,
    find_model_id,
    InputCard,
    kept_order_hash,
    outside_lemmas,
    sense_match,
    side_paths,
    size_check,
    write_rank_csv,
)
from anki_tools.ranking.corelist import TargetWord, build_core_targets
from anki_tools.ranking.core import (
    CoreCandidate,
    CoreOptions,
    CoreResult,
    assign_targets,
    card_cost,
    select,
    target_quotas,
)
from anki_tools.ranking.coregates import gate_gloss


def _card(
    sentence: str,
    index: int = 0,
    english: str = "A short English line.",
    audio: str = "a.mp3",
) -> InputCard:
    return InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def _target(
    key: str,
    tier: str = "S",
    position: int = 1,
    form: str = "",
    gloss: str = "",
    senses: frozenset[str] = frozenset(),
) -> corelist.CoreTarget:
    return corelist.CoreTarget(
        key=key,
        tier=tier,
        position=position,
        form=form or key.split(":")[-1],
        gloss=gloss,
        senses=senses,
    )


class TestTargetQuotas:
    def test_a_ubiquitous_target_gets_one_example(self) -> None:
        options = CoreOptions(per_word=2, ubiquitous_share=0.25)
        quotas = target_quotas(
            {"S1:je": 1, "S2:eau": 2},
            {"S1:je": 0.9, "S2:eau": 0.1},
            options,
        )

        assert quotas == {"S1:je": 1, "S2:eau": 2}

    def test_the_third_example_can_raise_the_top_target(self) -> None:
        options = CoreOptions(per_word=2, third_examples=1)
        quotas = target_quotas(
            {"S1:eau": 1, "S2:feu": 2}, {"S1:eau": 0.1, "S2:feu": 0.1}, options
        )

        assert quotas == {"S1:eau": 3, "S2:feu": 2}


class TestCost:
    def test_a_new_rare_word_costs_three_times_a_frequent_one(self) -> None:
        options = CoreOptions(lambda_new=1.0, lambda_repeat=0.0, lambda_len=0.0)
        ranks = {"rare": 5000, "common": 10}
        rare = CoreCandidate(0, ("F:x",), ("rare",), 5)
        common = CoreCandidate(1, ("F:x",), ("common",), 5)

        assert card_cost(rare, set(), ranks, options) == pytest.approx(3.0)
        assert card_cost(common, set(), ranks, options) == pytest.approx(1.0)

    def test_a_repeated_outside_word_costs_less(self) -> None:
        options = CoreOptions(lambda_new=1.0, lambda_repeat=0.0, lambda_len=0.0)
        card = CoreCandidate(0, ("F:x",), ("rare",), 5)

        assert card_cost(card, {"rare"}, {"rare": 5000}, options) == 0.0


def _selection_fixture() -> tuple[list[CoreCandidate], dict[str, int], dict[str, int]]:
    candidates = [
        CoreCandidate(0, ("S1:a", "S2:b"), ("x",), 5),
        CoreCandidate(1, ("S1:a",), ("y",), 5),
        CoreCandidate(2, ("S2:b",), ("z",), 5),
        CoreCandidate(3, ("S1:a", "S2:b"), ("x",), 5),
        CoreCandidate(4, ("S3:c",), ("w",), 5),
        CoreCandidate(5, ("S3:c",), ("v",), 5),
    ]
    positions = {"S1:a": 1, "S2:b": 2, "S3:c": 3}
    quotas = {"S1:a": 2, "S2:b": 2, "S3:c": 2}
    return candidates, positions, quotas


class TestSelection:
    def test_the_seed_pass_gives_each_target_one_card(self) -> None:
        candidates, positions, quotas = _selection_fixture()
        options = CoreOptions(max_cards=10, completion_cards=0, per_word=1)

        result = select(
            candidates,
            positions=positions,
            quotas={key: 1 for key in quotas},
            ranks={},
            options=options,
        )

        assert set(result.uses) == set(positions)
        assert all(result.pick_pass[index] == "seed" for index in result.kept)

    def test_the_completion_pass_fills_a_short_quota(self) -> None:
        candidates, positions, quotas = _selection_fixture()
        options = CoreOptions(max_cards=10, completion_cards=10, per_word=2)

        result = select(
            candidates, positions=positions, quotas=quotas, ranks={}, options=options
        )

        assert result.uses["S3:c"] == 2
        assert "completion" in set(result.pick_pass.values())

    def test_the_tail_needs_a_new_content_lemma(self) -> None:
        candidates, positions, _quotas = _selection_fixture()
        options = CoreOptions(max_cards=10, completion_cards=0, tail_cards=5)

        result = select(
            candidates,
            positions=positions,
            quotas={"S1:a": 1, "S2:b": 1, "S3:c": 1},
            ranks={"x": 10, "y": 20, "z": 30, "w": 40},
            options=options,
        )

        tail = [index for index in result.kept if result.pick_pass[index] == "tail"]
        assert tail
        for index in tail:
            assert candidates[index].outside

    def test_the_order_follows_the_core_list(self) -> None:
        candidates = [
            CoreCandidate(0, ("S2:b",), (), 5),
            CoreCandidate(1, ("S1:a",), (), 5),
            CoreCandidate(2, ("S1:a",), (), 5),
        ]
        positions = {"S1:a": 1, "S2:b": 2}
        options = CoreOptions(max_cards=10, completion_cards=0)

        result = select(
            candidates,
            positions=positions,
            quotas={"S1:a": 2, "S2:b": 1},
            ranks={},
            options=options,
        )

        assigned = [positions[result.assigned[index]] for index in result.order if index in result.assigned]
        assert assigned == sorted(assigned)

    def test_assign_targets_uses_the_lowest_free_target(self) -> None:
        candidates = [
            CoreCandidate(0, ("S1:a", "S2:b"), (), 5),
            CoreCandidate(1, ("S2:b",), (), 5),
        ]
        assigned = assign_targets([0, 1], candidates, {"S1:a": 1, "S2:b": 2})

        assert assigned == {0: "S1:a", 1: "S2:b"}


class TestBuildTargets:
    def test_two_glosses_with_one_sense_merge(self) -> None:
        table = [(1, "je", "I"), (2, "je", "me")]
        targets, dropped = build_core_targets(table, [], {}, 0)

        assert len(targets) == 1
        assert len(dropped) == 1

    def test_two_senses_keep_two_targets(self) -> None:
        table = [(1, "voler", "fly"), (2, "voler", "steal")]
        targets, dropped = build_core_targets(table, [], {}, 0)

        assert [target.key for target in targets] == ["S1:voler", "S2:voler"]
        assert not dropped

    def test_a_frequency_lemma_that_is_a_swadesh_form_is_not_added(self) -> None:
        table = [(1, "eau", "water")]
        frequency = [
            TargetWord("eau", 1, 3, 1.0, 1.0),
            TargetWord("pain", 2, 3, 1.0, 1.0),
        ]

        targets, _dropped = build_core_targets(table, frequency, {}, 5)

        assert [target.key for target in targets] == ["S1:eau", "F:pain"]

    def test_the_core_words_cut_the_frequency_list(self) -> None:
        frequency = [TargetWord(f"w{index}", index, 3, 1.0, 1.0) for index in range(1, 6)]

        targets, _dropped = build_core_targets([], frequency, {}, 2)

        assert [target.key for target in targets] == ["F:w1", "F:w2"]

    def test_a_frequency_lemma_gets_its_first_sense_as_the_gloss(self) -> None:
        frequency = [TargetWord("faire", 1, 3, 1.0, 1.0)]
        senses = {
            "faire": corelist.SenseWords(
                frozenset({"do", "make"}), frozenset(), "do"
            )
        }

        targets, _dropped = build_core_targets([], frequency, senses, 1)

        assert targets[0].gloss == "do"


class TestFilter:
    def test_c1_rejects_a_sentence_without_a_predicate(self) -> None:
        cards = [_card("Très bon.", english="Very good.")]
        tokens = [[("très", "ADV"), ("bon", "ADJ")]]

        passed, reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, [], [], min_units=1, max_units=9
        )

        assert reasons[0] == ""
        assert passed == [0]

    def test_c1_uses_a_verb_or_adjective(self) -> None:
        cards = [_card("Trois chats.")]
        tokens = [[("trois", "NUM"), ("chat", "NOUN")]]

        passed, reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, [], [], min_units=1, max_units=9
        )

        assert passed == []
        assert reasons[0] == "C1"

    def test_c3_matches_a_whole_name(self) -> None:
        cards = [_card("Tomber est facile.")]
        tokens = [[("tomber", "VERB"), ("être", "AUX"), ("facile", "ADJ")]]

        passed, _reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, ["tom"], [], min_units=1, max_units=9
        )

        assert passed == [0]

    def test_c3_rejects_a_name_word(self) -> None:
        cards = [_card("Tom mange une pomme.")]
        tokens = [[("tom", "PROPN"), ("manger", "VERB"), ("un", "DET"), ("pomme", "NOUN")]]

        passed, reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, ["tom"], [], min_units=1, max_units=9
        )

        assert passed == []
        assert reasons[0] in {"C3", "F8"}

    def test_c4_rejects_a_loan_word(self) -> None:
        cards = [_card("Le week-end est court.")]
        tokens = [[("le", "DET"), ("week", "NOUN"), ("est", "AUX"), ("court", "ADJ")]]

        passed, reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, [], ["week-end"], min_units=1, max_units=9
        )

        assert passed == []
        assert reasons[0] == "C4"

    def test_c5_rejects_a_card_without_audio(self) -> None:
        cards = [_card("Le chat mange.", audio="missing.mp3")]
        tokens = [[("le", "DET"), ("chat", "NOUN"), ("manger", "VERB")]]

        passed, reasons = core.filter_core_cards(
            cards, tokens, {"a.mp3"}, [], [], min_units=1, max_units=9
        )

        assert passed == []
        assert reasons[0] == "C5"


class TestSense:
    def test_a_hit_needs_a_sense_word(self) -> None:
        assert not sense_match(
            "I saw it.", frozenset({"time"}), frozenset()
        )
        assert sense_match("What time is it?", frozenset({"time"}), frozenset())

    def test_a_false_friend_blocks_the_hit(self) -> None:
        assert not sense_match(
            "I have time.", frozenset({"time"}), frozenset({"have"})
        )

    def test_an_unchecked_target_accepts_every_hit(self) -> None:
        assert sense_match("anything", frozenset(), frozenset())

    def test_the_sense_file_and_the_loan_list_load(self, tmp_path: Path) -> None:
        senses = tmp_path / "senses.tsv"
        senses.write_text(
            "word\tsenses\tfalse_friends\npas\tnot\t\n", encoding="utf-8"
        )
        loans = tmp_path / "loans.txt"
        loans.write_text("# a comment\nweek-end\n", encoding="utf-8")

        assert core.read_senses(senses) == {
            "pas": corelist.SenseWords(frozenset({"not"}), frozenset(), "not")
        }
        assert core.read_loanwords(loans) == frozenset({"week-end"})


class TestOutside:
    def test_a_core_form_is_not_an_outside_word(self) -> None:
        tokens = [[("chat", "NOUN"), ("manger", "VERB"), ("vite", "ADV")]]

        outside = outside_lemmas(0, tokens, frozenset({"chat", "vite"}))

        assert outside == ("manger",)


class TestCoverage:
    def test_the_status_follows_the_count_and_the_quota(self) -> None:
        targets = [
            _target("S1:a", position=1),
            _target("S2:b", position=2),
            _target("S3:c", position=3),
            _target("S4:d", position=4, senses=frozenset({"thing"})),
        ]
        result = CoreResult(
            kept=[],
            uses=Counter({"S1:a": 2, "S2:b": 1, "S4:d": 0}),
            quotas={"S1:a": 2, "S2:b": 2, "S3:c": 2, "S4:d": 2},
        )

        rows = coverage_rows_of(
            targets,
            result,
            {"S1:a": True, "S2:b": True, "S3:c": False, "S4:d": True},
            {"S1:a": 2, "S2:b": 2, "S3:c": 2, "S4:d": 2},
        )
        status = {row["key"]: row["status"] for row in rows}

        assert status == {
            "S1:a": "covered",
            "S2:b": "partial",
            "S3:c": "absent",
            "S4:d": "absent_sense",
        }

    def test_a_target_without_a_sense_row_is_unchecked(self) -> None:
        targets = [_target("F:x", tier="F", position=1)]
        result = CoreResult(
            kept=[], uses=Counter({"F:x": 1}), quotas={"F:x": 2}
        )

        rows = coverage_rows_of(
            targets, result, {"F:x": True}, {"F:x": 2}
        )

        assert rows[0]["status"] == "unchecked"


class TestSizeCheck:
    def test_a_small_budget_stops_the_run(self) -> None:
        with pytest.raises(ValueError, match="cannot fit"):
            size_check({f"t{index}": 2 for index in range(100)}, 50)

    def test_a_large_budget_passes(self) -> None:
        size_check({f"t{index}": 2 for index in range(100)}, 300)


class TestFiles:
    def test_the_rank_csv_holds_the_kept_cards_in_order(self, tmp_path: Path) -> None:
        cards = [_card("A.", index=0), _card("B.", index=1)]
        path = tmp_path / "x.rank.csv"

        write_rank_csv(cards, [1, 0], path)
        rows = list(csv.DictReader(path.open(encoding="utf-8")))

        assert [row["sentence"] for row in rows] == ["B.", "A."]
        assert [row["rank"] for row in rows] == ["1", "2"]
        assert kept_order_hash(cards, [1, 0]) == kept_order_hash(cards, [1, 0])

    def test_the_side_paths_hold_the_four_files(self, tmp_path: Path) -> None:
        paths = side_paths(tmp_path / "deck.apkg")

        assert [path.name for path in paths] == [
            "deck.core.csv",
            "deck.coverage.csv",
            "deck.manifest.json",
            "deck.rank.csv",
        ]

    def test_the_default_output_uses_the_core_suffix(self, tmp_path: Path) -> None:
        assert coreflow.default_output(tmp_path / "deck.apkg").name == "deck_core.apkg"


def _tiny_deck(tmp_path: Path) -> Path:
    sentences = ["Le chat dort.", "Le chat mange.", "Le lapin court."]
    media = []
    for index in range(len(sentences)):
        path = tmp_path / f"a{index}.mp3"
        path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
        media.append(str(path))
    rows = [
        {
            "French": sentence,
            "sound": f"[sound:a{index}.mp3]",
            "Back": f"English {index}.",
            "freq ranks": "[ 1 2 ]",
        }
        for index, sentence in enumerate(sentences)
    ]
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=["French", "sound", "Back", "freq ranks"],
        cards=rows,
        media_files=media,
        model_name=cull.MODEL_NAME,
    )
    return source


class TestCardDesign:
    def test_the_front_shows_only_the_audio(self) -> None:
        front = core.CORE_FRONT_TEMPLATE

        assert "{{sound}}" in front
        assert "{{French}}" not in front
        assert "{{Back}}" not in front
        assert "{{Core}}" not in front
        assert "hint" not in front

    def test_the_back_shows_audio_sentence_english_and_core_word(self) -> None:
        back = core.CORE_BACK_TEMPLATE
        order = [back.index(f"{{{{{name}}}}}") for name in ("sound", "French", "Back", "Core")]

        assert order == sorted(order)

    def test_the_deck_has_one_card_template(self) -> None:
        assert len(core.CORE_TEMPLATES) == 1

    def test_the_model_gets_the_core_field_and_templates(self, tmp_path: Path) -> None:
        source = _tiny_deck(tmp_path)
        cards = cull.load_input_deck(source)
        output = tmp_path / "core.apkg"
        cull.write_deck(source, cards, [0, 1, 2], output, "Core Deck")
        targets = [_target("S1:chat", form="chat", gloss="cat")]
        candidates = [
            CoreCandidate(index=0, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=1, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=2, targets=(), outside=(), units=3),
        ]
        result = CoreResult(
            kept=[0, 1, 2],
            uses=Counter(),
            quotas={},
            assigned={0: "S1:chat"},
            order=[0, 1, 2],
        )

        filled = core._write_core_model(
            output, cards, result, [0, 1, 2], targets, candidates
        )

        assert filled == 2
        with AnkiPackage(output) as package:
            models = package.get_models(include_templates=True)
            model = next(m for m in models.values() if m["name"] == cull.MODEL_NAME)
            names = [f["name"] for f in model["flds"]]
            templates = [(t["qfmt"], t["afmt"]) for t in model["tmpls"]]
            values = {
                row["flds"].split("\x1f")[0]: row["flds"].split("\x1f")
                for row in package.get_notes()
            }

        assert names == [
            "French",
            "sound",
            "Back",
            "freq ranks",
            "Core",
            "CoreGloss",
        ]
        assert templates == [tuple(pair) for pair in core.CORE_TEMPLATES]
        with AnkiPackage(output) as package:
            ords = [row["ord"] for row in package.get_cards()]
        assert ords == [0, 0, 0]
        assert values["Le chat dort."][4] == "chat"
        assert values["Le chat mange."][4] == "chat"
        assert values["Le lapin court."][4] == ""
        assert values["Le chat dort."][3] == "[ 1 2 ]"
        assert values["Le chat dort."][5] == "cat"
        assert values["Le chat mange."][5] == "cat"
        assert values["Le lapin court."][5] == ""

    def test_the_gloss_gate_passes_on_a_filled_deck(self, tmp_path: Path) -> None:

        source = _tiny_deck(tmp_path)
        cards = cull.load_input_deck(source)
        output = tmp_path / "core.apkg"
        cull.write_deck(source, cards, [0, 1, 2], output, "Core Deck")
        targets = [_target("S1:chat", form="chat", gloss="cat")]
        candidates = [
            CoreCandidate(index=0, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=1, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=2, targets=(), outside=(), units=3),
        ]
        result = CoreResult(
            kept=[0, 1, 2],
            uses=Counter(),
            quotas={},
            assigned={0: "S1:chat"},
            order=[0, 1, 2],
        )
        core._write_core_model(
            output, cards, result, [0, 1, 2], targets, candidates
        )

        passed, message = gate_gloss(core.CoreFiles(output))

        assert passed, message

    def test_the_gloss_gate_fails_on_a_wrong_gloss(self, tmp_path: Path) -> None:
        from anki_tools.ranking.coreflow import side_paths

        source = _tiny_deck(tmp_path)
        cards = cull.load_input_deck(source)
        output = tmp_path / "core.apkg"
        cull.write_deck(source, cards, [0, 1, 2], output, "Core Deck")
        targets = [_target("S1:chat", form="chat", gloss="cat")]
        candidates = [
            CoreCandidate(index=0, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=1, targets=("S1:chat",), outside=(), units=3),
            CoreCandidate(index=2, targets=(), outside=(), units=3),
        ]
        result = CoreResult(
            kept=[0, 1, 2],
            uses=Counter(),
            quotas={},
            assigned={0: "S1:chat"},
            order=[0, 1, 2],
        )
        core._write_core_model(
            output, cards, result, [0, 1, 2], targets, candidates
        )
        _core, coverage, _manifest, _rank = side_paths(output)
        coverage.write_text(
            "key,word,gloss,tier,position,quota,count,status\n"
            "S1:chat,chat,dog,S,1,2,2,covered\n",
            encoding="utf-8",
        )

        passed, message = gate_gloss(core.CoreFiles(output))

        assert not passed
        assert "is not the target gloss" in message

    def test_adding_the_field_twice_changes_nothing(self, tmp_path: Path) -> None:
        source = _tiny_deck(tmp_path)
        with AnkiPackage(source) as package:
            model_id = find_model_id(
                package.get_models(include_templates=True), cull.MODEL_NAME
            )
            first = package.add_model_field(model_id, "Core")
            second = package.add_model_field(model_id, "Core")
            widths = {len(row["flds"].split("\x1f")) for row in package.get_notes()}

        assert first == second == 4
        assert widths == {5}
