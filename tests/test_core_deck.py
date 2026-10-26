"""Tests for the core-word small deck (spec sections 2 to 10).

Every test is offline. The token lists are built by hand, so no test loads
spaCy, ffprobe, or the frequency list.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.french import core, cull
from anki_tools.ranking import coreflow, corelist
from anki_tools.ranking.coreflow import (
    coverage_rows_of,
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
from anki_tools.ranking import coregates
from anki_tools.ranking.coregates import gate_core_word
from anki_tools.ranking.scan import CoreProfile


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


def _filter(
    cards: Sequence[InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    names: Sequence[str] = (),
    loanwords: Sequence[str] = (),
) -> tuple[list[int], list[str]]:
    """Apply the shared rules with the French profile and a test window."""
    profile = core.FRENCH_CORE_PROFILE
    return coreflow.filter_core_cards(
        cards,
        tokens,
        {"a.mp3"},
        names,
        loanwords,
        profile.has_orphan,
        min_units=1,
        max_units=9,
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

        passed, reasons = _filter(cards, tokens)

        assert reasons[0] == ""
        assert passed == [0]

    def test_c1_uses_a_verb_or_adjective(self) -> None:
        cards = [_card("Trois chats.")]
        tokens = [[("trois", "NUM"), ("chat", "NOUN")]]

        passed, reasons = _filter(cards, tokens)

        assert passed == []
        assert reasons[0] == "C1"

    def test_c3_matches_a_whole_name(self) -> None:
        cards = [_card("Tomber est facile.")]
        tokens = [[("tomber", "VERB"), ("être", "AUX"), ("facile", "ADJ")]]

        passed, _reasons = _filter(cards, tokens, ["tom"])

        assert passed == [0]

    def test_c3_rejects_a_name_word(self) -> None:
        cards = [_card("Tom mange une pomme.")]
        tokens = [[("tom", "PROPN"), ("manger", "VERB"), ("un", "DET"), ("pomme", "NOUN")]]

        passed, reasons = _filter(cards, tokens, ["tom"])

        assert passed == []
        assert reasons[0] in {"C3", "F8"}

    def test_c4_rejects_a_loan_word(self) -> None:
        cards = [_card("Le week-end est court.")]
        tokens = [[("le", "DET"), ("week", "NOUN"), ("est", "AUX"), ("court", "ADJ")]]

        passed, reasons = _filter(cards, tokens, [], ["week-end"])

        assert passed == []
        assert reasons[0] == "C4"

    def test_c5_rejects_a_card_without_audio(self) -> None:
        cards = [_card("Le chat mange.", audio="missing.mp3")]
        tokens = [[("le", "DET"), ("chat", "NOUN"), ("manger", "VERB")]]

        passed, reasons = _filter(cards, tokens)

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

        assert coreflow.read_senses(senses) == {
            "pas": corelist.SenseWords(frozenset({"not"}), frozenset(), "not")
        }
        assert coreflow.read_loanwords(loans) == frozenset({"week-end"})


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
    english = ["The cat sleeps.", "The cat eats.", "The rabbit runs."]
    media = []
    for index in range(len(sentences)):
        path = tmp_path / f"a{index}.mp3"
        path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
        media.append(str(path))
    rows = [
        {
            "French": sentence,
            "sound": f"[sound:a{index}.mp3]",
            "Back": line,
            "freq ranks": "[ 1 2 ]",
        }
        for index, (sentence, line) in enumerate(zip(sentences, english))
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


def _tokens(cards: Sequence[InputCard]) -> list[list[tuple[str, str]]]:
    """Tokenize each sentence into its words, with a content tag."""
    return [
        [(word.strip(".").casefold(), "VERB") for word in card.sentence.split()]
        for card in cards
    ]


def _offline_profile(tmp_path: Path) -> CoreProfile:
    """Give the French core hooks with no spaCy model and no data file."""
    return replace(
        core.FRENCH_CORE_PROFILE,
        tokenize_deck=_tokens,
        load_frequency=list,
        read_names=set,
        word_table=lambda: [(1, "chat", "cat")],
        sense_path=tmp_path / "absent_senses.tsv",
        loan_path=tmp_path / "absent_loans.txt",
        min_units=1,
        max_units=9,
        default_options=CoreOptions(
            max_cards=10, per_word=1, completion_cards=0, tail_cards=1
        ),
    )


def _run_core(tmp_path: Path) -> tuple[coreflow.CoreRun, Path, CoreProfile]:
    """Run the core mode on the tiny deck."""
    source = _tiny_deck(tmp_path)
    profile = _offline_profile(tmp_path)
    output = tmp_path / "core.apkg"
    run = coreflow.run_core(
        coreflow.CoreRequest(
            input_path=source,
            output_path=output,
            core_words=0,
            options=profile.default_options,
        ),
        profile,
    )
    return run, output, profile


def _cards(path: Path) -> dict[int, tuple[int, int, int]]:
    """Give the card id, due, and ord of each note of a package."""
    with AnkiPackage(path) as package:
        rows = package.conn.execute("SELECT id, nid, due, ord FROM cards").fetchall()
    return {int(row["nid"]): (int(row["id"]), int(row["due"]), int(row["ord"])) for row in rows}


def _add_spare_field(path: Path, name: str) -> None:
    """Append a field to the only note type of a package."""
    with AnkiPackage(path) as package:
        conn = package.conn
        models = json.loads(conn.execute("SELECT models FROM col").fetchone()[0])
        model = next(iter(models.values()))
        extra = dict(model["flds"][-1])
        extra.update({"name": name, "ord": len(model["flds"]), "media": []})
        model["flds"].append(extra)
        conn.execute("UPDATE col SET models = ? WHERE id = 1", (json.dumps(models),))
        conn.commit()
        package._modified = True
        package.save()


class TestCardDesign:
    def test_the_front_shows_only_the_audio(self) -> None:
        front = core.CORE_FRONT_TEMPLATE

        assert "{{Audio}}" in front
        assert "{{Sentence}}" not in front
        assert "{{English}}" not in front
        assert "{{Core}}" not in front
        assert "hint" not in front

    def test_the_back_shows_audio_sentence_english_and_core_word(self) -> None:
        back = core.CORE_BACK_TEMPLATE
        order = [
            back.index(f"{{{{{name}}}}}")
            for name in ("Audio", "Sentence", "English", "Core", "CoreGloss")
        ]

        assert order == sorted(order)
        assert "{{#Core}}" not in back

    def test_the_deck_has_one_card_template(self) -> None:
        assert len(core.CORE_TEMPLATES) == 1

    def test_the_run_writes_one_note_type_with_the_output_fields(
        self, tmp_path: Path
    ) -> None:
        run, output, profile = _run_core(tmp_path)

        with AnkiPackage(output) as package:
            models = package.get_models(include_templates=True)
            cards = package.get_cards()

        assert [model["name"] for model in models.values()] == [
            profile.output_model_name
        ]
        model = next(iter(models.values()))
        assert [field["name"] for field in model["flds"]] == list(
            coreflow.output_fields(profile)
        )
        assert [
            (template["qfmt"], template["afmt"]) for template in model["tmpls"]
        ] == [core.CORE_TEMPLATES[0]]
        assert len(cards) == run.cards
        assert {int(row["ord"]) for row in cards} == {0}

    def test_the_model_gets_the_core_field_and_templates(self, tmp_path: Path) -> None:
        source = _tiny_deck(tmp_path)
        cards = cull.load_input_deck(source)
        output = tmp_path / "core.apkg"
        profile = core.FRENCH_CORE_PROFILE

        coreflow.write_deck(source, cards, [0, 1], output, "Core Deck", profile)
        filled = coreflow.fill_core_fields(
            output, cards, {0: ("chat", "cat"), 1: ("chat", "cat")}
        )

        with AnkiPackage(output) as package:
            models = package.get_models()
            notes = {
                int(row["id"]): str(row["flds"]).split("\x1f")
                for row in package.get_notes()
            }

        assert filled == 2
        assert list(models) != []
        model = next(iter(models.values()))
        assert model["name"] == profile.output_model_name
        assert [field["name"] for field in model["flds"]] == [
            "Audio",
            "Sentence",
            "English",
            "Core",
            "CoreGloss",
        ]
        assert notes[cards[0].note_id] == [
            "[sound:a0.mp3]",
            "Le chat dort.",
            "The cat sleeps.",
            "chat",
            "cat",
        ]
        assert notes[cards[1].note_id][3:] == ["chat", "cat"]

    def test_every_kept_note_holds_a_core_word_and_a_gloss(
        self, tmp_path: Path
    ) -> None:
        _run, output, profile = _run_core(tmp_path)
        files = coregates.CoreFiles(output, profile)

        passed, message = gate_core_word(files)
        core_word_problems, filled = coregates.core_word_problems(
            files.notes(),
            files.fields(),
            coreflow.coverage_glosses(files.coverage()),
        )
        notes = files.notes()
        fields = files.fields()
        notes[next(iter(notes))][fields.index("Core")] = ""

        failed, failed_message = gate_core_word(files)

        assert passed, message
        assert core_word_problems == []
        assert filled == 1
        assert not failed
        assert "no core word" in failed_message

    def test_every_kept_note_holds_a_bold_core_word(self, tmp_path: Path) -> None:
        _run, output, profile = _run_core(tmp_path)
        files = coregates.CoreFiles(output, profile)
        notes = files.notes()
        sentence_index = files.fields().index("Sentence")

        assert all(
            "<b>" in values[sentence_index] for values in notes.values()
        )

    def test_the_gloss_gate_fails_on_a_missing_gloss(self, tmp_path: Path) -> None:
        _run, output, profile = _run_core(tmp_path)
        with AnkiPackage(output) as package:
            note = package.get_notes()[0]
            package.update_note_field(int(note["id"]), 4, "")
            package.touch()
            package.save(output)

        passed, message = gate_core_word(coregates.CoreFiles(output, profile))

        assert not passed
        assert "has no gloss" in message

    def test_the_gloss_gate_fails_on_a_wrong_gloss(self, tmp_path: Path) -> None:
        _run, output, profile = _run_core(tmp_path)
        core_csv, coverage_csv, _manifest, _rank = side_paths(output)
        coverage_csv.write_text(
            "key,word,gloss,tier,position,quota,count,status\n"
            "S1:chat,chat,dog,S,1,1,1,covered\n",
            encoding="utf-8",
        )

        passed, message = gate_core_word(coregates.CoreFiles(output, profile))

        assert not passed
        assert "is not the coverage gloss" in message

    def test_a_kept_card_without_a_core_word_leaves_the_deck(
        self, tmp_path: Path
    ) -> None:
        run, output, profile = _run_core(tmp_path)

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        with run.core_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        dropped = [row for row in rows if row["reason"] == "no_core"]

        assert manifest["counts"]["dropped_no_core"] == 1
        assert len(dropped) == 1
        assert dropped[0]["kept"] == "0"
        assert "lapin" in dropped[0]["sentence"]
        assert run.cards == 1
        passed, message = coregates.gate_order(coregates.CoreFiles(output, profile))
        assert passed, message
        with AnkiPackage(output) as package:
            sentences = [
                str(row["flds"]).split("\x1f")[1] for row in package.get_notes()
            ]
        assert len(sentences) == 1
        assert all("lapin" not in sentence for sentence in sentences)

    def test_the_kept_card_keeps_its_note_id_and_due(self, tmp_path: Path) -> None:
        source = _tiny_deck(tmp_path)
        cards = cull.load_input_deck(source)
        with AnkiPackage(source) as package:
            package.conn.execute("UPDATE cards SET due = 42")
            package.conn.commit()
            package.touch()
            package.save()
        before = _cards(source)
        output = tmp_path / "core.apkg"

        counts = coreflow.write_deck(
            source, cards, [0], output, "Core Deck", core.FRENCH_CORE_PROFILE
        )

        after = _cards(output)
        note_id = cards[0].note_id
        assert counts == {"notes": 1, "media": 1, "items": 1}
        assert after[note_id] == before[note_id]
        assert after[note_id][1] == 42

    def test_gate_output_model_fails_on_a_spare_field(self, tmp_path: Path) -> None:
        _run, output, profile = _run_core(tmp_path)
        _add_spare_field(output, "Difficulty")

        passed, message = coregates.gate_output_model(
            coregates.CoreFiles(output, profile)
        )

        assert not passed
        assert "Difficulty" in message
