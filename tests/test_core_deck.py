"""Tests for the core-word small deck (spec sections 2 to 10).

Every test is offline. The token lists are built by hand, so no test loads
spaCy, ffprobe, or the frequency list.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.french import core, cull
from anki_tools.languages.mandarin.core import MANDARIN_CORE_PROFILE
from anki_tools.ranking import assemble, coreflow
from anki_tools.ranking.core import (
    CoreCandidate,
    CoreOptions,
    CoreResult,
    card_cost,
    select,
    target_quotas,
)
from anki_tools.ranking.coreflow import (
    InputCard,
    coverage_rows_of,
    kept_order_hash,
    outside_lemmas,
    size_check,
)
from anki_tools.ranking.corelist import TargetWord, build_core_targets
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.scan import drop_silent_cards
from anki_tools.sentence_pack.english_sense import SenseRow
from tests.cull_fixture import (
    card,
    core_deck,
    core_target,
    run_core,
    write_core_package,
)


@pytest.mark.parametrize(
    ("options", "shares", "expected"),
    [
        pytest.param(
            {"ubiquitous_share": 0.25},
            {"S1:je": 0.9, "S2:eau": 0.1},
            {"S1:je": 1, "S2:eau": 2},
            id="a ubiquitous target gets one example",
        ),
        pytest.param(
            {"ubiquitous_share": 0.25},
            {"S1:a": 0.25, "S2:b": 0.26},
            {"S1:a": 2, "S2:b": 1},
            id="a share at the limit is not ubiquitous",
        ),
        pytest.param(
            {"third_examples": 1},
            {"S1:eau": 0.1, "S2:feu": 0.1},
            {"S1:eau": 3, "S2:feu": 2},
            id="the third example raises the top target",
        ),
    ],
)
def test_the_quota_follows_the_share_and_the_position(
    options: dict[str, Any], shares: dict[str, float], expected: dict[str, int]
) -> None:
    positions = {key: number for number, key in enumerate(shares, 1)}

    quotas = target_quotas(positions, shares, CoreOptions(per_word=2, **options))

    assert quotas == expected


COST_BASE: dict[str, Any] = {"lambda_new": 1.0, "lambda_repeat": 0.0, "lambda_len": 0.0}
COST_RANKS = {"common": 10, "mid": 2000, "rare": 5000}


@pytest.mark.parametrize(
    ("options", "outside", "introduced", "units", "expected"),
    [
        pytest.param({}, ("common",), set(), 5, 1.0, id="a frequent new word"),
        pytest.param({}, ("mid",), set(), 5, 1.8, id="a harder new word"),
        pytest.param({}, ("rare",), set(), 5, 3.0, id="a rare new word"),
        pytest.param({}, ("rare",), {"rare"}, 5, 0.0, id="a free repeat"),
        pytest.param(
            {"lambda_repeat": 0.5}, ("rare",), {"rare"}, 5, 0.5, id="a repeat cost"
        ),
        pytest.param(
            {"lambda_len": 0.25}, ("common",), set(), 9, 1.5, id="a long card"
        ),
    ],
)
def test_the_cost_follows_the_outside_words_and_the_length(
    options: dict[str, Any],
    outside: tuple[str, ...],
    introduced: set[str],
    units: int,
    expected: float,
) -> None:
    candidate = CoreCandidate(0, ("F:x",), outside, units)

    cost = card_cost(
        candidate, introduced, COST_RANKS, CoreOptions(**COST_BASE | options)
    )

    assert cost == pytest.approx(expected)


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

        result = select(
            candidates,
            positions=positions,
            quotas={"S1:a": 2, "S2:b": 1},
            ranks={},
            options=CoreOptions(max_cards=10, completion_cards=0),
        )

        assigned = [
            positions[result.assigned[index]]
            for index in result.order
            if index in result.assigned
        ]
        assert assigned == sorted(assigned)


@pytest.mark.parametrize(
    ("budget", "kept", "reasons"),
    [
        pytest.param(0, [0, 1], {}, id="no budget keeps both"),
        pytest.param(1, [0], {1: "outside_budget"}, id="a second outside lemma breaks the budget"),
        pytest.param(2, [0, 1], {}, id="a lemma count at the budget fits"),
    ],
)
def test_the_outside_budget_caps_the_distinct_outside_lemmas(
    budget: int, kept: list[int], reasons: dict[int, str]
) -> None:
    result = select(
        [
            CoreCandidate(0, ("S1:a",), ("x",), 5),
            CoreCandidate(1, ("S2:b",), ("y",), 5),
        ],
        positions={"S1:a": 1, "S2:b": 2},
        quotas={"S1:a": 1, "S2:b": 1},
        ranks={},
        options=CoreOptions(max_cards=10, completion_cards=0, outside_budget=budget),
    )

    assert result.kept == kept
    assert {
        index: reason
        for index, reason in result.reasons.items()
        if reason != "max_cards"
    } == reasons


@pytest.mark.parametrize(
    ("table", "lemmas", "senses", "core_words", "expected", "dropped"),
    [
        pytest.param(
            [(1, "je", "I"), (2, "je", "me")],
            [],
            {},
            0,
            [("S1:je", "I")],
            1,
            id="two glosses with one sense merge",
        ),
        pytest.param(
            [(1, "voler", "fly"), (2, "voler", "steal")],
            [],
            {},
            0,
            [("S1:voler", "fly"), ("S2:voler", "steal")],
            0,
            id="two senses keep two targets",
        ),
        pytest.param(
            [(1, "eau", "water")],
            ["eau", "pain"],
            {},
            5,
            [("S1:eau", "water"), ("F:pain", "")],
            0,
            id="a swadesh form is not added again",
        ),
        pytest.param(
            [],
            ["w1", "w2", "w3", "w4", "w5"],
            {},
            2,
            [("F:w1", ""), ("F:w2", "")],
            0,
            id="the core words cut the frequency list",
        ),
        pytest.param(
            [],
            ["faire"],
            {("F", "faire"): SenseRow(frozenset({"do", "make"}), first="do")},
            1,
            [("F:faire", "do")],
            0,
            id="a frequency lemma gets its first sense as the gloss",
        ),
    ],
)
def test_the_core_list_merges_swadesh_rows_and_adds_frequency_lemmas(
    table: list[tuple[int, str, str]],
    lemmas: list[str],
    senses: dict[tuple[str, str], SenseRow],
    core_words: int,
    expected: list[tuple[str, str]],
    dropped: int,
) -> None:
    frequency = [
        TargetWord(lemma, rank, 3, 1.0, 1.0) for rank, lemma in enumerate(lemmas, 1)
    ]

    targets, merged = build_core_targets(table, frequency, senses, core_words)

    assert [(target.key, target.gloss) for target in targets] == expected
    assert len(merged) == dropped


def test_a_card_without_a_media_file_leaves_and_the_rest_are_reindexed(
    tmp_path: Path,
) -> None:
    source = core_deck(tmp_path)
    cards = cull.load_input_deck(source)
    cards[0] = replace(cards[0], audio="missing.mp3")

    kept = drop_silent_cards(cards, source)

    assert [(card.sentence, card.index) for card in kept] == [
        ("Le chat mange.", 0),
        ("Le lapin court.", 1),
    ]


def test_a_core_form_is_not_an_outside_word() -> None:
    tokens = [[("chat", "NOUN"), ("manger", "VERB"), ("vite", "ADV")]]

    assert outside_lemmas(0, tokens, frozenset({"chat", "vite"})) == ("manger",)


def test_the_status_follows_the_count_the_quota_and_the_sense_row() -> None:
    targets = [
        core_target("S1:a", position=1),
        core_target("S2:b", position=2),
        core_target("S3:c", position=3),
        core_target("S4:d", position=4, senses=frozenset({"thing"})),
        core_target("F:x", tier="F", position=5),
    ]
    result = CoreResult(
        kept=[],
        uses=Counter({"S1:a": 2, "S2:b": 1, "S4:d": 0, "F:x": 1}),
        quotas={"S1:a": 2, "S2:b": 2, "S3:c": 2, "S4:d": 2, "F:x": 2},
    )
    held = {"S1:a": True, "S2:b": True, "S3:c": False, "S4:d": True, "F:x": True}

    rows = coverage_rows_of(targets, result, held, {key: 2 for key in held})

    assert {row["key"]: row["status"] for row in rows} == {
        "S1:a": "covered",
        "S2:b": "partial",
        "S3:c": "absent",
        "S4:d": "absent_sense",
        "F:x": "unchecked",
    }


@pytest.mark.parametrize(
    ("budget", "fits"),
    [pytest.param(50, False, id="a small budget"), pytest.param(300, True, id="a large budget")],
)
def test_a_small_budget_stops_the_run(budget: int, fits: bool) -> None:
    quotas = {f"t{index}": 2 for index in range(100)}

    if fits:
        size_check(quotas, budget)
    else:
        with pytest.raises(ValueError, match="cannot fit"):
            size_check(quotas, budget)


def test_the_order_hash_follows_the_kept_sentences() -> None:
    cards = [card("A.", index=0), card("B.", index=1)]
    digest = hashlib.sha256(b"B.\nA.").hexdigest()

    assert kept_order_hash(cards, [1, 0]) == f"sha256:{digest}"
    assert kept_order_hash(cards, [0, 1]) != kept_order_hash(cards, [1, 0])


def test_the_run_writes_one_deck_with_one_note_type(tmp_path: Path) -> None:
    run, output, profile = run_core(tmp_path)
    identity = DeckIdentity(output)
    assert run.manifest_path is not None
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))

    with AnkiPackage(output) as package:
        models = package.get_models(include_templates=True)
        decks = package.get_decks()
        cards = package.get_cards()
        notes = [str(row["flds"]).split("\x1f") for row in package.get_notes()]
        model_decks = json.loads(
            package.conn.execute("SELECT models FROM col").fetchone()[0]
        )

    model = next(iter(models.values()))
    assert {deck["name"] for deck in decks.values()} == {identity.title}
    assert [model["name"] for model in models.values()] == [identity.title]
    assert str(next(iter(model_decks.values()))["did"]) in decks
    assert [field["name"] for field in model["flds"]] == [
        "Audio",
        "Sentence",
        "English",
        "Core",
        "CoreGloss",
    ]
    assert [(t["qfmt"], t["afmt"]) for t in model["tmpls"]] == [
        assemble.deck_templates(profile)[0]
    ]
    assert len(cards) == manifest["counts"]["cards"] == 1
    assert {int(row["ord"]) for row in cards} == {0}
    assert [values[3:] for values in notes] == [["chat", "cat"]]
    assert notes[0][0] == "[sound:a0.mp3]"
    assert "<b>chat</b>" in notes[0][1]


def test_the_run_writes_the_side_files_the_manifest_and_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, output, profile = run_core(tmp_path)
    identity = DeckIdentity(output)
    assert run.manifest_path is not None
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    shown = capsys.readouterr().out

    assert run.sides["core.csv"] == identity.side("core.csv")
    assert run.sides["coverage.csv"] == identity.side("coverage.csv")
    assert "rank.csv" not in run.sides
    assert manifest["source"] == {
        "input": str(tmp_path / "input.apkg"),
        "deck": identity.title,
        "model": identity.model_name,
        "input_model": profile.input_model_name,
    }
    assert manifest["params"]["name"] == identity.title
    assert manifest["files"]["core_csv"].startswith("sha256:")
    assert "Cards:" in shown
    assert str(output) in shown
    assert "Dropped, no audio file:" in shown


def test_the_kept_card_keeps_its_note_id_and_due(tmp_path: Path) -> None:
    source = core_deck(tmp_path)
    cards = cull.load_input_deck(source)
    with AnkiPackage(source) as package:
        package.conn.execute("UPDATE cards SET due = 42")
        package.conn.commit()
        package.touch()
        package.save()
    note_id = cards[0].note_id

    counts = write_core_package(
        source, tmp_path / "core.apkg", cards, [0], core.FRENCH_CORE_PROFILE
    )

    with AnkiPackage(tmp_path / "core.apkg") as package:
        rows = package.conn.execute("SELECT nid, due FROM cards").fetchall()
    assert counts == {"notes": 1, "media": 1, "items": 1}
    assert {int(row["nid"]): int(row["due"]) for row in rows} == {note_id: 42}


def test_the_mandarin_build_filter_rejects_a_repeated_phrase() -> None:
    cards = [
        InputCard(1, "起来 起来.", "Get up! Get up!", "a.mp3", 0),
        InputCard(2, "起来吧大家.", "Get up.", "a.mp3", 1),
    ]

    kept, reasons = coreflow.filter_core_cards(cards, MANDARIN_CORE_PROFILE)

    assert kept == [1]
    assert reasons[0] == "repeat"


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        pytest.param('"I am here."', "I am here.", id="straight quotes"),
        pytest.param("\u201cI am here.\u201d", "I am here.", id="curly quotes"),
        pytest.param(" I am here. ", "I am here.", id="no quotes"),
        pytest.param('"He said "no" twice."', 'He said "no" twice.', id="inner quote"),
        pytest.param('He said "no"', 'He said "no"', id="quote at the end only"),
    ],
)
def test_clean_english_strips_one_pair_of_wrapping_quotes(
    raw: str, cleaned: str
) -> None:
    assert coreflow.clean_english(raw) == cleaned
