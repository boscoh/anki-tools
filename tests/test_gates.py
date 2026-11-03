"""Tests for the shared gate runner and the one report format."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anki_tools.gate_files import GateFileReader
from anki_tools.ranking import coregates, cullgates, gates
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.cullflow import CullProfile
from anki_tools.ranking.scan import CoreProfile
from tests.cull_fixture import gate_reader, run_core, run_cull


def test_a_missing_file_fails_only_the_gates_that_read_it(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.csv"
    coverage.write_text("word\nchat\n", encoding="utf-8")
    files = GateFileReader(
        {
            "rows": lambda: gates.read_csv(tmp_path / "rows.csv"),
            "coverage": lambda: gates.read_csv(coverage),
        }
    )
    results = gates.run_gates(
        [
            gates.Gate("K1", lambda reader: (bool(reader.rows()), "rows")),
            gates.Gate("K2", lambda reader: (bool(reader.coverage()), "coverage")),
        ],
        files,
    )

    assert [(result.name, result.passed) for result in results] == [
        ("K1", False),
        ("K2", True),
    ]
    assert "rows.csv" in results[0].message

    with pytest.raises(FileNotFoundError) as first:
        files.rows()
    with pytest.raises(FileNotFoundError) as second:
        files.rows()

    assert first.value is second.value


def test_the_report_prints_one_table(capsys) -> None:
    assert gates.report([gates.GateResult("K1", True, "ok")]) == 0
    passed = capsys.readouterr().out

    assert "Gate" in passed
    assert "K1" in passed
    assert "PASS" in passed
    assert "all 1 gates passed" in passed

    assert gates.report([gates.GateResult("K1", False, "bad")]) == 1
    failed = capsys.readouterr().out

    assert "FAIL" in failed
    assert "1 of 1 gates failed: K1" in failed


@pytest.mark.parametrize(
    ("counts", "passed"),
    [(["2", "1", "1", "0", "0"], True), (["2", "1", "0", "0", "0"], False)],
)
def test_coverage_floor_counts_absent_targets_as_not_covered(
    counts: list[str], passed: bool
) -> None:
    files = gate_reader(
        SimpleNamespace(coverage_floor=0.6),
        coverage=[{"count": count} for count in counts],
    )

    ok, message = coregates.gate_coverage_floor(files)

    assert ok is passed
    assert "floor 0.60" in message


def _core_word_files(shown: dict[int, str]) -> Any:
    """Give a reader of two kept cards whose Core words may be swapped."""
    return gate_reader(
        SimpleNamespace(
            hits=lambda sentence, forms: {form for form in forms if form in sentence}
        ),
        coverage=[
            {"key": "S1:chat", "tier": "S", "word": "chat"},
            {"key": "S2:chien", "tier": "S", "word": "chien"},
        ],
        rows=[
            {"kept": "1", "note_id": "1", "sentence": "le chat dort", "targets": "S1:chat"},
            {"kept": "1", "note_id": "2", "sentence": "le chien mange", "targets": "S2:chien"},
        ],
        output={
            "cards": [
                SimpleNamespace(note_id=1, sentence="le chat dort"),
                SimpleNamespace(note_id=2, sentence="le chien mange"),
            ]
        },
        tokens=[[], []],
        fields=["Audio", "Sentence", "English", "Core", "CoreGloss"],
        notes={note: ["", "", "", word, ""] for note, word in shown.items()},
    )


@pytest.mark.parametrize(
    ("shown", "passed"),
    [
        ({1: "chat", 2: "chien"}, True),
        ({1: "chat", 2: ""}, True),
        ({1: "chien", 2: "chat"}, False),
    ],
)
def test_target_in_sentence_also_checks_the_shown_core_word(
    shown: dict[int, str], passed: bool
) -> None:
    ok, _message = coregates.gate_target_in_sentence(_core_word_files(shown))

    assert ok is passed


@pytest.fixture(scope="module")
def core_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CoreProfile]:
    _run, output, profile = run_core(tmp_path_factory.mktemp("core_run"))
    return output, profile


def _copy_run(output: Path, tmp_path: Path) -> Path:
    shutil.copytree(output.parent, tmp_path / "run")
    return tmp_path / "run" / output.name


def _failed(table: tuple[gates.Gate, ...], files: GateFileReader) -> list[str]:
    results = gates.run_gates(table, files)
    return sorted(result.name for result in results if not result.passed)


def _failed_core(output: Path, profile: CoreProfile) -> list[str]:
    return _failed(coregates.GATES, coregates.CoreFiles(output, profile))


def _failed_cull(output: Path, profile: CullProfile) -> list[str]:
    return _failed(cullgates.GATES, cullgates.CullFiles(output, profile))


def _corrupt(run: Path, side: str, old: str, new: str) -> None:
    path = run.parent / side
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_the_core_gates_pass_on_a_real_core_run(
    core_run: tuple[Path, CoreProfile], tmp_path: Path
) -> None:
    output, profile = core_run

    assert _failed_core(_copy_run(output, tmp_path), profile) == []


@pytest.mark.parametrize(
    ("side", "old", "new", "failed"),
    [
        pytest.param(
            "core.core.csv",
            "Le chat dort.,",
            "Le chat dort ici.,",
            ["K5"],
            id="K5: the CSV sentence differs from the output",
        ),
        pytest.param(
            "core.coverage.csv",
            "S1:chat,chat,",
            "S1:chat,cha,",
            ["K13"],
            id="K13: the target form is not a word of the sentence",
        ),
        pytest.param(
            "core.coverage.csv",
            "S,1,1,1,covered",
            "S,1,1,0,covered",
            ["K23"],
            id="K23: the coverage count is 0",
        ),
        pytest.param(
            "core.coverage.csv",
            "S1:chat,chat,",
            "S1:chat,lapin,",
            ["K13", "K24"],
            id="K24: the form is absent from the text",
        ),
    ],
)
def test_a_corrupt_core_output_fails_the_named_gates(
    core_run: tuple[Path, CoreProfile],
    tmp_path: Path,
    side: str,
    old: str,
    new: str,
    failed: list[str],
) -> None:
    output, profile = core_run
    run = _copy_run(output, tmp_path)
    _corrupt(run, side, old, new)

    assert _failed_core(run, profile) == failed


def _same_lemma_tokens(cards: list[Any]) -> list[list[tuple[str, str]]]:
    return [[("chat", "VERB")] for _ in cards]


@pytest.fixture(scope="module")
def twin_core_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, CoreProfile]:
    """Run a core run that keeps two cards with the same lemma set."""
    options = CoreOptions(
        max_cards=10,
        per_word=2,
        ubiquitous_share=1.0,
        completion_cards=0,
        tail_cards=1,
        similarity=1.1,
    )
    _run, output, profile = run_core(
        tmp_path_factory.mktemp("twin_core_run"),
        tokenize_deck=_same_lemma_tokens,
        default_options=options,
    )
    return output, profile


def test_k6_fails_when_the_run_keeps_two_cards_with_the_same_lemmas(
    twin_core_run: tuple[Path, CoreProfile], tmp_path: Path
) -> None:
    output, profile = twin_core_run

    assert _failed_core(_copy_run(output, tmp_path), profile) == ["K6"]


def _minou_tokens(cards: list[Any]) -> list[list[tuple[str, str]]]:
    """Tag the word chat as the noun lemma minou and the word le as a determiner."""
    words = [[word.strip(".").casefold() for word in card.sentence.split()] for card in cards]
    return [
        [("minou" if word == "chat" else word, "DET" if word == "le" else "NOUN") for word in line]
        for line in words
    ]


@pytest.fixture(scope="module")
def lemma_target_core_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, CoreProfile]:
    """Run a core run with a tier F target whose lemma is not a form of the sentence."""
    _run, output, profile = run_core(
        tmp_path_factory.mktemp("lemma_target_core_run"),
        core_words=1,
        tokenize_deck=_minou_tokens,
        word_table=lambda: [(1, "lapin", "rabbit")],
    )
    return output, profile


def test_k13_tests_a_tier_f_target_by_its_lemma(
    lemma_target_core_run: tuple[Path, CoreProfile], tmp_path: Path
) -> None:
    output, profile = lemma_target_core_run
    run = _copy_run(output, tmp_path)

    assert "F:minou,minou," in (run.parent / "core.coverage.csv").read_text(encoding="utf-8")
    assert _failed_core(run, profile) == []


def test_check_core_returns_the_exit_code_of_the_gate_table(
    core_run: tuple[Path, CoreProfile], tmp_path: Path
) -> None:
    output, profile = core_run
    run = _copy_run(output, tmp_path)

    assert coregates.check_core(run, profile) == 0

    _corrupt(run, "core.coverage.csv", "S,1,1,1,covered", "S,1,1,0,covered")

    assert coregates.check_core(run, profile) == 1


@pytest.fixture(scope="module")
def cull_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CullProfile]:
    return run_cull(tmp_path_factory.mktemp("cull_run"))


def test_the_cull_gates_pass_on_a_real_cull_run(
    cull_run: tuple[Path, CullProfile], tmp_path: Path
) -> None:
    output, profile = cull_run

    assert _failed_cull(_copy_run(output, tmp_path), profile) == []


@pytest.mark.parametrize(
    ("side", "old", "new", "failed"),
    [
        pytest.param(
            "gated.cull.csv",
            "a4.mp3,1,,",
            "a4.mp3,0,max_cards,",
            ["R1"],
            id="R1: a card left in the pool and the count is below the cap",
        ),
        pytest.param(
            "gated.coverage.csv",
            "chat,1,5,",
            "chien,1,5,",
            ["R2"],
            id="R2: a lemma with 5 uses is not exempt",
        ),
        pytest.param(
            "gated.cull.csv",
            "Le chat aime la fille.,",
            "Le chat.,",
            ["R5"],
            id="R5: a kept card is under the length window",
        ),
    ],
)
def test_a_corrupt_cull_output_fails_the_named_gates(
    cull_run: tuple[Path, CullProfile],
    tmp_path: Path,
    side: str,
    old: str,
    new: str,
    failed: list[str],
) -> None:
    output, profile = cull_run
    run = _copy_run(output, tmp_path)
    _corrupt(run, side, old, new)

    assert _failed_cull(run, profile) == failed


def test_check_cull_returns_the_exit_code_of_the_gate_table(
    cull_run: tuple[Path, CullProfile], tmp_path: Path
) -> None:
    output, profile = cull_run
    run = _copy_run(output, tmp_path)

    assert cullgates.check_cull(run, profile) == 0

    _corrupt(run, "gated.cull.csv", "a4.mp3,1,,", "a4.mp3,0,max_cards,")

    assert cullgates.check_cull(run, profile) == 1
