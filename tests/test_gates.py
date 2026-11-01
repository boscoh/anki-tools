"""Tests for the shared gate runner and the one report format."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools.gate_files import GateFileReader
from anki_tools.ranking import coregates, gates


class _Files:
    """Minimal file reader: the runner only passes it to the gate."""

    profile = None


def test_the_runner_reports_a_gate_that_cannot_read_a_file() -> None:
    def broken(_files):
        raise ValueError("bad file")

    results = gates.run_gates(
        [gates.Gate("K1", broken), gates.Gate("K2", lambda _files: (True, "ok"))],
        _Files(),  # type: ignore[arg-type]
    )

    assert [(result.name, result.passed) for result in results] == [
        ("K1", False),
        ("K2", True),
    ]
    assert "bad file" in results[0].message



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

    assert gates.report([gates.GateResult("G1", False, "bad")]) == 1
    failed = capsys.readouterr().out

    assert "FAIL" in failed
    assert "1 of 1 gates failed: G1" in failed


class _CoverageFiles:
    """Reader with one coverage table and one profile floor."""

    def __init__(self, counts: list[str], floor: float) -> None:
        self._rows = [{"count": count} for count in counts]
        self.profile = SimpleNamespace(coverage_floor=floor)

    def coverage(self) -> list[dict[str, str]]:
        return self._rows


@pytest.mark.parametrize(
    ("counts", "passed"),
    [(["2", "1", "1", "0", "0"], True), (["2", "1", "0", "0", "0"], False)],
)
def test_coverage_floor_counts_absent_targets_as_not_covered(
    counts: list[str], passed: bool
) -> None:
    ok, message = coregates.gate_coverage_floor(_CoverageFiles(counts, 0.6))  # type: ignore[arg-type]

    assert ok is passed
    assert "floor 0.60" in message


class _CoreWordFiles:
    """Reader with two kept cards whose Core words may be swapped."""

    def __init__(self, shown: dict[int, str]) -> None:
        self.profile = SimpleNamespace(
            hits=lambda sentence, forms: {form for form in forms if form in sentence}
        )
        self._shown = shown
        self._cards = [
            SimpleNamespace(note_id=1, sentence="le chat dort"),
            SimpleNamespace(note_id=2, sentence="le chien mange"),
        ]

    def coverage(self) -> list[dict[str, str]]:
        return [
            {"key": "S1:chat", "tier": "S", "word": "chat"},
            {"key": "S2:chien", "tier": "S", "word": "chien"},
        ]

    def rows(self) -> list[dict[str, str]]:
        return [
            {"kept": "1", "note_id": "1", "sentence": "le chat dort", "targets": "S1:chat"},
            {"kept": "1", "note_id": "2", "sentence": "le chien mange", "targets": "S2:chien"},
        ]

    def output(self) -> dict[str, list]:
        return {"cards": self._cards}

    def tokens(self) -> list[list]:
        return [[], []]

    def fields(self) -> list[str]:
        return ["Audio", "Sentence", "English", "Core", "CoreGloss"]

    def notes(self) -> dict[int, list[str]]:
        return {note: ["", "", "", word, ""] for note, word in self._shown.items()}


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
    ok, _message = coregates.gate_target_in_sentence(_CoreWordFiles(shown))  # type: ignore[arg-type]

    assert ok is passed
