"""Tests for the shared gate runner and the one report format."""

from __future__ import annotations

from anki_tools.ranking import gates


class _Files:
    """Minimal file reader: the runner only passes it to the gate."""

    profile = None


def test_the_runner_reports_a_gate_that_cannot_read_a_file() -> None:
    def broken(_files):
        raise ValueError("bad file")

    results = gates.run_gates(
        [gates.Gate("K1", broken), gates.Gate("K2", lambda _files: (True, "ok"))],
        _Files(),
    )

    assert [(result.name, result.passed) for result in results] == [
        ("K1", False),
        ("K2", True),
    ]
    assert "bad file" in results[0].message


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
