"""Tests for the select pipeline of ``anki fix yue``: request, names, and gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.languages.cantonese import select_pipeline
from anki_tools.languages.cantonese.frequency import cifu_path
from anki_tools.languages.cantonese.select import SelectionOptions
from anki_tools.languages.cantonese.select_pipeline import (
    SelectRequest,
    check_select,
    default_output,
    gate_kept_reasons,
    gate_no_duplicates,
    gate_rank_order,
    gate_select_rows,
    output_stem,
    rank_csv_path,
    side_paths,
)

BASE = Path("apkg/LTL Cantonese Deck Level 1 - Short")
LTL_OUTPUT = BASE.with_name(BASE.name + "_reordered.apkg")
LTL_RANK_CSV = BASE.with_name(BASE.name + ".rank.csv")
LTL_SELECT_CSV = BASE.with_name(BASE.name + ".select.csv")
LTL_PRESENT = (
    LTL_OUTPUT.exists() and LTL_RANK_CSV.exists() and LTL_SELECT_CSV.exists()
)
CIFU_PRESENT = cifu_path().exists()

SHARED_FIELDS = (
    "max_sentences",
    "per_word",
    "max_uses",
    "top_words",
    "order",
    "min_han",
    "max_han",
    "max_unknown",
    "known_top",
    "similarity_threshold",
    "final_mark",
)


def test_request_defaults_match_the_filter() -> None:
    request = SelectRequest(input_path=Path("a.apkg"), output_path=Path("b.apkg"))
    options = SelectionOptions()

    for name in SHARED_FIELDS:
        assert getattr(request, name) == getattr(options, name), name


def test_options_carry_the_request_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select_pipeline, "read_names", lambda *_args: ())
    request = SelectRequest(
        input_path=Path("a.apkg"),
        output_path=Path("b.apkg"),
        per_word=3,
        max_uses=5,
        order="score",
    )

    options = request.options()

    assert options.per_word == 3
    assert options.max_uses == 5
    assert options.order == "score"


def test_default_output_name() -> None:
    assert default_output(Path("x.apkg")) == Path("x_reordered.apkg")


def test_output_stem_strips_the_suffix() -> None:
    assert output_stem(Path("x_reordered.apkg")) == Path("x")


def test_output_stem_keeps_an_unsuffixed_name() -> None:
    assert output_stem(Path("x.apkg")) == Path("x")


def test_side_paths() -> None:
    select_csv, coverage_csv = side_paths(Path("x_reordered.apkg"))

    assert select_csv == Path("x.select.csv")
    assert coverage_csv == Path("x.coverage.csv")


def test_rank_csv_path() -> None:
    assert rank_csv_path(Path("x_reordered.apkg")) == Path("x.rank.csv")


def test_gate_kept_reasons_passes_on_a_clean_row() -> None:
    rows = [{"sentence": "你好", "kept": "1", "reason": ""}]

    gate, passed, detail = gate_kept_reasons(rows)

    assert gate == "G5"
    assert passed is True
    assert detail == "1 kept sentences fail no rule"


def test_gate_kept_reasons_fails_on_a_reason() -> None:
    rows = [{"sentence": "你好", "kept": "1", "reason": "similar"}]

    gate, passed, _detail = gate_kept_reasons(rows)

    assert gate == "G5"
    assert passed is False


def test_gate_no_duplicates_passes_on_distinct_lengths() -> None:
    rows = [{"sentence": "你好"}, {"sentence": "佢哋走咗"}]

    gate, passed, _detail = gate_no_duplicates(rows)

    assert gate == "G6"
    assert passed is True


def test_gate_no_duplicates_fails_on_a_repeat() -> None:
    rows = [{"sentence": "你好"}, {"sentence": "你好"}]

    gate, passed, _detail = gate_no_duplicates(rows)

    assert gate == "G6"
    assert passed is False


def test_gate_select_rows_passes_on_a_match() -> None:
    rank_rows = [{"sentence": "你好"}, {"sentence": "早晨"}]
    select_rows = [
        {"sentence": "你好", "kept": "1"},
        {"sentence": "早晨", "kept": "1"},
    ]

    gate, passed, _detail = gate_select_rows(rank_rows, select_rows)

    assert gate == "G7"
    assert passed is True


def test_gate_select_rows_fails_on_a_missing_row() -> None:
    rank_rows = [{"sentence": "你好"}, {"sentence": "早晨"}]
    select_rows = [{"sentence": "你好", "kept": "1"}]

    gate, passed, _detail = gate_select_rows(rank_rows, select_rows)

    assert gate == "G7"
    assert passed is False


def test_gate_rank_order_passes_on_ascending() -> None:
    rows = [{"hardest_rank": "1"}, {"hardest_rank": "5"}]

    gate, passed, _detail = gate_rank_order(rows)

    assert gate == "G9"
    assert passed is True


def test_gate_rank_order_fails_on_a_drop() -> None:
    rows = [{"hardest_rank": "5"}, {"hardest_rank": "2"}]

    gate, passed, _detail = gate_rank_order(rows)

    assert gate == "G9"
    assert passed is False


@pytest.mark.skipif(
    not (LTL_PRESENT and CIFU_PRESENT),
    reason="the LTL selection output files and the Cifu cache are absent",
)
def test_check_select_passes_on_the_ltl_output(capsys) -> None:
    assert check_select(LTL_OUTPUT) == 0

    lines = capsys.readouterr().out.splitlines()
    for gate in ("G5", "G6", "G7", "G9", "G12"):
        assert any(
            line.startswith(gate) and "PASS" in line for line in lines
        ), gate
