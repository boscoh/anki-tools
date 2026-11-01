"""Tests for the shared greedy cover."""

from __future__ import annotations

from typing import Any

import pytest

from anki_tools.ranking.cover import greedy_cover


def _key(index: int, value: float) -> tuple[Any, ...]:
    return (-value, index)


def _no_reject(_index: int) -> str | None:
    return None


def _no_pick(_index: int, _value: float) -> None:
    return None


@pytest.mark.parametrize(
    "gains, options, kept, reasons",
    [
        pytest.param(
            {0: 1.0, 1: 2.0},
            {"precheck": {0: "use_cap"}.get},
            [1],
            {0: "use_cap"},
            id="precheck-rejects-with-its-reason",
        ),
        pytest.param(
            {0: 0.0, 1: 1.0},
            {"include_zero": True},
            [1],
            {0: "no_gain"},
            id="zero-gain-is-no_gain-with-include_zero",
        ),
        pytest.param(
            {0: 0.0, 1: 1.0},
            {},
            [1],
            {},
            id="zero-gain-drops-out-without-include_zero",
        ),
        pytest.param(
            {0: 3.0, 1: 2.0, 2: 1.0},
            {"limit": 1},
            [0],
            {},
            id="limit-stops-the-cover",
        ),
        pytest.param(
            {0: 2.0, 1: 1.0},
            {"reject": {0: "similar"}.get},
            [1],
            {0: "similar"},
            id="reject-reason-is-recorded",
        ),
    ],
)
def test_the_cover_keeps_and_rejects_by_gain(
    gains: dict[int, float],
    options: dict[str, Any],
    kept: list[int],
    reasons: dict[int, str],
) -> None:
    kwargs: dict[str, Any] = {"reject": _no_reject} | options
    result = greedy_cover(
        list(gains), gain=gains.__getitem__, key=_key, on_pick=_no_pick, **kwargs
    )

    assert result == (kept, reasons)


def test_a_stale_gain_is_pushed_back_and_the_callbacks_see_each_gain() -> None:
    gains = {0: 5.0, 1: 5.0}
    picked: list[tuple[int, float]] = []
    measured: list[tuple[int, float]] = []

    def on_pick(index: int, value: float) -> None:
        picked.append((index, value))
        if index == 0:
            gains[1] = 2.0

    kept, reasons = greedy_cover(
        [0, 1],
        gain=lambda index: gains[index],
        key=_key,
        reject=_no_reject,
        on_pick=on_pick,
        on_measure=lambda index, value: measured.append((index, value)),
    )

    assert kept == [0, 1]
    assert reasons == {}
    assert picked == [(0, 5.0), (1, 2.0)]
    assert measured == [(0, 5.0), (1, 2.0), (1, 2.0)]
