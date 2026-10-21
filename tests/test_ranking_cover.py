"""Tests for the shared greedy cover."""

from __future__ import annotations

from typing import Any

from anki_tools.ranking.cover import greedy_cover


def _key(index: int, value: float) -> tuple[Any, ...]:
    return (-value, index)


def _no_reject(_index: int) -> str | None:
    return None


def _no_pick(_index: int, _value: float) -> None:
    return None


class TestGreedyCover:
    def test_precheck_runs_before_the_gain(self) -> None:
        seen: list[tuple[str, int]] = []
        gains = {0: 1.0, 1: 2.0}

        def gain(index: int) -> float:
            seen.append(("gain", index))
            return gains[index]

        def precheck(index: int) -> str | None:
            seen.append(("precheck", index))
            return "use_cap" if index == 0 else None

        kept, reasons = greedy_cover(
            [0, 1],
            gain=gain,
            key=_key,
            reject=_no_reject,
            on_pick=_no_pick,
            precheck=precheck,
        )

        assert kept == [1]
        assert reasons == {0: "use_cap"}
        assert seen == [
            ("gain", 0),
            ("gain", 1),
            ("precheck", 1),
            ("gain", 1),
            ("precheck", 0),
        ]

    def test_a_zero_gain_gives_no_gain_with_include_zero(self) -> None:
        kept, reasons = greedy_cover(
            [0, 1],
            gain=lambda index: 0.0 if index == 0 else 1.0,
            key=_key,
            reject=_no_reject,
            on_pick=_no_pick,
            include_zero=True,
        )

        assert kept == [1]
        assert reasons == {0: "no_gain"}

    def test_a_zero_gain_stays_out_of_the_heap_without_include_zero(self) -> None:
        kept, reasons = greedy_cover(
            [0, 1],
            gain=lambda index: 0.0 if index == 0 else 1.0,
            key=_key,
            reject=_no_reject,
            on_pick=_no_pick,
        )

        assert kept == [1]
        assert reasons == {}

    def test_a_stale_gain_is_pushed_back(self) -> None:
        gains = {0: 5.0, 1: 5.0}
        measured: list[tuple[int, float]] = []

        def on_pick(index: int, _value: float) -> None:
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
        assert measured == [(0, 5.0), (1, 2.0), (1, 2.0)]

    def test_limit_stops_the_cover(self) -> None:
        kept, reasons = greedy_cover(
            [0, 1, 2],
            gain=lambda index: 3.0 - index,
            key=_key,
            reject=_no_reject,
            on_pick=_no_pick,
            limit=1,
        )

        assert kept == [0]
        assert reasons == {}

    def test_the_reject_reason_is_recorded(self) -> None:
        kept, reasons = greedy_cover(
            [0, 1],
            gain=lambda index: 2.0 - index,
            key=_key,
            reject=lambda index: "similar" if index == 0 else None,
            on_pick=_no_pick,
        )

        assert kept == [1]
        assert reasons == {0: "similar"}

    def test_the_callbacks_receive_the_gain(self) -> None:
        picked: list[tuple[int, float]] = []
        measured: list[tuple[int, float]] = []

        kept, reasons = greedy_cover(
            [0],
            gain=lambda _index: 4.0,
            key=_key,
            reject=_no_reject,
            on_pick=lambda index, value: picked.append((index, value)),
            on_measure=lambda index, value: measured.append((index, value)),
        )

        assert kept == [0]
        assert reasons == {}
        assert picked == [(0, 4.0)]
        assert measured == [(0, 4.0)]
