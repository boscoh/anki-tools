"""Tests for the French and Cantonese fix commands: rank mode and select mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools import cli


def _fake_deck() -> SimpleNamespace:
    return SimpleNamespace(
        sentences=[object()],
        ranked=[object()],
        stats={"cards_reordered": 1, "cards_removed": 0},
    )


@pytest.fixture
def deck_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.apkg"
    path.write_bytes(b"")
    return path


class TestSelectMode:
    def test_select_routes_to_the_cull(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        requests = []

        def fake_run_cull(request):
            requests.append(request)
            return SimpleNamespace(cards=1)

        monkeypatch.setattr(cli.french_cull, "run_cull", fake_run_cull)

        cli.fr(deck_file, select=True)

        assert len(requests) == 1
        request = requests[0]
        assert request.input_path == deck_file
        assert request.output_path == deck_file.parent / "input_culled.apkg"
        assert request.no_reserve is False

    def test_select_uses_the_given_output(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = []
        monkeypatch.setattr(
            cli.french_cull, "run_cull", lambda request: requests.append(request)
        )
        output = deck_file.parent / "small.apkg"

        cli.fr(deck_file, select=True, output=output)

        assert requests[0].output_path == output

    def test_no_reserve_is_passed(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = []
        monkeypatch.setattr(
            cli.french_cull, "run_cull", lambda request: requests.append(request)
        )

        cli.fr(deck_file, select=True, reserve=False, cards=500)

        assert requests[0].no_reserve is True
        assert requests[0].max_cards == 500

    def test_model_id_with_select_prints_an_error(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        def fail(_request):
            raise AssertionError("run_cull must not run")

        monkeypatch.setattr(cli.french_cull, "run_cull", fail)

        cli.fr(deck_file, select=True, model_id=7)

        assert "Error: --model-id" in capsys.readouterr().out

    def test_check_with_select_calls_the_gates(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checked = []
        monkeypatch.setattr(
            cli.french_cull, "check_cull", lambda path: checked.append(path) or 0
        )

        cli.fr(deck_file, select=True, check=True)

        assert checked == [deck_file.parent / "input_culled.apkg"]

    def test_check_with_select_exits_on_a_gate_failure(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli.french_cull, "check_cull", lambda _path: 1)

        with pytest.raises(SystemExit):
            cli.fr(deck_file, select=True, check=True)


class TestRankMode:
    def test_rank_routes_to_the_pipeline(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def fake_run(paths, **kwargs):
            calls.append((paths, kwargs))
            return _fake_deck()

        monkeypatch.setattr(cli, "run_ranked_deck", fake_run)

        cli.fr(deck_file)

        paths, kwargs = calls[0]
        assert paths.input_path == deck_file
        assert paths.output_path == deck_file.parent / "input_reordered.apkg"
        assert paths.ranking_path == deck_file.parent / "input.rank.csv"
        assert kwargs["text_candidates"] == ("French",)

    def test_rank_uses_the_given_output(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def fake_run(paths, **_kwargs):
            calls.append(paths)
            return _fake_deck()

        monkeypatch.setattr(cli, "run_ranked_deck", fake_run)
        output = deck_file.parent / "ranked.apkg"

        cli.fr(deck_file, output=output)

        assert calls[0].output_path == output

    def test_check_without_select_prints_an_error(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        def fail(*_args, **_kwargs):
            raise AssertionError("the pipeline must not run")

        monkeypatch.setattr(cli, "run_ranked_deck", fail)

        cli.fr(deck_file, check=True)

        assert "Error: --check needs --select or --core." in capsys.readouterr().out

    def test_a_missing_input_prints_an_error(
        self, tmp_path: Path, capsys
    ) -> None:
        cli.fr(tmp_path / "absent.apkg")

        assert "Error: File not found" in capsys.readouterr().out


@pytest.fixture
def no_jyutping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fix_jyutping", lambda *_args, **_kwargs: None)


class TestCantoneseRankMode:
    def test_rank_routes_to_the_pipeline(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, no_jyutping: None
    ) -> None:
        calls = []

        def fake_run(paths, **kwargs):
            calls.append((paths, kwargs))
            return _fake_deck()

        monkeypatch.setattr(cli, "run_ranked_deck", fake_run)

        cli.yue(deck_file, no_select=True)

        paths, kwargs = calls[0]
        assert paths.input_path == deck_file
        assert paths.output_path == deck_file.parent / "input_reordered.apkg"
        assert paths.ranking_path == deck_file.parent / "input.rank.csv"
        assert kwargs["romanization_candidates"] == ("Jyutping",)
        assert kwargs["remove_unranked"] is False

    def test_rank_uses_the_given_output(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, no_jyutping: None
    ) -> None:
        calls = []

        def fake_run(paths, **_kwargs):
            calls.append(paths)
            return _fake_deck()

        monkeypatch.setattr(cli, "run_ranked_deck", fake_run)
        output = deck_file.parent / "ranked.apkg"

        cli.yue(deck_file, output=output, no_select=True)

        assert calls[0].output_path == output

    def test_an_empty_extraction_prints_no_sentences(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(
            cli,
            "run_ranked_deck",
            lambda _paths, **_kwargs: SimpleNamespace(
                sentences=[], ranked=[], stats={}
            ),
        )

        def fail(*_args, **_kwargs):
            raise AssertionError("fix_jyutping must not run")

        monkeypatch.setattr(cli, "fix_jyutping", fail)

        cli.yue(deck_file, no_select=True)

        assert "No sentences found!" in capsys.readouterr().out


def _capture_select(
    monkeypatch: pytest.MonkeyPatch,
    report: SimpleNamespace | None = None,
    ranked_count: int = 0,
    stats: dict[str, int] | None = None,
) -> list[cli.cantonese_select.SelectRequest]:
    """Replace the select run of the CLI with a stub that records the request.

    :param monkeypatch: The pytest monkeypatch fixture.
    :param report: The report that the stub returns.
    :param ranked_count: Sentences that the stub reports as ranked.
    :param stats: Card counts that the stub reports.
    :returns: The list that the stub appends each request to.
    """
    requests: list[cli.cantonese_select.SelectRequest] = []

    def fake_run(request: cli.cantonese_select.SelectRequest):
        requests.append(request)
        return cli.cantonese_select.SelectRun(
            cli.cantonese_select.rank_paths(request.input_path, request.output_path),
            ranked_count,
            report,
            stats or {},
        )

    monkeypatch.setattr(cli.cantonese_select, "run_select", fake_run)
    return requests


class TestCantoneseSelectMode:
    def test_select_builds_the_request(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _capture_select(monkeypatch)

        cli.yue(deck_file)

        request = requests[0]
        assert request.input_path == deck_file
        assert request.output_path == deck_file.parent / "input_reordered.apkg"
        assert request.max_sentences is None
        assert request.max_uses == 8
        assert request.freq == "blend"
        assert request.per_word == 2
        assert request.final_mark is True
        assert request.freq_report is False
        assert request.model_id is None

    def test_cards_sets_the_select_cap(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _capture_select(monkeypatch)

        cli.yue(deck_file, cards=300)

        assert requests[0].max_sentences == 300

    def test_select_uses_the_given_output(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _capture_select(monkeypatch)
        output = deck_file.parent / "small.apkg"

        cli.yue(deck_file, output=output)

        assert requests[0].output_path == output

    def test_select_passes_the_tuning_survivors(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _capture_select(monkeypatch)

        cli.yue(deck_file, max_uses=3, freq="written", model_id=7, freq_report=True)

        request = requests[0]
        assert request.max_uses == 3
        assert request.freq == "written"
        assert request.model_id == 7
        assert request.freq_report is True

    def test_select_with_no_report_prints_no_sentences(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _capture_select(monkeypatch)

        def fail(*_args, **_kwargs):
            raise AssertionError("fix_jyutping must not run")

        monkeypatch.setattr(cli, "fix_jyutping", fail)

        cli.yue(deck_file)

        assert "No sentences found!" in capsys.readouterr().out

    def test_select_prints_the_csv_names(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _capture_select(
            monkeypatch,
            report=SimpleNamespace(),
            ranked_count=5,
            stats={"cards_reordered": 3, "cards_removed": 1},
        )
        monkeypatch.setattr(cli, "fix_jyutping", lambda *_args, **_kwargs: None)

        cli.yue(deck_file)

        out = capsys.readouterr().out
        assert "Select CSV:" in out
        assert "Coverage CSV:" in out
        assert str(deck_file.parent / "input.select.csv") in out
        assert str(deck_file.parent / "input.coverage.csv") in out


class TestCantoneseCoreMode:
    def test_core_cards_and_weight(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = []
        monkeypatch.setattr(
            cli.cantonese_core, "run_core", lambda request: requests.append(request)
        )

        cli.yue(deck_file, core=True, cards=42)

        assert requests[0].options.max_cards == 42
        assert requests[0].spoken_weight == cli.cantonese_core.SPOKEN_WEIGHT


class TestSpanishMode:
    def test_check_without_core_prints_an_error(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        def fail(*_args, **_kwargs):
            raise AssertionError("the pipeline must not run")

        monkeypatch.setattr(cli, "run_ranked_deck", fail)

        cli.es(deck_file, check=True)

        assert "Error: --check needs --core." in capsys.readouterr().out

    def test_check_with_core_calls_the_gates(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checked = []
        monkeypatch.setattr(
            cli.spanish_core, "check_core", lambda path: checked.append(path) or 0
        )

        cli.es(deck_file, core=True, check=True)

        assert checked == [deck_file.parent / "input_core.apkg"]

    def test_check_with_core_exits_on_a_gate_failure(
        self, deck_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli.spanish_core, "check_core", lambda _path: 1)

        with pytest.raises(SystemExit):
            cli.es(deck_file, core=True, check=True)
