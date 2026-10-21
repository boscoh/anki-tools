"""Tests for the French fix command: rank mode and select mode."""

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

        cli.fr(deck_file, select=True, reserve=False, max_cards=500)

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
