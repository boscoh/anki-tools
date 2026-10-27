"""Tests for the fix commands: the rank, select, core, and check routes.

One language covers each route. The four commands pass their profile and label
to ``anki_tools.ranking.fixrun``, so these tests cross that one seam.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools import cli
from anki_tools.ranking import fixrun, identity


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


@pytest.fixture(autouse=True)
def no_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that reaches a pipeline it did not stub."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the pipeline must not run")

    monkeypatch.setattr(fixrun, "run_ranked_deck", refuse)
    monkeypatch.setattr(fixrun.cullflow, "run_cull", refuse)
    monkeypatch.setattr(fixrun.coreflow, "run_core", refuse)


def test_the_canonical_names_alias_the_short_codes() -> None:
    commands = cli.fix_app.resolved_commands()
    for short, canonical, function in (
        ("zh", "mandarin", cli.zh),
        ("fr", "french", cli.fr),
        ("yue", "cantonese", cli.yue),
        ("es", "spanish", cli.es),
    ):
        assert commands[short].default_command is function
        assert commands[canonical].default_command is function


def test_rank_goes_to_the_pipeline(
    deck_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        fixrun,
        "run_ranked_deck",
        lambda paths, **kwargs: calls.append((paths, kwargs)) or _fake_deck(),
    )
    output = deck_file.parent / "ranked.apkg"

    cli.fr(deck_file, rank=True, output=output)

    paths, kwargs = calls[0]
    assert paths.input_path == deck_file
    assert paths.output_path == output
    assert paths.ranking_path == deck_file.parent / "ranked.rank.csv"
    assert callable(kwargs["rank"])
    assert kwargs["text_candidates"] == ("French",)


def test_rank_defaults_the_output_name() -> None:
    assert identity.default_output("french", "rank") == Path("apkg/french_reordered.apkg")


def test_an_absent_input_prints_an_error(tmp_path: Path, capsys) -> None:
    cli.fr(tmp_path / "absent.apkg")

    assert "Error: File not found" in capsys.readouterr().out


def test_select_goes_to_the_cull(
    deck_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    monkeypatch.setattr(
        fixrun.cullflow,
        "run_cull",
        lambda request, run_profile: runs.append((request, run_profile)),
    )
    output = deck_file.parent / "small.apkg"

    cli.fr(deck_file, select=True, reserve=False, cards=500)
    cli.fr(deck_file, select=True, output=output)

    request, run_profile = runs[0]
    assert request.input_path == deck_file
    assert request.output_path == Path("apkg/french_1000.apkg")
    assert request.no_reserve is True
    assert request.max_cards == 500
    assert run_profile is cli.french_cull.FRENCH_CULL_PROFILE
    assert runs[1][0].output_path == output


def test_select_defaults_the_output_name() -> None:
    assert identity.default_output("french", "cull") == Path("apkg/french_1000.apkg")


def test_core_builds_the_request(
    deck_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    monkeypatch.setattr(
        fixrun.coreflow,
        "run_core",
        lambda request, run_profile: runs.append((request, run_profile)),
    )

    cli.fr(deck_file, core=True, cards=42)

    request, run_profile = runs[0]
    assert request.output_path == Path("apkg/french_core.apkg")
    assert request.core_words == cli.french_core.DEFAULT_CORE_WORDS
    assert request.options.max_cards == 42
    assert run_profile is cli.french_core.FRENCH_CORE_PROFILE


def test_check_without_a_mode_is_an_error(deck_file: Path) -> None:
    with pytest.raises(SystemExit, match="--check needs --select or --core"):
        cli.fr(deck_file, check=True)


def test_a_bare_command_is_an_error(deck_file: Path) -> None:
    with pytest.raises(SystemExit, match="choose a mode"):
        cli.fr(deck_file)


def test_two_modes_are_an_error(deck_file: Path) -> None:
    with pytest.raises(SystemExit, match="choose one mode"):
        cli.fr(deck_file, rank=True, core=True)


CHECK_ROUTES = [
    pytest.param(
        True,
        False,
        "check_cull",
        cli.french_cull.FRENCH_CULL_PROFILE,
        "apkg/french_1000.apkg",
        id="select",
    ),
    pytest.param(
        False,
        True,
        "check_core",
        cli.french_core.FRENCH_CORE_PROFILE,
        "apkg/french_core.apkg",
        id="core",
    ),
]


@pytest.mark.parametrize("select, core, checker, profile, output_name", CHECK_ROUTES)
def test_check_runs_the_gates_on_the_default_output(
    deck_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    select: bool,
    core: bool,
    checker: str,
    profile,
    output_name: str,
) -> None:
    checked = []
    monkeypatch.setattr(
        fixrun,
        checker,
        lambda path, run_profile: checked.append((path, run_profile)) or 0,
    )

    cli.fr(deck_file, select=select, core=core, check=True)

    assert checked == [(Path(output_name), profile)]


@pytest.mark.parametrize(
    "select, core, checker",
    [
        pytest.param(True, False, "check_cull", id="select"),
        pytest.param(False, True, "check_core", id="core"),
    ],
)
def test_check_exits_on_a_gate_failure(
    deck_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    select: bool,
    core: bool,
    checker: str,
) -> None:
    monkeypatch.setattr(fixrun, checker, lambda _path, _profile: 1)

    with pytest.raises(SystemExit):
        cli.fr(deck_file, select=select, core=core, check=True)
