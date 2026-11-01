"""Tests for the fix commands: the select, core, and check routes.

``fixrun.run_fix`` takes its flows and gates as keyword arguments (ADR 0049).
The route tests pass recorders and read the calls. Nothing patches repo code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from anki_tools import cli
from anki_tools.languages.cantonese import core as cantonese_core
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.languages.mandarin import core as mandarin_core
from anki_tools.languages.registry import LANGUAGES
from anki_tools.languages.spanish import core as spanish_core
from anki_tools.languages.spanish import cull as spanish_cull
from anki_tools.ranking import cullflow, fixrun


class Flows:
    """Record the calls of the four seam functions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []
        self.gate_code = 0

    @property
    def seam(self) -> dict[str, Any]:
        return {
            "run_core": lambda request, profile: self._record("run_core", request, profile),
            "run_cull": lambda request, profile: self._record("run_cull", request, profile),
            "check_core": lambda path, profile: self._gate("check_core", path, profile),
            "check_cull": lambda path, profile: self._gate("check_cull", path, profile),
        }

    def _record(self, name: str, request: Any, profile: Any) -> None:
        self.calls.append((name, request, profile))

    def _gate(self, name: str, path: Path, profile: Any) -> int:
        self.calls.append((name, path, profile))
        return self.gate_code


class Route(NamedTuple):
    language: str
    mode: str
    flow: str
    profile: Any
    default_output: Path

    @property
    def mode_option(self) -> dict[str, bool]:
        return {self.mode: True}


ROUTES = [
    pytest.param(
        Route("french", "select", "cull", french_cull.FRENCH_CULL_PROFILE,
              Path("apkg/french_1000.apkg")),
        id="french-select",
    ),
    pytest.param(
        Route("spanish", "select", "cull", spanish_cull.SPANISH_CULL_PROFILE,
              Path("apkg/spanish_1000.apkg")),
        id="spanish-select",
    ),
    pytest.param(
        Route("french", "core", "core", french_core.FRENCH_CORE_PROFILE,
              Path("apkg/french_core.apkg")),
        id="french-core",
    ),
    pytest.param(
        Route("spanish", "core", "core", spanish_core.SPANISH_CORE_PROFILE,
              Path("apkg/spanish_core.apkg")),
        id="spanish-core",
    ),
    pytest.param(
        Route("mandarin", "core", "core", mandarin_core.MANDARIN_CORE_PROFILE,
              Path("apkg/mandarin_core.apkg")),
        id="mandarin-core",
    ),
    pytest.param(
        Route("cantonese", "core", "core", cantonese_core.CANTONESE_CORE_PROFILE,
              Path("apkg/cantonese_core.apkg")),
        id="cantonese-core",
    ),
]

ERRORS = [
    pytest.param("french", {}, "choose a mode: --select --core", id="french-bare"),
    pytest.param("mandarin", {}, "choose a mode: --core", id="mandarin-bare"),
    pytest.param(
        "french", {"core": True, "select": True},
        "choose one mode: --select --core", id="two-modes",
    ),
    pytest.param(
        "french", {"check": True}, "--check needs --select or --core", id="check-french",
    ),
    pytest.param(
        "cantonese", {"check": True}, "--check needs --core", id="check-cantonese",
    ),
    pytest.param(
        "mandarin", {"select": True}, "Mandarin has no --select mode", id="no-select-zh",
    ),
    pytest.param(
        "cantonese", {"select": True, "check": True},
        "Cantonese has no --select mode", id="no-select-yue-check",
    ),
]


@pytest.fixture
def flows() -> Flows:
    return Flows()


@pytest.fixture
def deck_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.apkg"
    path.write_bytes(b"")
    return path


def fix(flows: Flows, language: str, apkg_path: Path, **options: Any) -> None:
    declaration = LANGUAGES[language].fix
    assert declaration is not None
    route = fixrun.FixLanguage(
        name=language,
        label=LANGUAGES[language].label,
        core=declaration.core,
        cull=declaration.cull,
    )
    fixrun.run_fix(route, apkg_path=apkg_path, **options, **flows.seam)


def command(name: str) -> Callable[..., None]:
    function = cli.fix_app.resolved_commands()[name].default_command
    assert function is not None
    return function


@pytest.mark.parametrize(
    "short, canonical",
    [("zh", "mandarin"), ("fr", "french"), ("yue", "cantonese"), ("es", "spanish")],
)
def test_the_canonical_names_alias_the_short_codes(short: str, canonical: str) -> None:
    assert command(short) is command(canonical)


def test_an_absent_input_prints_an_error(tmp_path: Path, capsys) -> None:
    command("fr")(tmp_path / "absent.apkg")

    assert "Error: File not found" in capsys.readouterr().out


def test_a_command_without_a_cull_rejects_check_without_core(deck_file: Path) -> None:
    with pytest.raises(SystemExit, match="--check needs --core"):
        command("zh")(deck_file, check=True)


@pytest.mark.parametrize("route", ROUTES)
def test_a_mode_runs_its_flow_on_the_default_output(
    flows: Flows, deck_file: Path, route: Route
) -> None:
    fix(flows, route.language, deck_file, **route.mode_option)

    [(name, request, profile)] = flows.calls
    assert name == f"run_{route.flow}"
    assert request.input_path == deck_file
    assert request.output_path == route.default_output
    assert profile is route.profile


@pytest.mark.parametrize("route", ROUTES)
def test_a_mode_runs_its_flow_on_a_given_output(
    flows: Flows, deck_file: Path, route: Route
) -> None:
    output = deck_file.parent / "out.apkg"

    fix(flows, route.language, deck_file, output=output, **route.mode_option)

    [(_, request, _)] = flows.calls
    assert request.output_path == output


@pytest.mark.parametrize("route", ROUTES)
def test_check_runs_the_gates_and_not_the_flow(
    flows: Flows, deck_file: Path, route: Route
) -> None:
    fix(flows, route.language, deck_file, check=True, **route.mode_option)

    assert flows.calls == [
        (f"check_{route.flow}", route.default_output, route.profile)
    ]


@pytest.mark.parametrize("route", ROUTES)
def test_check_gates_the_given_output(
    flows: Flows, deck_file: Path, route: Route
) -> None:
    output = deck_file.parent / "out.apkg"

    fix(flows, route.language, deck_file, output=output, check=True, **route.mode_option)

    assert flows.calls == [(f"check_{route.flow}", output, route.profile)]


@pytest.mark.parametrize("route", ROUTES)
def test_check_exits_with_the_gate_code(
    flows: Flows, deck_file: Path, route: Route
) -> None:
    flows.gate_code = 3

    with pytest.raises(SystemExit) as exit_info:
        fix(flows, route.language, deck_file, check=True, **route.mode_option)

    assert exit_info.value.code == 3


def test_select_options_reach_the_cull_request(flows: Flows, deck_file: Path) -> None:
    fix(flows, "french", deck_file, select=True, reserve=False, cards=500)
    fix(flows, "french", deck_file, select=True)

    (_, capped, _), (_, default, _) = flows.calls
    assert (capped.no_reserve, capped.max_cards) == (True, 500)
    assert (default.no_reserve, default.max_cards) == (False, cullflow.DEFAULT_MAX_CARDS)


def test_core_options_reach_the_core_request(flows: Flows, deck_file: Path) -> None:
    fix(flows, "french", deck_file, core=True, cards=42, words=7)
    fix(flows, "french", deck_file, core=True)

    (_, given, _), (_, default, _) = flows.calls
    assert (given.max_cards, given.core_words) == (42, 7)
    assert (default.max_cards, default.core_words) == (None, 0)


@pytest.mark.parametrize("language, options, message", ERRORS)
def test_a_bad_mode_is_an_error_and_runs_nothing(
    flows: Flows, deck_file: Path, language: str, options: dict[str, bool], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        fix(flows, language, deck_file, **options)

    assert flows.calls == []


def test_an_absent_input_runs_nothing(flows: Flows, tmp_path: Path, capsys) -> None:
    fix(flows, "french", tmp_path / "absent.apkg", core=True)

    assert flows.calls == []
    assert "Error: File not found" in capsys.readouterr().out
