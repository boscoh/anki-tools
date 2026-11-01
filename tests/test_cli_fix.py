"""Tests for the fix commands: the select, core, and check routes.

``fixrun.run_fix`` takes its flows and gates as keyword arguments (ADR 0050).
The route tests pass recorders and read the calls. Nothing patches repo code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anki_tools import cli
from anki_tools.languages.registry import LANGUAGES
from anki_tools.ranking import cullflow, fixrun

OUT = Path("out.apkg")
SELECT = {"select": True}
CORE = {"core": True}

ROUTES = [
    pytest.param("french", SELECT, "run_cull", "apkg/french_1000.apkg", id="french-select"),
    pytest.param("spanish", SELECT, "run_cull", "apkg/spanish_1000.apkg", id="spanish-select"),
    pytest.param("french", CORE, "run_core", "apkg/french_core.apkg", id="french-core"),
    pytest.param("spanish", CORE, "run_core", "apkg/spanish_core.apkg", id="spanish-core"),
    pytest.param("mandarin", CORE, "run_core", "apkg/mandarin_core.apkg", id="mandarin-core"),
    pytest.param("cantonese", CORE, "run_core", "apkg/cantonese_core.apkg", id="cantonese-core"),
    pytest.param("french", {**SELECT, "output": OUT}, "run_cull", "out.apkg", id="select-output"),
    pytest.param("french", {**CORE, "output": OUT}, "run_core", "out.apkg", id="core-output"),
    pytest.param(
        "french", {**SELECT, "check": True}, "check_cull", "apkg/french_1000.apkg",
        id="check-select",
    ),
    pytest.param(
        "mandarin", {**CORE, "check": True, "output": OUT}, "check_core", "out.apkg",
        id="check-core-output",
    ),
]

REQUEST_FIELDS = [
    pytest.param(
        SELECT,
        {"no_reserve": False, "max_cards": cullflow.DEFAULT_MAX_CARDS},
        id="select-defaults",
    ),
    pytest.param(
        {**SELECT, "reserve": False, "cards": 500},
        {"no_reserve": True, "max_cards": 500},
        id="select-options",
    ),
    pytest.param(CORE, {"max_cards": None, "core_words": 0}, id="core-defaults"),
    pytest.param(
        {**CORE, "cards": 42, "words": 7}, {"max_cards": 42, "core_words": 7}, id="core-options"
    ),
]

ERRORS = [
    pytest.param("french", {}, "choose a mode: --select --core", id="no-mode"),
    pytest.param(
        "french", {**CORE, **SELECT}, "choose one mode: --select --core", id="two-modes"
    ),
    pytest.param(
        "french", {"check": True}, "--check needs --select or --core", id="check-no-mode"
    ),
    pytest.param(
        "mandarin", {"check": True}, "--check needs --core", id="check-no-mode-single"
    ),
    pytest.param(
        "mandarin", SELECT, "Mandarin has no --select mode", id="select-unsupported"
    ),
]


class Flows:
    """Record the calls of the four seam functions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []
        self.gate_code = 0
        self.seam: dict[str, Any] = {
            name: self._recorder(name)
            for name in ("run_core", "run_cull", "check_core", "check_cull")
        }

    def _recorder(self, name: str) -> Callable[[Any, Any], int]:
        def record(target: Any, profile: Any) -> int:
            self.calls.append((name, target, profile))
            return self.gate_code

        return record


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


def test_an_absent_input_prints_an_error(tmp_path: Path, capsys) -> None:
    command("fr")(tmp_path / "absent.apkg")

    assert "Error: File not found" in capsys.readouterr().out


@pytest.mark.parametrize("language, options, call, output", ROUTES)
def test_a_route_runs_one_flow_or_gate_on_its_output(
    flows: Flows, deck_file: Path, language: str, options: dict[str, Any], call: str, output: str
) -> None:
    fix(flows, language, deck_file, **options)

    [(name, target, profile)] = flows.calls
    kind = call.split("_")[1]
    declared = LANGUAGES[language].fix
    assert declared is not None
    assert name == call
    assert Path(getattr(target, "output_path", target)) == Path(output)
    assert profile is getattr(declared, kind)


@pytest.mark.parametrize("options", [SELECT, CORE])
def test_a_failed_gate_exits_with_its_code(
    flows: Flows, deck_file: Path, options: dict[str, bool]
) -> None:
    flows.gate_code = 3

    with pytest.raises(SystemExit) as exit_info:
        fix(flows, "french", deck_file, check=True, **options)

    assert exit_info.value.code == 3


@pytest.mark.parametrize("options, fields", REQUEST_FIELDS)
def test_the_options_reach_the_request(
    flows: Flows, deck_file: Path, options: dict[str, Any], fields: dict[str, Any]
) -> None:
    fix(flows, "french", deck_file, **options)

    [(_, request, _)] = flows.calls
    assert {key: getattr(request, key) for key in fields} == fields


@pytest.mark.parametrize("language, options, message", ERRORS)
def test_a_bad_mode_is_an_error_and_runs_nothing(
    flows: Flows, deck_file: Path, language: str, options: dict[str, bool], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        fix(flows, language, deck_file, **options)

    assert flows.calls == []
