"""One test of the assembled output, for the core mode and the cull mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anki_tools.ranking import coreflow, cullflow
from anki_tools.ranking.identity import DeckIdentity
from tests.test_core_deck import _offline_profile, _tiny_deck as core_deck
from tests.test_cull_shared import _profile as cull_profile, _tiny_deck as cull_deck


def _core_run(tmp_path: Path) -> tuple[coreflow.CoreRun, Path]:
    source = core_deck(tmp_path)
    profile = _offline_profile(tmp_path)
    run = coreflow.run_core(
        coreflow.CoreRequest(
            input_path=source,
            output_path=tmp_path / "core.apkg",
            core_words=0,
            options=profile.default_options,
        ),
        profile,
    )
    return run, run.core_csv


def _cull_run(tmp_path: Path) -> tuple[cullflow.CullRun, Path]:
    source = cull_deck(tmp_path)
    run = cullflow.run_cull(
        cullflow.CullRequest(
            input_path=source,
            output_path=tmp_path / "culled.apkg",
            max_cards=2,
            target_words=2,
            min_words=2,
            max_words=4,
        ),
        cull_profile(),
    )
    return run, run.cull_csv


@pytest.mark.parametrize("run_mode", [_core_run, _cull_run], ids=["core", "cull"])
def test_the_assembler_writes_one_report_and_one_identity(
    tmp_path: Path, run_mode, capsys
) -> None:
    run, card_csv = run_mode(tmp_path)

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    identity = DeckIdentity(run.output_path)
    printed = capsys.readouterr().out

    assert run.output_path.exists()
    assert card_csv == identity.side(card_csv.name.removeprefix(f"{identity.stem}."))
    assert run.coverage_csv == identity.side("coverage.csv")
    assert manifest["source"]["model"] == identity.model_name
    assert manifest["params"]["name"] == identity.title
    assert f"{card_csv.name.split('.')[-2]}_csv" in manifest["files"]
    assert f"wrote {run.output_path}" in printed
    assert f"wrote {card_csv}" in printed
    assert f"wrote {run.coverage_csv}" in printed
    assert f"wrote {run.manifest_path}" in printed
