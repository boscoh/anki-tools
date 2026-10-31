"""Tests of the assembler: one run value, one writer, one stated shape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from anki_tools.apkg import deck_writer
from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import assemble
from anki_tools.ranking.identity import DeckIdentity

INPUT_MODEL = "Test Sentence"
INPUT_FIELDS = ["Text", "Meaning", "Audio"]


@dataclass(frozen=True)
class _StubProfile:
    """The seven members that the writer reads from a language profile."""

    @property
    def input_model_name(self) -> str:
        return INPUT_MODEL

    @property
    def input_sentence_field(self) -> str:
        return "Text"

    @property
    def input_english_field(self) -> str:
        return "Meaning"

    @property
    def input_audio_field(self) -> str:
        return "Audio"

    @property
    def romanization_source(self) -> str | None:
        return None

    @property
    def templates(self) -> tuple[tuple[str, str], ...]:
        return (("{{Audio}}", "{{Audio}}<hr id=answer>{{Sentence}}"),)

    @property
    def highlight(self) -> Callable[[str, str], str] | None:
        return None

    def card_css(self) -> str:
        return ".target { color: red; }"


def _input_deck(tmp_path: Path) -> Path:
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=INPUT_FIELDS,
        cards=[
            {"Text": "kat ser", "Meaning": "cat is big.", "Audio": ""},
            {"Text": "hundo", "Meaning": "dog", "Audio": ""},
        ],
        model_name=INPUT_MODEL,
    )
    return source


def test_the_writer_states_the_shape_from_the_profile(
    tmp_path: Path, capsys
) -> None:
    source = _input_deck(tmp_path)
    identity = DeckIdentity(tmp_path / "core.apkg")
    with AnkiPackage(source) as package:
        kept_id = int(next(iter(package.get_notes()))["id"])

    files = assemble.write_run(
        assemble.DeckRun(
            identity=identity,
            input_path=source,
            values={kept_id: ["", "kat ser", "cat is big.", "kat", "cat"]},
            card_table=deck_writer.SideTable(
                "core.csv", ("note_id", "kept"), [{"note_id": kept_id, "kept": "1"}]
            ),
            coverage_table=deck_writer.SideTable(
                "coverage.csv", ("key", "status"), [{"key": "kat", "status": "covered"}]
            ),
            manifest=assemble.ManifestValues(
                input_model=INPUT_MODEL,
                params={"max_cards": 1},
                counts={"cards": 1},
                frequency={"path": "test", "spoken_weight": 0.5},
                kept_order="hash",
            ),
            report=[("Cards", "1")],
        ),
        _StubProfile(),
    )

    manifest = json.loads(files.manifest_path.read_text(encoding="utf-8"))
    with AnkiPackage(files.output_path) as package:
        models = list(package.get_models().values())
        fields = [field["name"] for model in models for field in model["flds"]]
        templates = [
            (template["qfmt"], template["afmt"])
            for model in models
            for template in model["tmpls"]
        ]
    printed = capsys.readouterr().out

    assert files.sides["core.csv"] == identity.side("core.csv")
    assert files.sides["coverage.csv"] == identity.side("coverage.csv")
    assert "rank.csv" not in files.sides
    assert manifest["source"] == {
        "input": str(source),
        "deck": identity.title,
        "model": identity.model_name,
        "input_model": INPUT_MODEL,
    }
    assert manifest["params"]["name"] == identity.title
    assert manifest["files"]["core_csv"].startswith("sha256:")
    assert [model["name"] for model in models] == [identity.title]
    assert fields == ["Audio", "Sentence", "English", "Core", "CoreGloss"]
    assert templates == list(_StubProfile().templates)
    assert "Cards:" in printed
    assert str(files.output_path) in printed
