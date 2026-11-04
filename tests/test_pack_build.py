"""Tests for the shared Swadesh pack build on a one-card pool of each language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.cantonese import profile as cantonese
from anki_tools.languages.french import profile as french
from anki_tools.languages.greek import profile as greek
from anki_tools.languages.mandarin import profile as mandarin
from anki_tools.languages.mandarin.stages import GATES as MANDARIN_GATES
from anki_tools.languages.spanish import profile as spanish
from anki_tools.sentence_pack import pack
from anki_tools.sentence_pack.audit import run_pack_audit
from tests.mandarin_fixture import pack_row
from tests.pack_fixture import write_inputs
from tests.token_fixture import nlp  # noqa: F401

CARDS = [
    pytest.param(
        french.FRENCH,
        french.french_gates,
        french.WordForm(150, "water", "eau", "f", "primary", "single", 5.0),
        "Cette eau est froide.",
        id="french",
    ),
    pytest.param(
        greek.GREEK,
        greek.greek_gates,
        greek.WordForm(150, "water", "νερό", "n", "primary", "single", 5.0),
        "Το νερό είναι κρύο.",
        id="greek",
    ),
    pytest.param(
        spanish.SPANISH,
        spanish.spanish_gates,
        spanish.WordForm(150, "water", "agua", "f", "primary", "single", 5.0),
        "Esta agua está fría.",
        id="spanish",
    ),
    pytest.param(
        cantonese.CANTONESE,
        cantonese.cantonese_gates,
        cantonese.WordForm(150, "water", "水", "seoi2", "primary", "single", 5.0),
        "我想飲水。",
        id="cantonese",
    ),
    pytest.param(
        mandarin.MANDARIN,
        lambda: MANDARIN_GATES,
        mandarin.WordForm(150, "water", "水", "shuǐ", "水", "primary", 5.0),
        "我要喝水。",
        id="mandarin",
    ),
]


@pytest.mark.usefixtures("nlp")
@pytest.mark.parametrize(("profile", "gates", "form", "sentence"), CARDS)
def test_the_pack_build_writes_one_bold_card_that_passes_the_audit(
    tmp_path: Path, profile: Any, gates: Any, form: Any, sentence: str
) -> None:
    target = profile.form_text(form)
    row = pack_row(target, sentence, "", "I want water.", "200", "bob") | {
        "jyutping": "ngo5 soeng2 jam2 seoi2",
        "pinyin": "wǒ yào hē shuǐ",
    }
    env = write_inputs(
        tmp_path, [row], sentence_fields=[*row], word_fields=["english"], word_rows=[]
    )
    manifest_path = tmp_path / "manifest.json"

    result = pack.build_pack(
        profile,
        lambda **_: [form],
        csv_path=env["sentences_csv"],
        word_csv=env["word_csv"],
        audio_dir=env["audio_dir"],
        output=tmp_path / "pack.apkg",
        manifest_path=manifest_path,
        attribution_path=tmp_path / "ATTRIBUTION.md",
    )
    audit = run_pack_audit(
        profile,
        gates(),
        apkg_path=tmp_path / "pack.apkg",
        sentences_csv=env["sentences_csv"],
        word_csv=env["word_csv"],
        review_csv=env["review_csv"],
        audio_dir=env["audio_dir"],
        manifest_path=manifest_path,
        attribution_path=tmp_path / "ATTRIBUTION.md",
        load_table=lambda **_: [form],
        probe=lambda _path: 1.5,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with AnkiPackage(tmp_path / "pack.apkg") as package:
        notes = package.get_notes()
        names = [field["name"] for field in next(iter(package.get_models().values()))["flds"]]
        values = dict(zip(names, str(notes[0]["flds"]).split("\x1f")))
    assert [gate for gate, passed, _ in audit if not passed] == ["P1"]
    assert (result["media_count"], result["skipped"]) == (2, 0)
    assert manifest["counts"]["cards"] == 1
    assert names == list(profile.fields)
    assert len(notes) == 1
    assert f"<b>{target}</b>" in values["Sentence"]
    assert values["Audio"].startswith("[sound:tts_")
    assert values["WordAudio"].startswith("[sound:word_")
    assert "swadesh::water" in str(notes[0]["tags"])
    assert (tmp_path / "ATTRIBUTION.md").exists()
