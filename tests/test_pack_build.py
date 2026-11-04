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
from anki_tools.languages.spanish import profile as spanish
from anki_tools.sentence_pack import pack
from tests.mandarin_fixture import audio_entries, pack_row, write_audio, write_csv
from tests.token_fixture import nlp  # noqa: F401

CARDS = [
    pytest.param(
        french.FRENCH,
        french.WordForm(150, "water", "eau", "f", "primary", "single", 5.0),
        "Cette eau est froide.",
        id="french",
    ),
    pytest.param(
        greek.GREEK,
        greek.WordForm(150, "water", "νερό", "n", "primary", "single", 5.0),
        "Το νερό είναι κρύο.",
        id="greek",
    ),
    pytest.param(
        spanish.SPANISH,
        spanish.WordForm(150, "water", "agua", "f", "primary", "single", 5.0),
        "Esta agua está fría.",
        id="spanish",
    ),
    pytest.param(
        cantonese.CANTONESE,
        cantonese.WordForm(150, "water", "水", "seoi2", "primary", "single", 5.0),
        "我想飲水。",
        id="cantonese",
    ),
]


@pytest.mark.usefixtures("nlp")
@pytest.mark.parametrize(("profile", "form", "sentence"), CARDS)
def test_the_pack_build_writes_one_bold_card_with_its_audio(
    tmp_path: Path, profile: Any, form: Any, sentence: str
) -> None:
    target = profile.form_text(form)
    row = pack_row(target, sentence, "", "I want water.", "200", "bob") | {
        "jyutping": "ngo5 soeng2 jam2 seoi2"
    }
    audio_dir = tmp_path / "audio"
    write_audio(audio_dir, audio_entries([row]))
    csv_path = tmp_path / "sentences.csv"
    write_csv(csv_path, [*row], [row])
    word_csv = tmp_path / "words.csv"
    word_csv.write_text("english\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    result = pack.build_pack(
        profile,
        lambda **_: [form],
        csv_path=csv_path,
        word_csv=word_csv,
        audio_dir=audio_dir,
        output=tmp_path / "pack.apkg",
        manifest_path=manifest_path,
        attribution_path=tmp_path / "ATTRIBUTION.md",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with AnkiPackage(tmp_path / "pack.apkg") as package:
        notes = package.get_notes()
        names = [field["name"] for field in next(iter(package.get_models().values()))["flds"]]
        values = dict(zip(names, str(notes[0]["flds"]).split("\x1f")))
    assert (result["media_count"], result["skipped"]) == (2, 0)
    assert manifest["counts"]["cards"] == 1
    assert names == list(profile.fields)
    assert len(notes) == 1
    assert f"<b>{target}</b>" in values["Sentence"]
    assert values["Audio"].startswith("[sound:tts_")
    assert values["WordAudio"].startswith("[sound:word_")
    assert "swadesh::water" in str(notes[0]["tags"])
    assert (tmp_path / "ATTRIBUTION.md").exists()
