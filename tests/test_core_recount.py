"""Tests for gate K24, the substring recount of the tier S forms of a card."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import spacy

from anki_tools.gate_files import GateFileReader
from anki_tools.ranking import corerecount
from tests.ast_fixture import imported_names
from tests.cull_fixture import gate_reader

LEMMAS = {"sais": "savoir", "donne": "donner", "senté": "sentar"}


@pytest.fixture(autouse=True)
def _fake_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the spaCy model with a word splitter that knows ``LEMMAS``."""

    def nlp(text: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(lemma_=LEMMAS.get(word, word), lower_=word)
            for word in re.findall(r"\w+", text.casefold())
        ]

    monkeypatch.setattr(spacy, "load", lambda _name, **_kwargs: nlp)


def _files(sentence: str, targets: str, core: str, model: str = "") -> GateFileReader:
    """Give a reader with one kept card, one coverage table, and the card fields."""
    return gate_reader(
        SimpleNamespace(recount_model=model),
        coverage=[
            {"key": "S1:得到", "tier": "S", "word": "得到"},
            {"key": "S2:我", "tier": "S", "word": "我"},
            {"key": "F:好", "tier": "F", "word": "好"},
            {"key": "S3:savoir", "tier": "S", "word": "savoir"},
            {"key": "S4:ne pas", "tier": "S", "word": "ne pas"},
            {"key": "S5:fumée", "tier": "S", "word": "fumée"},
            {"key": "S6:sentarse", "tier": "S", "word": "sentarse"},
        ],
        rows=[
            {"kept": "1", "note_id": "1", "sentence": sentence, "targets": targets},
            {"kept": "0", "note_id": "2", "sentence": "没有", "targets": "S1:得到"},
        ],
        fields=["Sentence", "Core"],
        notes={1: [sentence, core]},
    )


MODEL = "fake_model"

CASES = [
    ("form in text", "我<b>得到</b>了。", "S1:得到|S2:我", "得到", "", True),
    ("target form absent", "我拿到了。", "S1:得到|S2:我", "我", "", False),
    ("core word absent", "我拿到了。", "S2:我", "得到", "", False),
    ("tier F target is not a form", "我很开心。", "S2:我|F:好", "", "", True),
    ("kept row only", "我拿到了。", "S2:我", "", "", True),
    ("no model means a plain test", "je sais", "S3:savoir", "", "", False),
    ("substring is not proof", "我听得到了。", "S1:得到", "得到", "", True),
    ("inflected form", "Je sais la réponse.", "S3:savoir", "", MODEL, True),
    ("target absent", "Je mange la pomme.", "S3:savoir", "", MODEL, False),
    ("split negation", "Il ne donne pas de lait.", "S4:ne pas", "", MODEL, True),
    ("reflexive stem", "Me senté entre Tom y John.", "S6:sentarse", "", MODEL, True),
]


@pytest.mark.parametrize(("label", "sentence", "targets", "core", "model", "passed"), CASES)
def test_recount_forms(label, sentence, targets, core, model, passed) -> None:
    ok, _message = corerecount.gate_recount_forms(_files(sentence, targets, core, model))

    assert ok is passed, label


def test_the_gate_names_the_missing_form() -> None:
    _ok, message = corerecount.gate_recount_forms(_files("我拿到了。", "S1:得到", ""))

    assert message.startswith("1 form(s) absent of 1 checked: 得到")


def test_the_gate_imports_no_producer_code() -> None:
    path = Path(corerecount.__file__)

    assert {name.split(".")[0] for name in imported_names(path)} <= {
        "__future__",
        "collections",
        "os",
        "re",
        "spacy",
        "typing",
    }
