"""Tests for gate K24, the substring recount of the tier S forms of a card."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools.ranking import corerecount
from tests.ast_fixture import imported_names


class _Files:
    """Reader with one kept card, one coverage table, and the card fields."""

    def __init__(self, sentence: str, targets: str, core: str, model: str = "") -> None:
        self.profile = SimpleNamespace(recount_model=model)
        self._fields = [sentence, core]
        self._targets = targets

    def coverage(self) -> list[dict[str, str]]:
        return [
            {"key": "S1:得到", "tier": "S", "word": "得到"},
            {"key": "S2:我", "tier": "S", "word": "我"},
            {"key": "F:好", "tier": "F", "word": "好"},
            {"key": "S3:savoir", "tier": "S", "word": "savoir"},
            {"key": "S4:ne pas", "tier": "S", "word": "ne pas"},
            {"key": "S5:fumée", "tier": "S", "word": "fumée"},
            {"key": "S6:sentarse", "tier": "S", "word": "sentarse"},
        ]

    def rows(self) -> list[dict[str, str]]:
        return [
            {"kept": "1", "note_id": "1", "sentence": self._fields[0], "targets": self._targets},
            {"kept": "0", "note_id": "2", "sentence": "没有", "targets": "S1:得到"},
        ]

    def fields(self) -> list[str]:
        return ["Sentence", "Core"]

    def notes(self) -> dict[int, list[str]]:
        return {1: self._fields}


FRENCH = "fr_core_news_sm"
SPANISH = "es_core_news_sm"

CASES = [
    ("form in text", "我<b>得到</b>了。", "S1:得到|S2:我", "得到", "", True),
    ("target form absent", "我拿到了。", "S1:得到|S2:我", "我", "", False),
    ("core word absent", "我拿到了。", "S2:我", "得到", "", False),
    ("tier F target is not a form", "我很开心。", "S2:我|F:好", "", "", True),
    ("kept row only", "我拿到了。", "S2:我", "", "", True),
    ("no model means a plain test", "je sais", "S3:savoir", "", "", False),
    ("substring is not proof", "我听得到了。", "S1:得到", "得到", "", True),
    ("french inflected form", "Je sais la réponse.", "S3:savoir", "", FRENCH, True),
    ("french target absent", "Je mange la pomme.", "S3:savoir", "", FRENCH, False),
    ("french split negation", "Il ne donne pas de lait.", "S4:ne pas", "", FRENCH, True),
    ("spanish reflexive form", "Me senté entre Tom y John.", "S6:sentarse", "", SPANISH, True),
]


@pytest.mark.parametrize(("label", "sentence", "targets", "core", "model", "passed"), CASES)
def test_recount_forms(label, sentence, targets, core, model, passed) -> None:
    ok, _message = corerecount.gate_recount_forms(_Files(sentence, targets, core, model))  # type: ignore[arg-type]

    assert ok is passed, label


def test_the_gate_names_the_missing_form() -> None:
    _ok, message = corerecount.gate_recount_forms(_Files("我拿到了。", "S1:得到", ""))  # type: ignore[arg-type]

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
