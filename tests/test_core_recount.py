"""Tests for gate K24, the substring recount of the tier S forms of a card."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools.ranking import corerecount
from tests.ast_fixture import imported_names


class _Files:
    """Reader with one kept card, one coverage table, and the card fields."""

    def __init__(self, sentence: str, targets: str, core: str, exact: bool = True) -> None:
        self.profile = SimpleNamespace(exact_forms=exact)
        self._fields = [sentence, core]
        self._targets = targets

    def coverage(self) -> list[dict[str, str]]:
        return [
            {"key": "S1:得到", "tier": "S", "word": "得到"},
            {"key": "S2:我", "tier": "S", "word": "我"},
            {"key": "F:好", "tier": "F", "word": "好"},
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


CASES = [
    ("form in text", "我<b>得到</b>了。", "S1:得到|S2:我", "得到", True, True),
    ("target form absent", "我拿到了。", "S1:得到|S2:我", "我", True, False),
    ("core word absent", "我拿到了。", "S2:我", "得到", True, False),
    ("tier F target is not a form", "我很开心。", "S2:我|F:好", "", True, True),
    ("kept row only", "我拿到了。", "S2:我", "", True, True),
    ("inflected language skips", "je sais", "S1:得到", "得到", False, True),
    ("substring is not proof", "我听得到了。", "S1:得到", "得到", True, True),
]


@pytest.mark.parametrize(("label", "sentence", "targets", "core", "exact", "passed"), CASES)
def test_recount_forms(label, sentence, targets, core, exact, passed) -> None:
    ok, _message = corerecount.gate_recount_forms(_Files(sentence, targets, core, exact))  # type: ignore[arg-type]

    assert ok is passed, label


def test_the_gate_names_the_missing_form() -> None:
    _ok, message = corerecount.gate_recount_forms(_Files("我拿到了。", "S1:得到", ""))  # type: ignore[arg-type]

    assert message.startswith("1 form(s) absent of 1 checked: 得到")


def test_the_gate_imports_no_producer_code() -> None:
    path = Path(corerecount.__file__)

    assert {name.split(".")[0] for name in imported_names(path)} <= {"__future__", "re", "typing"}
