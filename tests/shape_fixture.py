"""One table of sentence shape rules: the first failed guard of a sentence, or empty."""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest

from anki_tools.languages.french.profile import FRENCH
from anki_tools.languages.greek.profile import GREEK
from anki_tools.languages.mandarin.profile import MANDARIN
from anki_tools.languages.spanish import core as spanish_core

PROFILES: dict[str, Any] = {
    "spanish": spanish_core.SPANISH_CORE_PROFILE,
    "french": FRENCH,
    "greek": GREEK,
    "mandarin": MANDARIN,
}


class ShapeRule(NamedTuple):
    language: str
    sentence: str
    expected: str


SHAPE_RULES = [
    ShapeRule("spanish", "El perro come pan.", ""),
    ShapeRule("spanish", "Ella comió el pan ayer.", ""),
    ShapeRule("spanish", "El perro.", "length_window"),
    ShapeRule("spanish", "El perro tiene 2 años.", "digit_marks"),
    ShapeRule("spanish", "El perro come хлеб.", "script"),
    ShapeRule("spanish", "El perro come pan", "marks"),
    ShapeRule("spanish", "El perro come. Pan.", "marks"),
    ShapeRule("spanish", 'El perro dice "hola".', "marks"),
    ShapeRule("spanish", "Sí, no, tal vez.", "marks"),
    ShapeRule("french", "Le chien mange vite.", ""),
    ShapeRule("french", "Le chien.", "length_window"),
    ShapeRule("french", "Le chien mange 42 fois ici.", "digit_marks"),
    ShapeRule("french", "Le chat [noir] mange vite.", "digit_marks"),
    ShapeRule("french", "Le chat mange vite…", "digit_marks"),
    ShapeRule("french", "Voir le site www.exemple.fr.", "digit_marks"),
    ShapeRule("french", "Le chien mange хлеб ici.", "script"),
    ShapeRule("french", "Oui oui oui oui.", "repeat"),
    ShapeRule("french", "Le chien mange vite", "marks"),
    ShapeRule("french", "Le chien, le chat, l'oiseau.", "marks"),
    ShapeRule("french", "Le chien mange. Il aboie.", "marks"),
    ShapeRule("french", "Le chien « mange » vite.", "marks"),
    ShapeRule("french", "— Le chien mange vite.", "marks"),
    ShapeRule("greek", "Η γάτα έχει 3 μήνες.", "digit_marks"),
    ShapeRule("mandarin", "我要喝水。", ""),
    ShapeRule("mandarin", "我们今天去学校。", ""),
    ShapeRule("mandarin", "我喝水。", "length_window"),
    ShapeRule("mandarin", "我们今天要去学校。", "length_window"),
    ShapeRule("mandarin", "我想喝水2杯。", "digit_marks"),
    ShapeRule("mandarin", "我要喝水ABCD。", "script"),
    ShapeRule("mandarin", "走走走走。", "repeat"),
    ShapeRule("mandarin", "走走走走", "repeat"),
    ShapeRule("mandarin", "我要喝水。。", "marks"),
    ShapeRule("mandarin", "我要喝水", "marks"),
    ShapeRule("mandarin", "我，要，喝水。", "marks"),
    ShapeRule("mandarin", "他说“我要喝水”。", "marks"),
]


def shape_rules(*languages: str) -> list[object]:
    """Give the rows of the languages as ``parametrize`` rows with ids."""
    return [
        pytest.param(
            *rule, id=f"{rule.language}-{rule.expected or 'pass'}-{rule.sentence}"
        )
        for rule in SHAPE_RULES
        if rule.language in languages
    ]
