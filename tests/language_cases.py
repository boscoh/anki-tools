"""One case table for each hook of a language profile."""

from __future__ import annotations

from typing import NamedTuple

import pytest

from anki_tools.languages.cantonese.profile import CANTONESE
from anki_tools.languages.french.profile import FRENCH
from anki_tools.languages.greek.profile import GREEK
from anki_tools.languages.mandarin.profile import MANDARIN
from anki_tools.languages.spanish.profile import SPANISH
from anki_tools.sentence_pack.profile import LanguageProfile

LANGUAGES: dict[str, LanguageProfile] = {
    "french": FRENCH,
    "spanish": SPANISH,
    "greek": GREEK,
    "cantonese": CANTONESE,
    "mandarin": MANDARIN,
}


class HighlightCase(NamedTuple):
    language: str
    sentence: str
    form: str
    expected: str
    rule: str


class MatchCase(NamedTuple):
    language: str
    sentence: str
    form: str
    matches: int
    rule: str


class HitCase(NamedTuple):
    language: str
    sentence: str
    forms: tuple[str, ...]
    found: dict[str, int]
    rule: str


class PrepareCase(NamedTuple):
    language: str
    raw: str
    expected: str
    rule: str


class UnitCase(NamedTuple):
    language: str
    sentence: str
    expected: int
    rule: str


HIGHLIGHT_CASES = [
    HighlightCase(
        "french", "Ils se battent ici.", "se battre", "Ils <b>se battent</b> ici.",
        "a pair form is bolded as one span",
    ),
    HighlightCase(
        "french", "Je ne parle pas.", "ne", "Je <b>ne</b> parle <b>pas</b>.",
        "the form ne bolds both parts of the negation",
    ),
    HighlightCase("spanish", "El perro come pan.", "perro", "El <b>perro</b> come pan.", "a surface match is bolded"),
    HighlightCase("spanish", "Los perros comen pan.", "perro", "Los <b>perros</b> comen pan.", "a lemma match is bolded as the whole token"),
    HighlightCase("spanish", "Perro come pan.", "perro", "<b>Perro</b> come pan.", "the keys are casefolded"),
    HighlightCase("spanish", "El gato come pan.", "perro", "El gato come pan.", "no match leaves the sentence as it is"),
    HighlightCase("spanish", "El perro ve al perro.", "perro", "El <b>perro</b> ve al perro.", "only the first match is bolded"),
    HighlightCase("spanish", "El niño está de pie.", "estar de pie", "El niño <b>está de pie</b>.", "a pair form is bolded as one span"),
    HighlightCase("spanish", "El niño está siempre de pie.", "estar de pie", "El niño está siempre de pie.", "a pair form needs adjacent parts"),
    HighlightCase("spanish", "Este perro come.", "este", "<b>Este</b> perro come.", "a pronoun form is bolded by its surface"),
    HighlightCase("spanish", "Me gusta el pan.", "yo", "Me gusta el pan.", "a pronoun form does not match by lemma"),
    HighlightCase("spanish", "Quiero acostar al niño.", "acostarse", "Quiero <b>acostar</b> al niño.", "a reflexive infinitive is bolded without the clitic"),
    HighlightCase("greek", "Η γάτα τρώει ψωμί.", "γάτα", "Η <b>γάτα</b> τρώει ψωμί.", "a whole token is bolded"),
    HighlightCase("greek", "Η γάτα τρώει ψωμί.", "σκύλος", "Η γάτα τρώει ψωμί.", "an absent word leaves the sentence as it is"),
    HighlightCase("cantonese", "我飲水。", "水", "我飲<b>水</b>。", "a whole token is bolded"),
    HighlightCase("cantonese", "我飲水。", "火", "我飲水。", "an absent word leaves the sentence as it is"),
    HighlightCase("mandarin", "我要喝水。", "水", "我要喝<b>水</b>。", "a whole token is bolded"),
    HighlightCase("mandarin", "我要喝水。", "火", "我要喝水。", "an absent word leaves the sentence as it is"),
]  # fmt: skip

MATCH_CASES = [
    MatchCase("french", "La pluie tombe depuis hier.", "pluie", 1, "a plain form"),
    MatchCase("french", "La voiture est en panne.", "eau", 0, "an absent form"),
    MatchCase("french", "Ils se battent dans la rue.", "se battre", 1, "a pair form"),
    MatchCase(
        "french", "Il aime battre le tambour.", "se battre", 0,
        "a pair form needs the clitic",
    ),
    MatchCase(
        "french", "Il se dit de battre le tambour.", "se battre", 0,
        "the parts of a pair form are adjacent",
    ),
    MatchCase("french", "Je ne veux pas partir.", "ne pas", 1, "both parts of ne pas"),
    MatchCase("french", "Je viens.", "parce que", 0, "no part of the form"),
    MatchCase("spanish", "El perro come pan.", "perro", 1, "a surface key matches"),
    MatchCase("spanish", "Los perros comen pan.", "perro", 1, "a lemma key matches"),
    MatchCase("spanish", "Perro come pan.", "perro", 1, "the keys are casefolded"),
    MatchCase("spanish", "El gato come pan.", "perro", 0, "an absent form"),
    MatchCase("spanish", "El perro ve al perro.", "perro", 2, "every match is counted"),
    MatchCase("spanish", "El niño está de pie.", "estar de pie", 1, "the parts of a pair form are adjacent"),
    MatchCase("spanish", "El niño está siempre de pie.", "estar de pie", 0, "a pair form needs adjacent parts"),
    MatchCase("spanish", "Este perro come.", "este", 1, "a pronoun form matches by its surface"),
    MatchCase("spanish", "Me gusta el pan.", "yo", 0, "a pronoun form does not match by lemma"),
    MatchCase("spanish", "Quiero acostar al niño.", "acostarse", 1, "a reflexive infinitive matches without the clitic"),
    MatchCase("greek", "Η γάτα τρώει ψωμί.", "γάτα", 1, "an exact word"),
    MatchCase("greek", "Η γάτα τρώει ψωμί.", "σκύλος", 0, "an absent word"),
    MatchCase("cantonese", "我飲水。", "水", 1, "an exact word"),
    MatchCase("cantonese", "我飲水。", "火", 0, "an absent word"),
    MatchCase("mandarin", "我要喝水。", "水", 1, "an exact word"),
    MatchCase("mandarin", "我要喝水。", "火", 0, "an absent word"),
]  # fmt: skip

HIT_CASES = [
    HitCase(
        "french", "Cessez de fumer si vous voulez vivre longtemps !", ("fumée",), {},
        "a noun form is not credited by its verb",
    ),
    HitCase(
        "french", "Il y a de la fumée dans la cuisine.", ("fumée",), {"fumée": 1},
        "a noun form is credited by the noun",
    ),
    HitCase(
        "french", "La pluie suit la pluie.", ("pluie",), {},
        "a form with two matches is not a hit",
    ),
    HitCase("spanish", "El perro come pan.", ("perro",), {"perro": 1}, "a form with one match is a hit"),
    HitCase("spanish", "El perro ve al perro.", ("perro",), {}, "a form with two matches is not a hit"),
    HitCase("spanish", "Los perros ven al perro.", ("perro",), {}, "a surface match and a lemma match count together"),
    HitCase("spanish", "El gato come pan.", ("perro",), {}, "an absent form is not a hit"),
    HitCase("spanish", "El perro ve al gato.", ("perro", "gato"), {"perro": 1, "gato": 1}, "each form of the list is counted alone"),
    HitCase("spanish", "El niño está de pie.", ("estar de pie",), {"estar de pie": 1}, "a pair form with one match is a hit"),
    HitCase("spanish", "Me gusta el pan.", ("yo",), {}, "a pronoun form is not a hit by lemma"),
    HitCase("greek", "Η γάτα τρώει ψωμί.", ("γάτα",), {"γάτα": 1}, "an exact word is a hit"),
    HitCase("greek", "Η γάτα τρώει ψωμί.", ("σκύλος",), {}, "an absent word is not a hit"),
    HitCase("cantonese", "我飲水。", ("水",), {"水": 1}, "an exact word is a hit"),
    HitCase("cantonese", "我飲水。", ("火",), {}, "an absent word is not a hit"),
    HitCase("mandarin", "我要喝水。", ("水",), {"水": 1}, "an exact word is a hit"),
    HitCase("mandarin", "我要喝水。", ("火",), {}, "an absent word is not a hit"),
]  # fmt: skip

PREPARE_CASES = [
    PrepareCase(
        "french", "Bonjour\u202f! Ça va\u00a0? Oui\u202f!", "Bonjour ! Ça va ? Oui !",
        "the narrow no-break space and the no-break space become a space",
    ),
    PrepareCase(
        "french", "Écoute\u00a0: le chien.", "Écoute : le chien.",
        "the no-break space before a colon becomes a space",
    ),
    PrepareCase("spanish", "Hola\u00a0mundo", "Hola mundo", "the no-break space becomes a space"),
    PrepareCase("spanish", "Hola\u202fmundo", "Hola mundo", "the narrow no-break space becomes a space"),
    PrepareCase("spanish", "Cafe\u0301 con leche", "Café con leche", "the text is NFC"),
    PrepareCase("spanish", "El  Perro   come.", "El Perro come.", "the spaces are normalized and the case stays"),
]  # fmt: skip

UNIT_CASES = [
    UnitCase("french", "Ça va\u00a0?", 2, "a no-break space is not a word"),
    UnitCase("spanish", "El perro come pan.", 4, "the unit count is the word count"),
    UnitCase("spanish", "Sí, sí, sí.", 3, "the unit count leaves out the punctuation"),
    UnitCase("spanish", "¿Qué quieres?", 2, "the inverted mark is not a word"),
    UnitCase("spanish", "¿Qué\u00a0tal?", 2, "a no-break space is not a word"),
]  # fmt: skip


def _rows(cases: list, languages: tuple[str, ...]) -> list[object]:
    return [
        pytest.param(*case[:-1], id=f"{case.language}-{case.rule}")
        for case in cases
        if not languages or case.language in languages
    ]


def highlight_rules(*languages: str) -> list[object]:
    """Give the highlight rows of the languages as ``parametrize`` rows with ids."""
    return _rows(HIGHLIGHT_CASES, languages)


def match_rules(*languages: str) -> list[object]:
    """Give the match rows of the languages as ``parametrize`` rows with ids."""
    return _rows(MATCH_CASES, languages)


def hit_rules(*languages: str) -> list[object]:
    """Give the hit rows of the languages as ``parametrize`` rows with ids."""
    return _rows(HIT_CASES, languages)


def prepare_rules(*languages: str) -> list[object]:
    """Give the prepare rows of the languages as ``parametrize`` rows with ids."""
    return _rows(PREPARE_CASES, languages)


def unit_rules(*languages: str) -> list[object]:
    """Give the unit rows of the languages as ``parametrize`` rows with ids."""
    return _rows(UNIT_CASES, languages)
