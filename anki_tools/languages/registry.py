"""Language registry of the ``anki swadesh`` command.

:data:`LANGUAGES` names every Swadesh language once and gives each language its
capability:

* a sentence-pack module, which the shared pipeline drives;
* a vocabulary deck, which the ``vocab`` command serves from an
  ``english-<lang>.csv`` file and an audio directory.

:data:`ALIASES` maps the short codes that a user types to the one canonical
name of the language.

Importing this module also registers the Mandarin, French, Greek, and Cantonese
profiles of :data:`anki_tools.sentence_pack.profile.PROFILES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Sequence

from anki_tools.sentence_pack.profile import (
    PROFILES,
    LanguageProfile,
    get_profile,
    profile_names,
)

from anki_tools.languages.cantonese import profile as cantonese
from anki_tools.languages.french import profile as french
from anki_tools.languages.greek import profile as greek
from anki_tools.languages.mandarin import profile as mandarin
from anki_tools.languages.mandarin import stages as mandarin_stages
from anki_tools.languages.spanish import profile as spanish


@dataclass(frozen=True)
class CliLanguage:
    """One language of the ``anki swadesh`` command.

    :param name: Language name, for example ``greek``.
    :param label: Display name of the deck commands, for example ``Greek``.
    :param profile: Sentence-pack profile, None for a vocabulary-only language.
    :param module: Sentence-pack module that the shared pipeline drives, None
        when the language has no pipeline module.
    :param audit: Gate runner of the language, None when it has none.
    :param vocab: True when the ``vocab`` command serves the language from an
        ``english-<lang>.csv`` file and an audio directory.
    :param tts: gTTS language code of the ``vocab`` command, None when it has
        none.
    """

    name: str
    label: str
    profile: LanguageProfile | None = None
    module: ModuleType | None = None
    audit: Callable[[], Sequence[Any]] | None = None
    vocab: bool = False
    tts: str | None = None


LANGUAGES: dict[str, CliLanguage] = {
    language.name: language
    for language in (
        CliLanguage(
            "mandarin",
            label="Mandarin",
            profile=mandarin.MANDARIN,
            module=mandarin,
            audit=mandarin_stages.run_audit,
            vocab=True,
            tts="zh-cn",
        ),
        CliLanguage(
            "french",
            label="French",
            profile=french.FRENCH,
            module=french,
            audit=french.run_french_audit,
        ),
        CliLanguage(
            "spanish",
            label="Spanish",
            profile=spanish.SPANISH,
            module=spanish,
            audit=spanish.run_spanish_audit,
            vocab=True,
            tts="es",
        ),
        CliLanguage(
            "greek",
            label="Greek",
            profile=greek.GREEK,
            module=greek,
            audit=greek.run_greek_audit,
            vocab=True,
            tts="el",
        ),
        CliLanguage("german", label="German", vocab=True, tts="de"),
        CliLanguage(
            "cantonese",
            label="Cantonese",
            profile=cantonese.CANTONESE,
            module=cantonese,
            audit=cantonese.run_cantonese_audit,
            vocab=True,
            tts="yue",
        ),
        CliLanguage("arabic", label="Arabic", vocab=True, tts="ar"),
        CliLanguage("hindi", label="Hindi", vocab=True, tts="hi"),
    )
}

ALIASES: dict[str, str] = {
    "yue": "cantonese",
    "cmn": "mandarin",
    "zh": "mandarin",
    "el": "greek",
}


def resolve(name: str) -> str:
    """Give the one canonical language name of a user input.

    :param name: Language name or short code, for example ``yue``.
    :returns: The canonical name, or the folded input when no alias matches.
    """
    key = name.strip().lower()
    return ALIASES.get(key, key)


def vocab_languages() -> tuple[str, ...]:
    """Give the language names of the ``vocab`` command, in display order.

    :returns: Names of the languages with a vocabulary deck.
    """
    return tuple(language.name for language in LANGUAGES.values() if language.vocab)


def profile_languages() -> tuple[CliLanguage, ...]:
    """Give the languages that have a Swadesh sentence pack.

    :returns: Registry entries with a profile, in display order.
    """
    return tuple(language for language in LANGUAGES.values() if language.profile)


def pipeline_languages() -> tuple[str, ...]:
    """Give the languages that the shared sentence pipeline drives.

    :returns: Names of the languages with a pipeline module, in display order.
    """
    return tuple(language.name for language in LANGUAGES.values() if language.module)


__all__ = [
    "ALIASES",
    "LANGUAGES",
    "PROFILES",
    "CliLanguage",
    "cantonese",
    "french",
    "get_profile",
    "greek",
    "mandarin",
    "pipeline_languages",
    "profile_languages",
    "profile_names",
    "resolve",
    "spanish",
    "vocab_languages",
]
