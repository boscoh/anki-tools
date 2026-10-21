"""Language registry of the ``anki swadesh`` command.

:data:`LANGUAGES` names every Swadesh language once and gives each language its
capability:

* a sentence-pack module, which the shared pipeline drives;
* a profile only, for a language with its own caller code;
* neither, for a vocabulary-only language, which the age-old ``build`` command
  serves from an ``english-<lang>.csv`` file and an audio directory.

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


@dataclass(frozen=True)
class CliLanguage:
    """One language of the ``anki swadesh`` command.

    :param name: Language name, for example ``greek``.
    :param profile: Sentence-pack profile, None for a vocabulary-only language.
    :param module: Sentence-pack module that the shared pipeline drives, None
        when the language has no pipeline module.
    :param audit: Gate runner of the language, None when it has none.
    :param build: True when the ``build`` command serves the language from an
        ``english-<lang>.csv`` file and an audio directory.
    """

    name: str
    profile: LanguageProfile | None = None
    module: ModuleType | None = None
    audit: Callable[[], Sequence[Any]] | None = None
    build: bool = False


LANGUAGES: dict[str, CliLanguage] = {
    language.name: language
    for language in (
        CliLanguage("mandarin", profile=mandarin.MANDARIN, build=True),
        CliLanguage(
            "french",
            profile=french.FRENCH,
            module=french,
            audit=french.run_french_audit,
        ),
        CliLanguage("spanish", build=True),
        CliLanguage(
            "greek",
            profile=greek.GREEK,
            module=greek,
            audit=greek.run_greek_audit,
            build=True,
        ),
        CliLanguage("german", build=True),
        CliLanguage(
            "cantonese",
            profile=cantonese.CANTONESE,
            module=cantonese,
            audit=cantonese.run_cantonese_audit,
            build=True,
        ),
        CliLanguage("arabic", build=True),
        CliLanguage("hindi", build=True),
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


def swadesh_languages() -> tuple[str, ...]:
    """Give the language names of the ``build`` command, in display order.

    :returns: Every Swadesh language name.
    """
    return tuple(language.name for language in LANGUAGES.values() if language.build)


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
    "swadesh_languages",
]
