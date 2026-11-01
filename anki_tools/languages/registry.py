"""Language registry of the ``anki swadesh`` command.

:data:`LANGUAGES` names every Swadesh language once and gives each language its
capability:

* a sentence-pack module, which the shared pipeline drives;
* a vocabulary deck, which the ``vocab`` command serves from an
  ``english-<lang>.csv`` file and an audio directory.

:data:`ALIASES` maps the short codes that a user types to the one canonical
name of the language.

A language package is imported on the first read of one of its lazy values
(:attr:`CliLanguage.profile`, :attr:`CliLanguage.module`,
:attr:`CliLanguage.audit`, :attr:`FixCommand.core`, :attr:`FixCommand.cull`).
Importing this module imports no language package. A wrong path fails at that
first read, not at start.

Reading :data:`PROFILES`, or calling :func:`get_profile` or
:func:`profile_names`, registers every profile first.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Sequence

from anki_tools.sentence_pack import profile as _profile_registry
from anki_tools.sentence_pack.profile import LanguageProfile

if TYPE_CHECKING:
    from anki_tools.ranking.cullflow import CullProfile
    from anki_tools.ranking.scan import CoreProfile


def _load(path: str | None) -> Any:
    """Import a module, or one attribute of it, from a dotted path.

    :param path: ``package.module`` or ``package.module:attribute``, or None.
    :returns: The module or the attribute, None when the path is None.
    :raises ImportError: The module does not exist.
    :raises AttributeError: The module has no such attribute.
    """
    if path is None:
        return None
    module, _, attribute = path.partition(":")
    loaded = import_module(module)
    return getattr(loaded, attribute) if attribute else loaded


@dataclass(frozen=True)
class FixCommand:
    """Declaration of one ``fix`` deck command.

    :param code: Short command name, for example ``zh``.
    :param apkg_path: Default input package of the command.
    :param core_path: Path of the core mode view of the language.
    :param cull_path: Path of the cull mode view, None when it has no cull.
    :param words: Default tier F word count of the core mode.
    :param description: Description block of the command help.
    :param select_help: Help line of the ``--select`` flag.
    """

    code: str
    apkg_path: Path
    core_path: str
    cull_path: str | None = None
    words: int = 0
    description: str = ""
    select_help: str = ""

    @property
    def core(self) -> CoreProfile:
        """Give the core mode view of the language."""
        return _load(self.core_path)

    @property
    def cull(self) -> CullProfile | None:
        """Give the cull mode view of the language, None when it has none."""
        return _load(self.cull_path)


@dataclass(frozen=True)
class CliLanguage:
    """One language of the ``anki swadesh`` command.

    :param name: Language name, for example ``greek``.
    :param label: Display name of the deck commands, for example ``Greek``.
    :param profile_path: Path of the sentence-pack profile, None for a
        vocabulary-only language.
    :param module_path: Path of the sentence-pack module that the shared
        pipeline drives, None when the language has no pipeline module.
    :param audit_path: Path of the gate runner, None when it has none.
    :param vocab: True when the ``vocab`` command serves the language from an
        ``english-<lang>.csv`` file and an audio directory.
    :param tts: gTTS language code of the ``vocab`` command, None when it has
        none.
    :param fix: Declared ``fix`` deck command, None when the language has none.
    """

    name: str
    label: str
    profile_path: str | None = None
    module_path: str | None = None
    audit_path: str | None = None
    vocab: bool = False
    tts: str | None = None
    fix: FixCommand | None = None

    @property
    def profile(self) -> LanguageProfile | None:
        """Give the sentence-pack profile, None for a vocabulary-only language."""
        return _load(self.profile_path)

    @property
    def module(self) -> ModuleType | None:
        """Give the pipeline module, None when the language has none."""
        return _load(self.module_path)

    @property
    def audit(self) -> Callable[[], Sequence[Any]] | None:
        """Give the gate runner of the language, None when it has none."""
        return _load(self.audit_path)


LANGUAGES: dict[str, CliLanguage] = {
    language.name: language
    for language in (
        CliLanguage(
            "mandarin",
            label="Mandarin",
            profile_path="anki_tools.languages.mandarin.profile:MANDARIN",
            module_path="anki_tools.languages.mandarin.profile",
            audit_path="anki_tools.languages.mandarin.stages:run_audit",
            vocab=True,
            tts="zh-cn",
            fix=FixCommand(
                "zh", Path("apkg/Neri's Chinese Sents Part 2.apkg"),
                core_path="anki_tools.languages.mandarin.core:MANDARIN_CORE_PROFILE",
                words=50,
                description="""Fix Chinese deck: build the core deck.

``--core`` builds ``apkg/mandarin_core.apkg`` with its CSV files and manifest.""",
            ),
        ),
        CliLanguage(
            "french",
            label="French",
            profile_path="anki_tools.languages.french.profile:FRENCH",
            module_path="anki_tools.languages.french.profile",
            audit_path="anki_tools.languages.french.profile:run_french_audit",
            fix=FixCommand(
                "fr", Path("apkg/6000_French_Sentences_w_Audio.apkg"),
                core_path="anki_tools.languages.french.core:FRENCH_CORE_PROFILE",
                cull_path="anki_tools.languages.french.cull:FRENCH_CULL_PROFILE",
                words=50,
                description="""Fix French deck: select a smaller deck, or build a core deck.

``--select`` runs the 1000-card cull and writes ``apkg/french_1000.apkg``.
``--core`` builds ``apkg/french_core.apkg`` with its CSV files and manifest.""",
                select_help="Run the card cull.",
            ),
        ),
        CliLanguage(
            "spanish",
            label="Spanish",
            profile_path="anki_tools.languages.spanish.profile:SPANISH",
            module_path="anki_tools.languages.spanish.profile",
            audit_path="anki_tools.languages.spanish.profile:run_spanish_audit",
            vocab=True,
            tts="es",
            fix=FixCommand(
                "es", Path("apkg/Spanish 9000 sentences with native audio.apkg"),
                core_path="anki_tools.languages.spanish.core:SPANISH_CORE_PROFILE",
                cull_path="anki_tools.languages.spanish.cull:SPANISH_CULL_PROFILE",
                words=50,
                description="""Fix Spanish deck: select a smaller deck, or build a core deck.

``--select`` runs the cull and writes ``apkg/spanish_1000.apkg``. ``--core``
builds ``apkg/spanish_core.apkg`` with its CSV files and manifest.""",
                select_help="Run the cull.",
            ),
        ),
        CliLanguage(
            "greek",
            label="Greek",
            profile_path="anki_tools.languages.greek.profile:GREEK",
            module_path="anki_tools.languages.greek.profile",
            audit_path="anki_tools.languages.greek.profile:run_greek_audit",
            vocab=True,
            tts="el",
        ),
        CliLanguage("german", label="German", vocab=True, tts="de"),
        CliLanguage(
            "cantonese",
            label="Cantonese",
            profile_path="anki_tools.languages.cantonese.profile:CANTONESE",
            module_path="anki_tools.languages.cantonese.profile",
            audit_path="anki_tools.languages.cantonese.profile:run_cantonese_audit",
            vocab=True,
            tts="yue",
            fix=FixCommand(
                "yue", Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg"),
                core_path="anki_tools.languages.cantonese.core:CANTONESE_CORE_PROFILE",
                description="""Fix Cantonese deck: build the core deck.

``--core`` builds ``apkg/cantonese_core.apkg`` with its CSV files and manifest.""",
            ),
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
    return tuple(
        language for language in LANGUAGES.values() if language.profile_path
    )


def pipeline_languages() -> tuple[str, ...]:
    """Give the languages that the shared sentence pipeline drives.

    :returns: Names of the languages with a pipeline module, in display order.
    """
    return tuple(
        language.name for language in LANGUAGES.values() if language.module_path
    )


def _register_profiles() -> None:
    """Import every profile module, so each one registers its profile."""
    for language in profile_languages():
        language.profile


def get_profile(name: str) -> LanguageProfile:
    """Look up a registered profile.

    :param name: Language name, for example ``greek``.
    :raises KeyError: The name has no registered profile.
    :returns: The language profile.
    """
    _register_profiles()
    return _profile_registry.get_profile(name)


def profile_names() -> list[str]:
    """List the registered profile names.

    :returns: Sorted names.
    """
    _register_profiles()
    return _profile_registry.profile_names()


def __getattr__(name: str) -> Any:
    """Give :data:`PROFILES` after every profile is registered.

    :param name: Attribute name.
    :returns: The profile dict.
    :raises AttributeError: The name is not ``PROFILES``.
    """
    if name != "PROFILES":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _register_profiles()
    return _profile_registry.PROFILES


__all__ = [
    "ALIASES",
    "LANGUAGES",
    "PROFILES",
    "CliLanguage",
    "FixCommand",
    "get_profile",
    "pipeline_languages",
    "profile_languages",
    "profile_names",
    "resolve",
    "vocab_languages",
]
