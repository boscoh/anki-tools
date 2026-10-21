"""Language packages of ``anki_tools``.

Each language has one package under this directory, and it holds all the code
of that language. The registry lives in :mod:`anki_tools.languages.registry`.

This module re-exports the registry. It binds the subpackages, so
``from anki_tools.languages import cantonese`` gives the Cantonese package, not
the Cantonese profile.
"""

from __future__ import annotations

from . import cantonese, french, greek, mandarin, spanish
from .registry import (
    ALIASES,
    LANGUAGES,
    PROFILES,
    CliLanguage,
    get_profile,
    pipeline_languages,
    profile_languages,
    profile_names,
    resolve,
    swadesh_languages,
)

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
    "swadesh_languages",
]
