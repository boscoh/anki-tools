"""Language packages of ``anki_tools``.

Each language has one package under this directory, and it holds all the code
of that language. The registry lives in :mod:`anki_tools.languages.registry`.

This module re-exports the registry and imports no language package.
``from anki_tools.languages import cantonese`` gives the Cantonese package, not
the Cantonese profile.
"""

from __future__ import annotations

from typing import Any

from .registry import (
    ALIASES,
    LANGUAGES,
    CliLanguage,
    get_profile,
    pipeline_languages,
    profile_languages,
    profile_names,
    resolve,
    vocab_languages,
)


def __getattr__(name: str) -> Any:
    """Give :data:`PROFILES` of the registry.

    :param name: Attribute name.
    :returns: The profile dict.
    :raises AttributeError: The name is not ``PROFILES``.
    """
    if name != "PROFILES":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import registry

    return registry.PROFILES


__all__ = [
    "ALIASES",
    "LANGUAGES",
    "PROFILES",
    "CliLanguage",
    "get_profile",
    "pipeline_languages",
    "profile_languages",
    "profile_names",
    "resolve",
    "vocab_languages",
]
