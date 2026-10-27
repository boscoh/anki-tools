"""Cantonese language package.

Holds the Cantonese profile, the jyutping helpers, and the frequency
sources.

The package gives the profile attributes, so ``from anki_tools.languages import
cantonese`` gives the same names as the flat language module of the old layout.
"""

from __future__ import annotations

from typing import Any

from . import frequency, jyutping  # noqa: F401


def __getattr__(name: str) -> Any:
    """Give an attribute of the Cantonese profile.

    :param name: Attribute name.
    :returns: The attribute of :mod:`anki_tools.languages.cantonese.profile`.
    :raises AttributeError: The profile has no such attribute.
    """
    from importlib import import_module

    module = import_module(f"{__name__}.profile")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
