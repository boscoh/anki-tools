"""Greek language package.

Holds the Greek profile and the Greek text helpers.

The package gives the profile attributes, so ``from anki_tools.languages import
greek`` gives the same names as the flat language module of the old layout.
"""

from __future__ import annotations

from typing import Any

from . import profile as profile  # noqa: F401
from . import rank  # noqa: F401


def __getattr__(name: str) -> Any:
    """Give an attribute of the Greek profile.

    :param name: Attribute name.
    :returns: The attribute of :mod:`anki_tools.languages.greek.profile`.
    :raises AttributeError: The profile has no such attribute.
    """
    return getattr(profile, name)
