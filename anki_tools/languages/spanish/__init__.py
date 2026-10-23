"""Spanish language package.

Holds the Spanish profile, the Spanish text helpers, and the Spanish ranking.
The Spanish Swadesh deck has a vocabulary layer and a sentence layer.

The package gives the profile attributes, so ``from anki_tools.languages import
spanish`` gives the same names as the flat language module of the old layout.
"""

from __future__ import annotations

from typing import Any

from . import profile as profile  # noqa: F401
from . import rank  # noqa: F401


def __getattr__(name: str) -> Any:
    """Give an attribute of the Spanish profile.

    :param name: Attribute name.
    :returns: The attribute of :mod:`anki_tools.languages.spanish.profile`.
    :raises AttributeError: The profile has no such attribute.
    """
    return getattr(profile, name)
