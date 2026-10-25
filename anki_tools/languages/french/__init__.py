"""French language package.

Holds the French profile, the French text helpers, the French ranking, and the
French cull pipeline.

The package gives the profile attributes, so ``from anki_tools.languages import
french`` gives the same names as the flat language module of the old layout.
"""

from __future__ import annotations

from typing import Any

from . import cull, rank

__all__ = ["cull", "rank"]


def __getattr__(name: str) -> Any:
    """Give an attribute of the French profile.

    :param name: Attribute name.
    :returns: The attribute of :mod:`anki_tools.languages.french.profile`.
    :raises AttributeError: The profile has no such attribute.
    """
    from . import profile

    return getattr(profile, name)
