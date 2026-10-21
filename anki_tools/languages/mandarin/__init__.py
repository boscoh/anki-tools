"""Mandarin language package.

Holds the Mandarin profile, the pinyin helpers, the sentence ranking, and the
stage callers of the shared pipeline.

The package gives the profile attributes, so ``from anki_tools.languages import
mandarin`` gives the same names as the flat language module of the old layout.
"""

from __future__ import annotations

from typing import Any

from . import pinyin, rank, stages  # noqa: F401


def __getattr__(name: str) -> Any:
    """Give an attribute of the Mandarin profile.

    :param name: Attribute name.
    :returns: The attribute of :mod:`anki_tools.languages.mandarin.profile`.
    :raises AttributeError: The profile has no such attribute.
    """
    from . import profile

    return getattr(profile, name)
