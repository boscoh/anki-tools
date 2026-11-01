"""Shared helpers of the language guards.

The helpers hold no language data. A language module gives the forms. The
Mandarin pack scans the Han character stream. The Cantonese pack scans the
sentence text. Both packs use the same scan helper. The Cantonese pack keeps a
form occurrence by a whole-token match, because 我 and 我哋 are different
concepts.

The shape predicate :func:`check_sentence_shape` is also here. It applies the
shape guards with the profile that the caller gives, so a language package
does not import another language package for it.
"""

from __future__ import annotations

from typing import Collection

from anki_tools.sentence_pack.pool import shape_failure
from anki_tools.sentence_pack.profile import SentenceShape


def occurrences(text: str, form: str) -> list[int]:
    """Give every start offset of a form in the text.

    :param text: Prepared sentence text.
    :param form: Form to find.
    :returns: Start offsets in the text, lowest first.
    """
    if not form:
        return []
    found: list[int] = []
    position = text.find(form)
    while position != -1:
        found.append(position)
        position = text.find(form, position + 1)
    return found


def check_sentence_shape(
    sentence: str,
    profile: SentenceShape,
    min_units: int,
    max_units: int,
    *,
    final_mark: bool = True,
    keys: Collection[str] = (),
) -> str | None:
    """Apply the shape guards to one sentence.

    :param sentence: Prepared sentence text.
    :param profile: Shape values of the language.
    :param min_units: Minimum unit count of the window.
    :param max_units: Maximum unit count of the window.
    :param final_mark: Apply the final-mark checks of the ``marks`` guard.
    :param keys: Guard keys to apply, empty for every key.
    :returns: The first failed guard key, or None when all guards pass.
    """
    return shape_failure(
        sentence, profile, min_units, max_units, final_mark=final_mark, keys=keys
    )
