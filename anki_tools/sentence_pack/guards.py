"""Shared helpers of the language guards.

The helpers hold no language data. A language module gives the compounds and the
forms. The Mandarin pack scans the Han character stream. The Cantonese pack
scans the sentence text. Both packs use the same scan helpers.

Two guards use the scan helpers:

1. A false-friend compound blocks a form occurrence. For example, 闻 does not
   count inside 新闻.
2. A whole-token match keeps a form occurrence. The Cantonese pack uses this
   rule, because 我 and 我哋 are different concepts.

The shape predicate :func:`check_sentence_shape` is also here. It applies the
shape guards with the profile that the caller gives, so a language package
does not import another language package for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterable, Sequence

from anki_tools.sentence_pack.english_sense import sense_row
from anki_tools.sentence_pack.pool import shape_failure
from anki_tools.sentence_pack.profile import SentenceShape

Span = tuple[int, int]


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


def compound_spans(text: str, compounds: Iterable[str]) -> list[Span]:
    """Give the span of every compound in the text.

    :param text: Prepared sentence text.
    :param compounds: Compound strings to find.
    :returns: ``(start, end)`` spans, in the order of the compounds.
    """
    return [
        (position, position + len(compound))
        for compound in compounds
        for position in occurrences(text, compound)
    ]


def form_compounds(path: Path, form: str) -> tuple[str, ...]:
    """Give the compounds in which a form takes another sense.

    :param path: Sense file of the language.
    :param form: Word-table form.
    :returns: Compound strings of the ``compounds`` column, empty when none.
    """
    row = sense_row(path, "S", form)
    return row.compounds if row else ()


def inside_spans(position: int, spans: Sequence[Span]) -> bool:
    """Tell whether a form start sits inside one blocked span.

    :param position: Start offset of the form.
    :param spans: Blocked spans.
    :returns: True when a span holds the position.
    """
    return any(start <= position < end for start, end in spans)


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
