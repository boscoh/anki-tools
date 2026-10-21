"""Shared scan helpers of the language guards.

The helpers hold no language data. A language module gives the compounds and the
forms. The Mandarin pack scans the Han character stream. The Cantonese pack
scans the sentence text. Both packs use the same scan helpers.

Two guards use the helpers:

1. A false-friend compound blocks a form occurrence. For example, 闻 does not
   count inside 新闻.
2. A whole-token match keeps a form occurrence. The Cantonese pack uses this
   rule, because 我 and 我哋 are different concepts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

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


def inside_spans(position: int, spans: Sequence[Span]) -> bool:
    """Tell whether a form start sits inside one blocked span.

    :param position: Start offset of the form.
    :param spans: Blocked spans.
    :returns: True when a span holds the position.
    """
    return any(start <= position < end for start, end in spans)
