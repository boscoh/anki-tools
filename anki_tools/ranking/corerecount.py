"""Gate K24: recount the tier S forms of each card, independent of the build.

The module imports no producer code: no tokenizer, no lemma rule, no sense
test. The other core gates share those with the build, so a bug in one of them
passes the gates and the build alike. This gate reads only the written files,
the standard library, and, for a language that inflects its words, the spaCy
model that ``CoreProfile.recount_model`` names. The row layout below is the
file contract of ``<stem>.core.csv`` and of the note fields.
"""

from __future__ import annotations

import os.path
import re
from collections.abc import Callable
from typing import Any

SENTENCE_FIELD = "Sentence"
CORE_FIELD = "Core"
TARGET_SEPARATOR = "|"
FORM_TIER = "S"
MAX_SAMPLE = 5
STEM_LENGTH = 4


def _plain(text: str) -> str:
    """Strip the tags and fold the case."""
    return re.sub(r"<[^>]*>", "", text).casefold()


def _token_reader(model: str) -> Callable[[str], set[str]]:
    """Give a reader of the lemmas and the surface words of a text.

    The model loads once, and only when the language names one. Without a model
    the reader gives nothing, and only the substring test holds.
    """
    if not model:
        return lambda _text: set()
    import spacy

    nlp = spacy.load(model, exclude=["parser", "ner"])
    return lambda text: {
        word for token in nlp(text) for word in (token.lemma_.casefold() + " " + token.lower_).split()
    }


def _holds(word: str, tokens: set[str]) -> bool:
    """Each part of the form is a token, or shares a stem of 4 characters with one.

    The parts of ``ne pas`` need not touch, and ``s'étendre`` is held by
    ``étend``. A one-letter part (the ``s`` of ``s'étendre``) is ignored.
    """
    return all(
        part in tokens
        or any(len(os.path.commonprefix([part, token])) >= STEM_LENGTH for token in tokens)
        for part in re.split(r"[\s']+", word)
        if len(part) > 1
    )


def missing_forms(files: Any) -> tuple[list[str], int]:
    """Give the faults of the recount and the count of forms checked."""
    forms = {row["key"]: row["word"] for row in files.coverage() if row["tier"] == FORM_TIER}
    tokens_of = _token_reader(files.profile.recount_model)
    sentence_index = files.fields().index(SENTENCE_FIELD)
    core_index = files.fields().index(CORE_FIELD)
    notes = files.notes()
    missing: list[str] = []
    checked = 0
    for row in files.rows():
        if row["kept"] != "1":
            continue
        fields = notes.get(int(row["note_id"]))
        if fields is None:
            missing.append(f"no note for {row['sentence']}")
            continue
        text = _plain(fields[sentence_index])
        tokens = tokens_of(text)
        wanted = [forms[key] for key in row["targets"].split(TARGET_SEPARATOR) if key in forms]
        if fields[core_index] in forms.values():
            wanted.append(fields[core_index])
        for word in dict.fromkeys(wanted):
            checked += 1
            if word.casefold() not in text and not (tokens and _holds(word.casefold(), tokens)):
                missing.append(f"{word} not in {text} [{row['note_id']}]")
    return missing, checked


def gate_recount_forms(files: Any) -> tuple[bool, str]:
    """K24: each kept card text holds the form of each of its tier S targets.

    The Core word counts too when it is a tier S form. A tier F target is a
    lemma, and an inflected card text does not hold it, so this gate skips it.
    A form that the text does not hold is a fault. A form that the text holds is
    not proof: ``得到`` is a substring of ``听得到``, so a pass means no
    missing form, not a correct hit.

    A language that inflects its words (French, Spanish) has tier S headwords
    such as ``savoir`` that the text holds as ``sais``. For these the profile
    names a spaCy model, and a form also counts when it is one of the lemmas of
    the text. The model is the only tool that the gate shares with the build.
    """
    missing, checked = missing_forms(files)
    sample = "; ".join(missing[:MAX_SAMPLE])
    return not missing, f"{len(missing)} form(s) absent of {checked} checked: {sample}"
