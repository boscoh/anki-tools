"""Gate K24: recount the tier S forms of each card with a plain substring match.

The module imports no producer code: no tokenizer, no lemma rule, no sense
test. The other core gates share those with the build, so a bug in one of them
passes the gates and the build alike. This gate reads only the written files
and the standard library. The row layout below is the file contract of
``<stem>.core.csv`` and of the note fields.
"""

from __future__ import annotations

import re
from typing import Any

SENTENCE_FIELD = "Sentence"
CORE_FIELD = "Core"
TARGET_SEPARATOR = "|"
FORM_TIER = "S"
MAX_SAMPLE = 5


def _plain(text: str) -> str:
    """Strip the tags and fold the case."""
    return re.sub(r"<[^>]*>", "", text).casefold()


def gate_recount_forms(files: Any) -> tuple[bool, str]:
    """K24: each kept card text holds the form of each of its tier S targets.

    The Core word counts too when it is a tier S form. A tier F target is a
    lemma, and an inflected card text does not hold it, so this gate skips it.
    A form that the text does not hold is a fault. A form that the text holds is
    not proof: ``得到`` is a substring of ``听得到``, so a pass means no
    missing form, not a correct hit.

    A language that inflects its words (French, Spanish) has tier S headwords
    such as ``savoir`` that the text holds as ``sais``. A substring test would
    fail every such card, so the gate passes and says it skipped
    (``CoreProfile.exact_forms`` is False).
    """
    if not files.profile.exact_forms:
        return True, "skipped: the language inflects its forms"
    forms = {row["key"]: row["word"] for row in files.coverage() if row["tier"] == FORM_TIER}
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
        wanted = [forms[key] for key in row["targets"].split(TARGET_SEPARATOR) if key in forms]
        if fields[core_index] in forms.values():
            wanted.append(fields[core_index])
        for word in dict.fromkeys(wanted):
            checked += 1
            if word.casefold() not in text:
                missing.append(f"{word} not in {text}")
    sample = "; ".join(missing[:MAX_SAMPLE])
    return not missing, f"{len(missing)} form(s) absent of {checked} checked: {sample}"
