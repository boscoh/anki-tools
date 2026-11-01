"""Tests for the deck field resolution of ``extract_sentences``.

Each field of the deck holds its own name, so the text of the sentence names the
field that the extraction chose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.rank import Sentence, extract_sentences


def _extract(
    tmp_path: Path,
    fields: list[str],
    options: dict[str, Any] | None = None,
    values: dict[str, str] | None = None,
) -> Sentence:
    path = tmp_path / "deck.apkg"
    card = {name: name for name in fields} | (values or {})
    AnkiPackage.create(path, "Deck", fields=fields, cards=[card])
    (sentence,) = extract_sentences(str(path), **(options or {}))
    return sentence


@pytest.mark.parametrize(
    "fields, options, text, romanization",
    [
        pytest.param(["Front", "Back"], {"text_field": "Back"}, "Back", "", id="text-field-wins"),
        pytest.param(
            ["Front", "French"],
            {"text_candidates": ("French",)},
            "French",
            "",
            id="candidate-beats-fallback",
        ),
        pytest.param(["Front", "Sentence"], {}, "Sentence", "", id="sentence-fallback"),
        pytest.param(["Front", "Back"], {}, "Front", "", id="front-fallback"),
        pytest.param(
            ["Front", "French"], {}, "Front", "", id="language-field-name-is-no-fallback"
        ),
        pytest.param(["Back"], {}, "Back", "", id="first-field-is-the-last-resort"),
        pytest.param(
            ["Front", "Pinyin"],
            {"romanization_candidates": ("Pinyin",)},
            "Front",
            "Pinyin",
            id="romanization-candidate",
        ),
        pytest.param(
            ["Front", "Romanization"], {}, "Front", "Romanization", id="romanization-fallback"
        ),
        pytest.param(["Front", "Jyutping"], {}, "Front", "", id="no-romanization-field"),
    ],
)
def test_the_extraction_picks_the_text_and_romanization_fields(
    tmp_path: Path,
    fields: list[str],
    options: dict[str, Any],
    text: str,
    romanization: str,
) -> None:
    sentence = _extract(tmp_path, fields, options)

    assert sentence.text == text
    assert sentence.romanization == romanization


def test_the_counter_field_sets_the_original_order(tmp_path: Path) -> None:
    sentence = _extract(tmp_path, ["Front", "[counter 2]"], values={"[counter 2]": "7"})

    assert sentence.original_order == 7
