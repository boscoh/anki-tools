"""Tests for the Spanish cull profile and the Spanish frequency blend.

The tests that read ``data/spanish_frequency.tsv`` are skipped when the local
cache file is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_tools.apkg.templates import load_text
from anki_tools.languages.spanish import cull as cull_es
from anki_tools.languages.spanish import frequency
from anki_tools.ranking import cullflow
from anki_tools.languages.spanish.frequency import (
    LICENSE,
    SOURCE_NAME,
    SpanishFrequencyRow,
    cull_frequency_rows,
)

needs_cache = pytest.mark.skipif(
    not frequency.FREQUENCY_PATH.exists(),
    reason="the Spanish frequency cache is absent",
)


def _row(
    word: str,
    lemma: str,
    tatoeba_count: int = 10,
    tatoeba_pm: float = 250.0,
    written_pm: float = 100.0,
    zipf: float = 5.0,
) -> SpanishFrequencyRow:
    return SpanishFrequencyRow(
        word=word,
        lemma=lemma,
        count=1,
        pm=1.0,
        rank=1,
        pos="NOUN",
        source=SOURCE_NAME,
        license=LICENSE,
        tatoeba_count=tatoeba_count,
        tatoeba_pm=tatoeba_pm,
        written_pm=written_pm,
        zipf=zipf,
    )


@needs_cache
def test_the_spoken_weight_changes_the_blend_of_a_spanish_row() -> None:
    """A Spanish row holds a spoken term, so the weight changes the score."""
    row = next(row for row in cull_es.read_cull_frequency() if row.tatoeba_count >= 5)

    assert cullflow.blend_pm(row, spoken_weight=0.0) == pytest.approx(row.written_pm)
    assert cullflow.blend_pm(row, spoken_weight=1.0) == pytest.approx(row.tatoeba_pm)
    assert cullflow.blend_pm(row, 0.0) != pytest.approx(cullflow.blend_pm(row, 1.0))


@needs_cache
def test_the_cull_rows_hold_one_row_per_lemma() -> None:
    """The surface rows of the cache collapse to one cull row per lemma."""
    rows = cull_es.read_cull_frequency()
    words = [row.word for row in rows]

    assert len(rows) == len(set(words))
    assert sum(row.tatoeba_count for row in rows) > 0


def test_the_map_takes_the_lemma_and_the_two_new_columns() -> None:
    """The map aggregates the surface rows and keeps the spoken term."""
    rows = cull_frequency_rows(
        [
            _row("gato", "gato", 10, 250.0),
            _row("gatos", "gato", 10, 250.0),
            _row("perro", "perro", 30, 750.0),
        ]
    )

    assert [row.word for row in rows] == ["gato", "perro"]
    assert rows[0].tatoeba_count == 20
    assert rows[0].tatoeba_pm == pytest.approx(500.0)
    assert rows[0].written_pm == pytest.approx(100.0)
    assert rows[0].zipf == pytest.approx(5.0)
    assert rows[1].tatoeba_pm == pytest.approx(750.0)


def test_the_map_merges_the_clitic_lemmas() -> None:
    """The verb and its clitic lemma become one row of the cull."""
    rows = cull_frequency_rows(
        [
            _row("hacer", "hacer", 40, 20.0, 200.0, 5.3),
            _row("hacerlo", "hacer él", 60, 30.0, 0.001, 0.0),
        ]
    )

    assert [row.word for row in rows] == ["hacer"]
    assert rows[0].tatoeba_count == 100
    assert rows[0].written_pm == pytest.approx(200.0)


def test_the_clitic_lemma_drops_its_pronoun() -> None:
    """The Spanish model lemma keeps a clitic pronoun, and the cull drops it."""
    assert cull_es.base_lemma("hacer él") == "hacer"
    assert cull_es.base_lemma("acostar yo") == "acostar"
    assert cull_es.base_lemma("hacer") == "hacer"
    assert cull_es.base_lemma("sin embargo") == "sin embargo"


def test_the_reserve_fields_fit_the_output_note_type() -> None:
    """The reserve card fills the five output fields in the output order."""
    card = cullflow.ReserveCard(
        sentence="el gato come",
        html="el <b>gato</b> come",
        english="the cat eats",
        target="gato",
        gloss="cat",
        audio="a.mp3",
    )
    fields = cull_es._reserve_fields(card)

    assert fields == ["[sound:a.mp3]", "el <b>gato</b> come", "the cat eats", "gato", "cat"]


def test_the_cull_templates_come_from_files() -> None:
    """The Spanish cull deck reads its template files."""
    assert cull_es.FRONT_TEMPLATE == load_text(cull_es.TEMPLATE_DIR / "front.html")
    assert cull_es.BACK_TEMPLATE == load_text(cull_es.TEMPLATE_DIR / "back.html")
    assert cull_es.CARD_TEMPLATES == (
        (cull_es.FRONT_TEMPLATE, cull_es.BACK_TEMPLATE),
    )


def test_the_spanish_cull_names_no_other_language() -> None:
    """The Spanish adapter holds Spanish values only."""
    text = Path(cull_es.__file__).read_text(encoding="utf-8")

    assert "French" not in text

