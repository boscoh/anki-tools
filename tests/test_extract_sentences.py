"""Tests for the deck field resolution of ``extract_sentences``."""

from __future__ import annotations

from anki_tools.ranking.rank import _resolve_extract_fields


def test_text_field_wins() -> None:
    text, _, _ = _resolve_extract_fields(["Front", "Back"], "Back", (), ())
    assert text == "Back"


def test_text_candidate_beats_fallback() -> None:
    text, _, _ = _resolve_extract_fields(["Front", "French"], None, ("French",), ())
    assert text == "French"


def test_fallback_order() -> None:
    text, _, _ = _resolve_extract_fields(["Front", "Sentence"], None, (), ())
    assert text == "Sentence"
    text, _, _ = _resolve_extract_fields(["Front", "Back"], None, (), ())
    assert text == "Front"


def test_language_field_name_is_no_fallback() -> None:
    """The shared resolver holds no language field name of its own."""
    text, _, _ = _resolve_extract_fields(["Front", "French"], None, (), ())
    assert text == "Front"


def test_first_field_is_the_last_resort() -> None:
    text, _, _ = _resolve_extract_fields(["Back"], None, (), ())
    assert text == "Back"
    text, _, _ = _resolve_extract_fields([], None, (), ())
    assert text == "Front"


def test_romanization_candidate() -> None:
    _, _, romanization = _resolve_extract_fields(
        ["Front", "Pinyin"], None, (), ("Pinyin",)
    )
    assert romanization == "Pinyin"


def test_romanization_fallback() -> None:
    _, _, romanization = _resolve_extract_fields(["Front", "Romanization"], None, (), ())
    assert romanization == "Romanization"


def test_no_romanization_field() -> None:
    _, _, romanization = _resolve_extract_fields(["Front", "Jyutping"], None, (), ())
    assert romanization is None


def test_order_field() -> None:
    _, order, _ = _resolve_extract_fields(["Front", "[counter 2]"], None, (), ())
    assert order == "[counter 2]"
