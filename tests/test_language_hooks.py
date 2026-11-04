"""One runner for each hook of a language profile, driven by ``language_cases``."""

import pytest

from tests.language_cases import (
    LANGUAGES,
    highlight_rules,
    hit_rules,
    match_rules,
    prepare_rules,
    unit_rules,
)


@pytest.mark.parametrize("language, sentence, form, expected", highlight_rules())
def test_highlight_bolds_the_matched_tokens(
    language: str, sentence: str, form: str, expected: str
) -> None:
    highlight = LANGUAGES[language].highlight
    assert highlight is not None
    assert highlight(sentence, form) == expected


@pytest.mark.parametrize("language, sentence, form, matches", match_rules())
def test_match_count_counts_the_matches_of_a_form(
    language: str, sentence: str, form: str, matches: int
) -> None:
    assert LANGUAGES[language].match_count(sentence, form) == matches


@pytest.mark.parametrize("language, sentence, forms, found", hit_rules())
def test_hits_keeps_the_forms_with_one_match(
    language: str, sentence: str, forms: tuple[str, ...], found: dict[str, int]
) -> None:
    assert LANGUAGES[language].hits(sentence, forms) == found


@pytest.mark.parametrize("language, raw, expected", prepare_rules())
def test_prepare_text_normalizes_a_raw_sentence(
    language: str, raw: str, expected: str
) -> None:
    assert LANGUAGES[language].prepare_text(raw) == expected


@pytest.mark.parametrize("language, sentence, expected", unit_rules())
def test_unit_count_counts_a_prepared_sentence(
    language: str, sentence: str, expected: int
) -> None:
    profile = LANGUAGES[language]

    assert profile.unit_count(profile.prepare_text(sentence)) == expected
