"""Tests for the Mandarin character helpers."""

import pytest

from anki_tools.languages.mandarin.characters import (
    char_similarity_zh,
    get_chinese_chars,
    is_chinese_char,
)


@pytest.mark.parametrize(
    "char, expected",
    [("你", True), ("好", True), ("a", False), ("1", False), ("！", False)],
)
def test_is_chinese_char(char: str, expected: bool) -> None:
    assert is_chinese_char(char) is expected


@pytest.mark.parametrize(
    "text, expected",
    [("你好", ["你", "好"]), ("Hello你好World", ["你", "好"]), ("123", [])],
)
def test_get_chinese_chars_keeps_only_the_chinese_characters(
    text: str, expected: list[str]
) -> None:
    assert get_chinese_chars(text) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("你好", "你好", 1.0),
        ("你好", "再见", 0.0),
        ("", "你好", 0.0),
        ("你好", "", 0.0),
    ],
)
def test_char_similarity_is_zero_to_one(left: str, right: str, expected: float) -> None:
    assert char_similarity_zh(left, right) == expected


def test_a_partial_overlap_gives_a_fraction() -> None:
    assert 0 < char_similarity_zh("你好", "你们好") < 1
