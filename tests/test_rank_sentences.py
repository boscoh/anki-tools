"""Tests for the Mandarin character helpers."""


from anki_tools.languages.mandarin.characters import (
    char_similarity_zh,
    get_chinese_chars,
    is_chinese_char,
)


class TestHelpers:
    def test_is_chinese_char(self):
        assert is_chinese_char("你")
        assert is_chinese_char("好")
        assert not is_chinese_char("a")
        assert not is_chinese_char("1")
        assert not is_chinese_char("！")

    def test_get_chinese_chars(self):
        assert get_chinese_chars("你好") == ["你", "好"]
        assert get_chinese_chars("Hello你好World") == ["你", "好"]
        assert get_chinese_chars("123") == []


class TestCharSimilarity:
    def test_identical(self):
        assert char_similarity_zh("你好", "你好") == 1.0

    def test_no_overlap(self):
        assert char_similarity_zh("你好", "再见") == 0.0

    def test_partial_overlap(self):
        sim = char_similarity_zh("你好", "你们好")
        assert 0 < sim < 1

    def test_empty_returns_zero(self):
        assert char_similarity_zh("", "你好") == 0.0
        assert char_similarity_zh("你好", "") == 0.0
