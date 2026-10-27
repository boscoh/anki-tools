"""Tests for the Mandarin character helpers and the Cantonese frequency data."""

import pytest

from anki_tools.languages.cantonese.frequency import (
    FreqRow,
    YueFrequency,
    to_frequency_data,
)
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


class TestYueFrequency:
    @pytest.fixture
    def frequency(self):
        counts = (("嘅", 900.0), ("我", 800.0), ("食", 10.0))
        rows = [
            FreqRow(word, "", 0, 0.0, count, count, rank=rank, source="ad")
            for rank, (word, count) in enumerate(counts, 1)
        ]
        return YueFrequency(rows)

    def test_ranks_follow_count_order(self, frequency):
        data = to_frequency_data(frequency, 50_000)
        assert data.word_freq == {"嘅": 1, "我": 2, "食": 3}

    def test_char_rank_from_first_word(self, frequency):
        data = to_frequency_data(frequency, 50_000)
        assert data.char_freq == {"嘅": 1, "我": 2, "食": 3}

    def test_max_words_truncates(self, frequency):
        data = to_frequency_data(frequency, 2)
        assert set(data.word_freq) == {"嘅", "我"}
