"""Tests for sentence ranking functions."""

import pytest

from rank_sentences import (
    Sentence,
    FrequencyData,
    complexity_score,
    frequency_score,
    get_word_rank,
    char_similarity,
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


class TestComplexityScore:
    def test_simple_sentence(self):
        score = complexity_score("你好")
        assert 0 < score < 20  # Simple sentence should have low complexity

    def test_complex_sentence(self):
        score = complexity_score("这是一个非常复杂的句子，包含很多不同的汉字")
        assert score > 50  # Complex sentence should have high complexity

    def test_single_char(self):
        score = complexity_score("好")
        assert score < 15  # Single char is very simple

    def test_empty_returns_zero(self):
        assert complexity_score("") == 0.0
        assert complexity_score("hello") == 0.0  # No Chinese chars


class TestFrequencyScore:
    @pytest.fixture
    def freq_data(self):
        """Simple frequency data for testing."""
        data = FrequencyData()
        data.word_freq = {
            "我": 1,
            "你": 2,
            "好": 3,
            "是": 4,
            "的": 5,
            "不": 6,
        }
        data.char_freq = data.word_freq.copy()
        data.hsk_vocab = {
            "我": 1,
            "你": 1,
            "好": 1,
            "是": 1,
        }
        return data

    def test_high_frequency_words(self, freq_data):
        score = frequency_score("你好", freq_data)
        assert score > 80  # Common words should have high score

    def test_unknown_words_lower_score(self, freq_data):
        score = frequency_score("稀罕词语", freq_data)
        assert score < 50  # Unknown words should have lower score

    def test_empty_returns_zero(self, freq_data):
        assert frequency_score("", freq_data) == 0.0

    def test_no_freq_data_returns_neutral(self):
        empty_data = FrequencyData()
        assert frequency_score("你好", empty_data) == 50.0


class TestGetWordRank:
    @pytest.fixture
    def freq_data(self):
        data = FrequencyData()
        data.word_freq = {"你": 2, "好": 3}
        data.char_freq = {"你": 2, "好": 3}
        return data

    def test_known_word(self, freq_data):
        assert get_word_rank("你", freq_data) == 2

    def test_unknown_word_fallback_to_chars(self, freq_data):
        # 你好 not in word_freq, should average char ranks (2+3)/2 = 2.5 -> 2
        rank = get_word_rank("你好", freq_data)
        assert rank == 2  # Average of 2 and 3, truncated

    def test_unknown_chars_max_rank(self, freq_data):
        rank = get_word_rank("稀罕", freq_data, max_rank=50000)
        assert rank == 50000


class TestCharSimilarity:
    def test_identical(self):
        assert char_similarity("你好", "你好") == 1.0

    def test_no_overlap(self):
        assert char_similarity("你好", "再见") == 0.0

    def test_partial_overlap(self):
        sim = char_similarity("你好", "你们好")
        assert 0 < sim < 1

    def test_empty_returns_zero(self):
        assert char_similarity("", "你好") == 0.0
        assert char_similarity("你好", "") == 0.0
