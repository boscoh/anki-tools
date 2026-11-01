"""Tests for the jieba tokens of the sense-file compounds."""

from __future__ import annotations

import pytest

from anki_tools.languages.mandarin import core


@pytest.mark.parametrize(
    ("sentence", "tokens"),
    [
        ("我听得到他。", ["我", "听得到", "他"]),
        ("我得到他。", ["我", "得到", "他"]),
    ],
)
def test_a_compound_is_one_tier_f_token(sentence: str, tokens: list[str]) -> None:
    words = [word for word, _pos in core.tokenize_texts([sentence])[0] if word != "。"]

    assert words == tokens
