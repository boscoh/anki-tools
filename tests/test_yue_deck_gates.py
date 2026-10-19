"""Gate checks for the committed LTL Cantonese deck selection.

These tests read the CSV files that ``anki fix yue`` writes for the LTL deck.
They need the Cifu cache; ``anki freq yue`` fills it. Without the cache the
tests skip, so a fresh clone stays green and the suite never downloads.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import pytest

from anki_tools.rank import char_similarity_zh_same_len, get_chinese_chars
from anki_tools.yue_frequency import (
    Tokenizer,
    YueFrequency,
    build_yue_tokenizer,
    cifu_path,
    download_cifu,
    han_tokens,
    load_yue_frequency,
    read_cifu,
    read_overrides,
)

BASE = Path("apkg/LTL Cantonese Deck Level 1 - Short")
RANK_CSV = BASE.with_name(BASE.name + ".rank.csv")
SELECT_CSV = BASE.with_name(BASE.name + ".select.csv")
DECK = BASE.with_name(BASE.name + ".apkg")
DECK_SENTENCES = 999
SPEC_CARDS = 647
SPEC_TOP500 = 223
MAX_USES = 8
EXEMPT_TOP = 100
SIMILARITY = 0.67

pytestmark = pytest.mark.skipif(
    not (RANK_CSV.exists() and SELECT_CSV.exists() and cifu_path().exists()),
    reason="the LTL selection CSV files and the Cifu cache are absent",
)


@pytest.fixture(scope="module")
def rank_rows() -> list[dict[str, str]]:
    with open(RANK_CSV, encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["kept"] == "1"]


@pytest.fixture(scope="module")
def select_rows() -> list[dict[str, str]]:
    with open(SELECT_CSV, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def freq() -> YueFrequency:
    return load_yue_frequency()


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return build_yue_tokenizer(read_cifu(download_cifu()), extra_words=read_overrides())


@pytest.fixture(scope="module")
def kept_tokens(
    rank_rows: list[dict[str, str]],
    freq: YueFrequency,
    tokenizer: Tokenizer,
) -> list[list[str]]:
    return [
        [freq.variants.get(token, token) for token in han_tokens(tokenizer, row["sentence"])]
        for row in rank_rows
    ]


def test_g5_no_kept_sentence_fails_a_rule(select_rows: list[dict[str, str]]) -> None:
    failures = [row["sentence"] for row in select_rows if row["kept"] == "1" and row["reason"]]
    assert failures == []


def test_g6_no_duplicate_kept_sentences(rank_rows: list[dict[str, str]]) -> None:
    token_sets = {row["sentence"]: row for row in rank_rows}
    assert len(token_sets) == len(rank_rows)
    by_size: dict[int, list[str]] = {}
    for row in rank_rows:
        text = row["sentence"]
        size = len(set(get_chinese_chars(text)))
        assert all(
            char_similarity_zh_same_len(text, other) < SIMILARITY
            for other in by_size.get(size, [])
        )
        by_size.setdefault(size, []).append(text)


def test_g7_one_select_row_per_input_sentence(select_rows: list[dict[str, str]]) -> None:
    expected = DECK_SENTENCES
    assert len(select_rows) == expected
    assert len({row["sentence"] for row in select_rows}) == expected


def test_g9_hardest_rank_does_not_decrease(rank_rows: list[dict[str, str]]) -> None:
    ranks = [int(row["hardest_rank"]) for row in rank_rows]
    assert ranks == sorted(ranks)


def test_g11_card_count_is_within_five_percent(rank_rows: list[dict[str, str]]) -> None:
    assert abs(len(rank_rows) - SPEC_CARDS) <= 0.05 * SPEC_CARDS


def test_g11_kept_words_and_top500(
    kept_tokens: list[list[str]],
    freq: YueFrequency,
    rank_rows: list[dict[str, str]],
) -> None:
    words = {token for tokens in kept_tokens for token in tokens}
    assert len(words) >= 940
    top500 = sum(1 for row in freq.rows[:500] if row.word in words)
    assert top500 >= SPEC_TOP500
    kept_ids = {row["note_id"] for row in rank_rows}
    assert len(kept_ids) == len(rank_rows)


def test_g12_word_use_cap(kept_tokens: list[list[str]], freq: YueFrequency) -> None:
    uses: Counter[str] = Counter()
    for tokens in kept_tokens:
        uses.update(set(tokens))
    ranks = freq.ranks
    limited = {
        word: count
        for word, count in uses.items()
        if ranks.get(word, math.inf) > EXEMPT_TOP
    }
    assert max(limited.values(), default=0) <= MAX_USES


def test_spoken_mass(kept_tokens: list[list[str]], freq: YueFrequency) -> None:
    words = {token for tokens in kept_tokens for token in tokens}
    total = sum(row.spoken_count for row in freq.rows)
    covered = sum(row.spoken_count for row in freq.rows if row.word in words)
    assert total > 0
    assert 100 * covered / total >= 66.0
