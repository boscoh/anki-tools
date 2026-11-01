"""Tests for the Mandarin sentence candidate pool and the selection stage."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.languages.mandarin.profile import (
    MANDARIN,
    SENTENCE_FIELDS,
    SENTENCE_REVIEW_FIELDS,
    Candidate,
    WordForm,
    known_words,
    count_form_occurrences,
    han_tokens,
    sentence_pinyin,
    single_hits,
    write_sentence_csv,
    write_sentence_review_csv,
)
from anki_tools.sentence_pack.english_sense import sense_failure, sense_row
from anki_tools.languages.mandarin.stages import (
    mandarin_candidate_pool,
    mandarin_rank_key_parts,
    mandarin_rank_order,
    select_mandarin_sentences,
)
from anki_tools.sentence_pack.pool import build_pack_report
from anki_tools.sentence_pack.select import rank_key_parts, serialize_rank_key
from tests.tatoeba_fixture import PoolRow, write_tatoeba_dir

POOL_ROWS = [
    PoolRow(
        1,
        "我要喝水。",
        100,
        "I want to drink water.",
        audio=(5001, "alice", "CC BY 2.0 FR"),
    ),
    PoolRow(2, "我要喝水", 200, "I want to drink water."),
    PoolRow(3, "我要喝水。。", 300, "I want to drink water."),
    PoolRow(4, "我要喝水了ABCD。", 400, "I want to drink water."),
    PoolRow(5, "走走走走。", 500, "I walk and walk."),
    PoolRow(6, "我，要，喝水。", 600, "I want to drink water."),
    PoolRow(7, "他说“我要喝水”。", 700, "He said I want water."),
    PoolRow(8, "我要喝水。", 800, "I want to drink water."),
    PoolRow(10, "我喝那杯水。", 999, None),
    PoolRow(
        11,
        "我喝那杯凉水。",
        1100,
        "the quick brown fox jumps over the lazy dog again and again today",
    ),
    PoolRow(12, "火灭了。", 1200, "The fire went out.", author="bob"),
    PoolRow(13, "因为下雨。", 1300, "Because it rains."),
    PoolRow(14, "那只小狗。", 1400, "That small dog."),
    PoolRow(15, "饕餮来了。", 1500, "The monster came.", author="bob"),
]


def _form(
    index: int, english: str, mandarin: str, variant: str = "primary"
) -> WordForm:
    return WordForm(
        swadesh_index=index,
        english=english,
        mandarin=mandarin,
        pinyin="shui",
        traditional=mandarin,
        variant=variant,
        zipf=5.0,
    )


def _candidate(**overrides: object) -> Candidate:
    values: dict[str, object] = {
        "target": "水",
        "sentence": "我要喝水。",
        "text_original": "我要喝水。",
        "pinyin": "",
        "english": "I want to drink water.",
        "tatoeba_id": 1,
        "tatoeba_author": "alice",
        "english_id": 100,
        "audio_id": 0,
        "audio_license": "",
        "audio_source": "tts",
        "unknown_words": 0,
        "han_len": 4,
        "zipf_mean": 5.0,
        "rank_key": "",
        "tier": "core",
        "audio_missing": True,
        "covered": frozenset({"水"}),
    }
    values.update(overrides)
    candidate = Candidate(**values)  # type: ignore[arg-type]
    return replace(
        candidate, rank_key=serialize_rank_key(mandarin_rank_key_parts(candidate))
    )


@pytest.fixture
def pool_table() -> list[WordForm]:
    return [
        _form(1, "water", "水"),
        _form(2, "fire", "火"),
    ]


@pytest.fixture
def pool_data(tmp_path: Path) -> Path:
    return write_tatoeba_dir(tmp_path / "tatoeba", "cmn", POOL_ROWS)


def test_unknown_tokens_come_from_the_word_table(pool_table: list[WordForm]) -> None:
    known = known_words(pool_table)

    assert {"水", "火", "的"} <= known
    assert han_tokens("我要喝水。") == ["我", "要", "喝水"]
    assert [word for word in han_tokens("我要喝水。") if word not in known] == ["喝水"]
    assert not {word for word in han_tokens("饕餮来了。") if word not in known} & known


@pytest.mark.parametrize(
    ("sentence", "form", "expected"),
    [
        pytest.param("我要喝水。", "水", 1, id="once"),
        pytest.param("我要喝水，水很凉。", "水", 2, id="twice"),
        pytest.param("你看新闻吗？", "闻", 0, id="inside a compound"),
        pytest.param("我闻花香。", "闻", 1, id="the verb"),
        pytest.param("一切都好。", "切", 0, id="compound with yi"),
        pytest.param("他的数学很好。", "数", 0, id="compound with xue"),
    ],
)
def test_a_form_inside_a_compound_does_not_hit(
    sentence: str, form: str, expected: int
) -> None:
    assert count_form_occurrences(sentence, form) == expected


def test_a_target_must_appear_once() -> None:
    assert single_hits("我要喝水。", ["水", "火"]) == {"水": 1}
    assert single_hits("我要喝水，水很凉。", ["水"]) == {}
    assert single_hits("一切都好。", ["切", "好"]) == {"好": 1}


@pytest.mark.parametrize(
    ("form", "english", "expected"),
    [
        ("抓", "He grabbed a handful of rice.", True),
        ("抓", "I scratched my arm.", False),
        ("数", "What are those numbers?", True),
        ("数", "I counted to ten.", False),
        ("挤", "This room is cramped.", True),
        ("挤", "Squeeze the lemon.", False),
    ],
)
def test_sense_rows_reject_false_friend_english(
    form: str, english: str, expected: bool
) -> None:
    row = sense_row(MANDARIN.sense_path, "S", form)

    assert (sense_failure(english, row, "") == "false_friend") == expected


class TestRankingKey:
    def test_key_order(self) -> None:
        best = _candidate(unknown_words=0, zipf_mean=5.0, han_len=5, tatoeba_id=1)
        high_id = _candidate(han_len=5, tatoeba_id=9)
        low_zipf = _candidate(zipf_mean=3.0, han_len=5, tatoeba_id=4)
        short = _candidate(han_len=4, tatoeba_id=5)
        no_audio = _candidate(han_len=5, tatoeba_id=3)
        more_unknown = _candidate(unknown_words=1, han_len=5, tatoeba_id=2)
        shuffled = [more_unknown, short, no_audio, low_zipf, high_id, best]
        assert sorted(shuffled, key=mandarin_rank_order) == [
            best,
            no_audio,
            high_id,
            short,
            low_zipf,
            more_unknown,
        ]

    def test_key_serialization(self) -> None:
        candidate = _candidate(unknown_words=2, zipf_mean=5.1234, tatoeba_id=42)
        assert candidate.rank_key == "2|0|-5.123|1|42"
        assert mandarin_rank_key_parts(candidate) == (2, 0, -5.1234, 1, 42)

    @pytest.mark.parametrize(
        ("audio_id", "prefer_native", "penalty"),
        [(7, True, 0), (0, True, 1), (7, False, 0), (0, False, 0)],
    )
    def test_the_audio_term_follows_the_profile_flag(
        self, audio_id: int, prefer_native: bool, penalty: int
    ) -> None:
        profile = replace(MANDARIN, prefer_native_audio=prefer_native)

        assert rank_key_parts(_candidate(audio_id=audio_id), profile)[1] == penalty


class TestCandidatePool:
    def test_filter_counts(self, pool_table: list[WordForm], pool_data: Path) -> None:
        pool, report = mandarin_candidate_pool(pool_table, data_dir=pool_data)
        counts = report["counts"]
        assert report["total_linked"] == 14
        assert report["english_paired"] == 13
        assert counts["length_window"] == 12
        assert counts["script"] == 11
        assert counts["repeat"] == 10
        assert counts["marks"] == 6
        assert counts["unique"] == 5
        assert report["pool_candidates"] == len(pool)
        assert report["core_sentences"] == len(pool)

    def test_the_audio_tier_is_off_and_the_author_is_loaded(
        self, pool_table: list[WordForm], pool_data: Path
    ) -> None:
        """Decision anki-gj6: the Mandarin profile has no native tier."""
        pool, _ = mandarin_candidate_pool(pool_table, data_dir=pool_data)
        first = next(candidate for candidate in pool if candidate.tatoeba_id == 1)
        assert first.sentence == "我要喝水。"
        assert first.tier == "core"
        assert first.audio_id == 0
        assert first.audio_license == ""
        assert first.audio_source == "tts"
        assert first.audio_missing is True
        assert first.tatoeba_author == "alice"
        assert first.english == "I want to drink water."

    def test_fallback_scaffold(
        self, pool_table: list[WordForm], pool_data: Path
    ) -> None:
        core_pool, core_report = mandarin_candidate_pool(pool_table, data_dir=pool_data)
        assert not any("火" in candidate.covered for candidate in core_pool)
        assert pool_table[1].candidate_count == 0

        pool, report = mandarin_candidate_pool(
            pool_table, data_dir=pool_data, fallback=True
        )
        scaffold = [candidate for candidate in pool if candidate.tier == "scaffold"]
        assert [candidate.target for candidate in scaffold] == ["火"]
        assert scaffold[0].sentence == "火灭了。"
        assert report["scaffold_sentences"] == 1
        assert report["scaffold_candidates"] == 1
        assert report["forms_with_core_candidate"] == 1
        assert core_report["scaffold_candidates"] == 0
        assert pool_table[1].candidate_count == 1
        assert pool_table[1].tier == "scaffold"


class TestSelection:
    def test_sentence_covers_two_targets(self, pool_table: list[WordForm]) -> None:
        candidate = _candidate(covered=frozenset({"水", "火"}))
        cards, report = select_mandarin_sentences([candidate], pool_table, per_word=1)
        assert [card.target for card in cards] == ["水"]
        assert report["cards"] == 1
        assert report["uncovered"] == [
            {
                "target": "火",
                "swadesh_index": 2,
                "english": "fire",
                "pool_candidates": 1,
            }
        ]
        assert report["cut_by_cap"] == ["火"]

    def test_primary_form_comes_first(self) -> None:
        table = [_form(1, "old", "老"), _form(1, "old", "旧", variant="alternate")]
        pool = [
            _candidate(
                target="旧",
                sentence="旧书很多。",
                covered=frozenset({"旧"}),
                tatoeba_id=2,
            ),
            _candidate(
                target="老",
                sentence="老人来了。",
                covered=frozenset({"老"}),
                tatoeba_id=3,
            ),
        ]
        cards, _ = select_mandarin_sentences(pool, table, per_word=2)
        assert [card.target for card in cards] == ["老", "旧"]

    def test_per_word_limit(self, pool_table: list[WordForm]) -> None:
        pool = [
            _candidate(tatoeba_id=1),
            _candidate(tatoeba_id=2, sentence="我要喝水了。"),
            _candidate(tatoeba_id=3, sentence="我喝那杯水。"),
        ]
        cards, report = select_mandarin_sentences(pool, pool_table, per_word=2)
        assert len(cards) == 2
        assert report["tier_counts"]["core"] == 2

    def test_card_cap(self, pool_table: list[WordForm]) -> None:
        sentences = [
            "我要喝水。",
            "他吃饭了。",
            "今天很冷。",
            "外面下雪。",
            "我喜欢你。",
        ]
        pool = [
            _candidate(tatoeba_id=index, sentence=sentence)
            for index, sentence in enumerate(sentences, start=1)
        ]
        cards, report = select_mandarin_sentences(
            pool, pool_table, per_word=5, max_cards=3
        )
        assert len(cards) == 3
        assert report["max_cards"] == 3

    @pytest.mark.parametrize(
        "second",
        [
            pytest.param(
                {
                    "tatoeba_id": 1,
                    "sentence": "他吃饭了。",
                    "covered": frozenset({"火"}),
                },
                id="same id",
            ),
            pytest.param({"tatoeba_id": 2}, id="same text"),
        ],
    )
    def test_a_repeated_sentence_is_skipped(
        self, pool_table: list[WordForm], second: dict[str, object]
    ) -> None:
        pool = [_candidate(tatoeba_id=1), _candidate(**second)]
        cards, _ = select_mandarin_sentences(pool, pool_table, per_word=2)
        assert len(cards) == 1

    def test_scaffold_card_is_selected(self, pool_table: list[WordForm]) -> None:
        pool = [
            _candidate(tatoeba_id=1),
            _candidate(
                target="火",
                tier="scaffold",
                tatoeba_id=12,
                sentence="火灭了。",
                text_original="火灭了。",
                english="The fire went out.",
                han_len=3,
                covered=frozenset({"火"}),
            ),
        ]
        cards, report = select_mandarin_sentences(pool, pool_table, per_word=1)
        assert {card.target: card.tier for card in cards} == {
            "水": "core",
            "火": "scaffold",
        }
        assert report["tier_counts"] == {"core": 1, "scaffold": 1}
        assert report["core_coverage"]["ratio"] == 1.0

    def test_pinyin_is_filled(self, pool_table: list[WordForm]) -> None:
        cards, _ = select_mandarin_sentences([_candidate()], pool_table, per_word=1)
        assert cards[0].pinyin == "wǒ yào hē shuǐ"


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("我要喝水。", "wǒ yào hē shuǐ"),
        ("我不去。", "wǒ bú qù"),
        ("我不喝。", "wǒ bù hē"),
        ("我喝一杯水。", "wǒ hē yì bēi shuǐ"),
        ("我喝一瓶水。", "wǒ hē yì píng shuǐ"),
        ("我喝一水。", "wǒ hē yì shuǐ"),
        ("一。", "yī"),
    ],
)
def test_sentence_pinyin_applies_the_tone_sandhi(sentence: str, expected: str) -> None:
    assert sentence_pinyin(sentence) == expected


class TestWriters:
    def test_sentence_csv_header(self, tmp_path: Path) -> None:
        path = write_sentence_csv([_candidate()], tmp_path / "sentences.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == SENTENCE_FIELDS
        assert rows[1][0] == "水"
        assert rows[1][SENTENCE_FIELDS.index("audio_missing")] == "true"

    def test_review_csv_header_and_rows(
        self, pool_table: list[WordForm], tmp_path: Path
    ) -> None:
        pool = [_candidate(tatoeba_id=1)]
        path = write_sentence_review_csv(pool, pool_table, path=tmp_path / "review.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0].keys()) == SENTENCE_REVIEW_FIELDS
        by_target = {row["target"]: row for row in rows}
        assert by_target["火"]["status"] == "uncovered"
        assert by_target["火"]["rank"] == "0"
        assert by_target["水"]["status"] == "thin"
        assert by_target["水"]["sentence"] == "我要喝水。"

    def test_pack_report_has_coverage(self, pool_table: list[WordForm]) -> None:
        cards, selection = select_mandarin_sentences(
            [_candidate()], pool_table, per_word=1, max_cards=10
        )
        report = build_pack_report({"total_linked": 1, "counts": {}}, selection)
        assert report["cards"] == len(cards) == 1
        assert report["coverage"]["with_card"] == 1
        assert report["form_coverage"]["total"] == 2
