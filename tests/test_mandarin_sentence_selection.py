"""Tests for the Mandarin sentence candidate pool and the selection stage."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.mandarin.swadesh import (
    MANDARIN,
    SENTENCE_FIELDS,
    SENTENCE_REVIEW_FIELDS,
    Candidate,
    WordForm,
    build_candidate_pool,
    build_known_set,
    build_pack_report,
    check_sentence_shape,
    count_form_occurrences,
    han_tokens,
    has_orphan_conjunction,
    has_predicate,
    rank_key_parts,
    rank_order,
    select_sentences,
    sentence_hits,
    sentence_pinyin,
    serialize_rank_key,
    write_sentence_csv,
    write_sentence_review_csv,
)
from anki_tools.sentence_pack.languages.mandarin import sense_filter

POOL_SENTENCES = [
    (1, "我要喝水。", 100, "I want to drink water.", "alice", (5001, "CC BY 2.0 FR")),
    (2, "我要喝水", 200, "I want to drink water.", "alice", None),
    (3, "我要喝水。。", 300, "I want to drink water.", "alice", None),
    (4, "我要喝水了ABCD。", 400, "I want to drink water.", "alice", None),
    (5, "走走走走。", 500, "I walk and walk.", "alice", None),
    (6, "我，要，喝水。", 600, "I want to drink water.", "alice", None),
    (7, "他说“我要喝水”。", 700, "He said I want water.", "alice", None),
    (8, "我要喝水。", 800, "I want to drink water.", "alice", None),
    (10, "我喝那杯水。", 999, "I drink that glass of water.", "alice", None),
    (
        11,
        "我喝那杯凉水。",
        1100,
        "the quick brown fox jumps over the lazy dog again and again today",
        "alice",
        None,
    ),
    (12, "火灭了。", 1200, "The fire went out.", "bob", None),
    (13, "因为下雨。", 1300, "Because it rains.", "alice", None),
    (14, "那只小狗。", 1400, "That small dog.", "alice", None),
    (15, "饕餮来了。", 1500, "The monster came.", "bob", None),
]

POOL_AUTHORS = {row[0]: row[4] for row in POOL_SENTENCES}


def _write_tsv(path: Path, rows: list[tuple[object, ...]]) -> Path:
    text = "\n".join("\t".join(str(cell) for cell in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


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
    return replace(candidate, rank_key=serialize_rank_key(rank_key_parts(candidate)))


def _write_pool_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        data_dir / "cmn_sentences.tsv",
        [(row[0], "cmn", row[1]) for row in POOL_SENTENCES],
    )
    _write_tsv(
        data_dir / "cmn-eng_links.tsv",
        [(row[0], row[2]) for row in POOL_SENTENCES if row[0] != 9],
    )
    _write_tsv(
        data_dir / "eng_sentences.tsv",
        [(row[2], "eng", row[3]) for row in POOL_SENTENCES if row[2] != 999],
    )
    _write_tsv(
        data_dir / "cmn_sentences_detailed.tsv",
        [
            (row[0], "cmn", row[1], POOL_AUTHORS[row[0]], "\\N", "0000-00-00")
            for row in POOL_SENTENCES
        ],
    )
    _write_tsv(
        data_dir / "cmn_sentences_with_audio.tsv",
        [
            (row[0], row[5][0], row[4], row[5][1], "")
            for row in POOL_SENTENCES
            if row[5]
        ],
    )
    return data_dir


@pytest.fixture
def pool_table() -> list[WordForm]:
    return [
        _form(1, "water", "水"),
        _form(2, "fire", "火"),
    ]


@pytest.fixture
def pool_data(tmp_path: Path) -> Path:
    return _write_pool_data(tmp_path / "tatoeba")


class TestSentenceShape:
    def test_accepts_a_complete_sentence(self) -> None:
        assert check_sentence_shape("我要喝水。") is None

    def test_length_window_stops_at_seven(self) -> None:
        assert check_sentence_shape("我们今天去学校。") is None
        assert check_sentence_shape("我们今天要去学校。") == "F2"
        assert check_sentence_shape("我们今天要去学校。", max_han=8) is None

    def test_profile_window_is_four_to_seven(self) -> None:
        assert (MANDARIN.min_units, MANDARIN.max_units) == (4, 7)

    @pytest.mark.parametrize(
        ("sentence", "expected"),
        [
            ("我喝水。", "F2"),
            ("我要喝水我要喝水我要喝水。", "F2"),
            ("我要喝水ABCD。", "F3"),
            ("走走走走。", "F4"),
            ("我要喝水", "F5"),
            ("我要喝水。。", "F6"),
            ("他说“我要喝水”。", "F7"),
            ("我，要，喝水。", "F8"),
        ],
    )
    def test_rejects_in_order(self, sentence: str, expected: str) -> None:
        assert check_sentence_shape(sentence) == expected

    def test_fallback_window_accepts_short_sentence(self) -> None:
        assert check_sentence_shape("火灭了。") == "F2"
        assert check_sentence_shape("火灭了。", min_han=3) is None

    def test_predicate_and_conjunction(self) -> None:
        assert has_predicate("我要喝水。") is True
        assert has_predicate("那只小狗。") is False
        assert has_orphan_conjunction("因为下雨。") is True
        assert has_orphan_conjunction("因为下雨，我没去。") is False
        assert has_orphan_conjunction("我要喝水。") is False


class TestTokenChecks:
    def test_punctuation_stays_out(self) -> None:
        assert han_tokens("我要喝水。") == ["我", "要", "喝水"]

    def test_known_set_grows_with_the_word_table(
        self, pool_table: list[WordForm]
    ) -> None:
        known = build_known_set(pool_table)
        assert {"水", "火"} <= known
        assert "的" in known

    def test_unknown_tokens(self, pool_table: list[WordForm]) -> None:
        known = build_known_set(pool_table)
        assert [word for word in han_tokens("我要喝水。") if word not in known] == [
            "喝水"
        ]
        unknown = [word for word in han_tokens("饕餮来了。") if word not in known]
        assert unknown
        assert not set(unknown) & known

    def test_target_must_appear_once(self) -> None:
        assert count_form_occurrences("我要喝水。", "水") == 1
        assert count_form_occurrences("我要喝水，水很凉。", "水") == 2
        assert sentence_hits("我要喝水。", ["水", "火"]) == {"水": 1}
        assert sentence_hits("我要喝水，水很凉。", ["水"]) == {"水": 2}

    def test_false_friend_compound_does_not_hit(self) -> None:
        assert count_form_occurrences("你看新闻吗？", "闻") == 0
        assert count_form_occurrences("我闻花香。", "闻") == 1
        assert count_form_occurrences("一切都好。", "切") == 0
        assert count_form_occurrences("他的数学很好。", "数") == 0
        assert sentence_hits("一切都好。", ["切", "好"]) == {"好": 1}

    def test_sense_filter_rejects_false_friend_english(self) -> None:
        assert sense_filter("抓", "He grabbed a handful of rice.")
        assert not sense_filter("抓", "I scratched my arm.")
        assert sense_filter("数", "What are those numbers?")
        assert not sense_filter("数", "I counted to ten.")
        assert sense_filter("挤", "This room is cramped.")
        assert not sense_filter("挤", "Squeeze the lemon.")


class TestRankingKey:
    def test_key_order(self) -> None:
        best = _candidate(
            unknown_words=0, audio_id=7, zipf_mean=5.0, han_len=5, tatoeba_id=1
        )
        high_id = _candidate(audio_id=7, han_len=5, tatoeba_id=9)
        low_zipf = _candidate(audio_id=7, zipf_mean=3.0, han_len=5, tatoeba_id=4)
        short = _candidate(audio_id=7, han_len=4, tatoeba_id=5)
        no_audio = _candidate(han_len=5, tatoeba_id=3)
        more_unknown = _candidate(unknown_words=1, audio_id=7, han_len=5, tatoeba_id=2)
        shuffled = [more_unknown, no_audio, short, low_zipf, high_id, best]
        assert sorted(shuffled, key=rank_order) == [
            best,
            high_id,
            short,
            low_zipf,
            no_audio,
            more_unknown,
        ]

    def test_key_serialization(self) -> None:
        candidate = _candidate(unknown_words=2, zipf_mean=5.1234, tatoeba_id=42)
        assert candidate.rank_key == "2|1|-5.123|1|42"
        assert rank_key_parts(candidate) == (2, 1, -5.1234, 1, 42)


class TestCandidatePool:
    def test_filter_counts(self, pool_table: list[WordForm], pool_data: Path) -> None:
        pool, report = build_candidate_pool(pool_table, data_dir=pool_data)
        counts = report["counts"]
        assert report["total_linked"] == 14
        assert counts["F1"] == 13
        assert counts["F2"] == 12
        assert counts["F3"] == 11
        assert counts["F4"] == 10
        assert counts["F5"] == 9
        assert counts["F6"] == 8
        assert counts["F7"] == 7
        assert counts["F8"] == 6
        assert counts["F9"] == 5
        assert counts["F10"] == 4
        assert counts["F11"] == 3
        assert counts["F12"] == 2
        assert report["pool_candidates"] == len(pool)
        assert report["core_sentences"] == len(pool)

    def test_accepts_only_clean_sentences(
        self, pool_table: list[WordForm], pool_data: Path
    ) -> None:
        pool, _ = build_candidate_pool(pool_table, data_dir=pool_data)
        sentences = {candidate.sentence for candidate in pool}
        assert "我要喝水。" in sentences
        for candidate in pool:
            assert check_sentence_shape(candidate.sentence) is None
            assert candidate.tier == "core"
            assert candidate.tatoeba_author
            assert candidate.audio_source in {"native", "tts"}

    def test_native_audio_and_author_are_loaded(
        self, pool_table: list[WordForm], pool_data: Path
    ) -> None:
        pool, _ = build_candidate_pool(pool_table, data_dir=pool_data)
        first = next(candidate for candidate in pool if candidate.tatoeba_id == 1)
        assert first.audio_id == 5001
        assert first.audio_license == "CC BY 2.0 FR"
        assert first.audio_source == "native"
        assert first.audio_missing is False
        assert first.english == "I want to drink water."

    def test_unknown_budget(self, pool_table: list[WordForm], pool_data: Path) -> None:
        _, strict = build_candidate_pool(pool_table, data_dir=pool_data, max_unknown=0)
        _, loose = build_candidate_pool(pool_table, data_dir=pool_data, max_unknown=2)
        assert strict["counts"]["unknown"] <= loose["counts"]["unknown"]

    def test_fallback_scaffold(
        self, pool_table: list[WordForm], pool_data: Path
    ) -> None:
        core_pool, core_report = build_candidate_pool(pool_table, data_dir=pool_data)
        assert not any("火" in candidate.covered for candidate in core_pool)
        assert pool_table[1].candidate_count == 0

        pool, report = build_candidate_pool(
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
        cards, report = select_sentences([candidate], pool_table, per_word=1)
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
        cards, _ = select_sentences(pool, table, per_word=2)
        assert [card.target for card in cards] == ["老", "旧"]

    def test_per_word_limit(self, pool_table: list[WordForm]) -> None:
        pool = [
            _candidate(tatoeba_id=1, covered=frozenset({"水"})),
            _candidate(
                tatoeba_id=2, covered=frozenset({"水"}), sentence="我要喝水了。"
            ),
            _candidate(
                tatoeba_id=3, covered=frozenset({"水"}), sentence="我喝那杯水。"
            ),
        ]
        cards, report = select_sentences(pool, pool_table, per_word=2)
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
        cards, report = select_sentences(pool, pool_table, per_word=5, max_cards=3)
        assert len(cards) == 3
        assert report["max_cards"] == 3

    def test_same_sentence_is_never_used_twice(
        self, pool_table: list[WordForm]
    ) -> None:
        pool = [
            _candidate(tatoeba_id=1, covered=frozenset({"水"})),
            _candidate(tatoeba_id=1, covered=frozenset({"火"})),
        ]
        cards, _ = select_sentences(pool, pool_table, per_word=2)
        assert len({card.tatoeba_id for card in cards}) == len(cards)

    def test_scaffold_card_is_selected(self, pool_table: list[WordForm]) -> None:
        pool = [
            _candidate(tatoeba_id=1, covered=frozenset({"水"})),
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
        cards, report = select_sentences(pool, pool_table, per_word=1)
        assert {card.target: card.tier for card in cards} == {
            "水": "core",
            "火": "scaffold",
        }
        assert report["tier_counts"] == {"core": 1, "scaffold": 1}
        assert report["core_coverage"]["ratio"] == 1.0

    def test_similar_sentence_is_skipped(self, pool_table: list[WordForm]) -> None:
        pool = [
            _candidate(tatoeba_id=1, covered=frozenset({"水"}), sentence="我要喝水。"),
            _candidate(tatoeba_id=2, covered=frozenset({"水"}), sentence="我要喝水。"),
        ]
        cards, _ = select_sentences(pool, pool_table, per_word=2)
        assert len(cards) == 1

    def test_pinyin_is_filled(self, pool_table: list[WordForm]) -> None:
        cards, _ = select_sentences([_candidate()], pool_table, per_word=1)
        assert cards[0].pinyin == "wǒ yào hē shuǐ"


class TestPinyin:
    def test_sentence_pinyin(self) -> None:
        assert sentence_pinyin("我要喝水。") == "wǒ yào hē shuǐ"

    def test_bu_sandhi(self) -> None:
        assert sentence_pinyin("我不去。") == "wǒ bú qù"
        assert sentence_pinyin("我不喝。") == "wǒ bù hē"

    def test_yi_sandhi(self) -> None:
        assert sentence_pinyin("我喝一杯水。") == "wǒ hē yì bēi shuǐ"
        assert sentence_pinyin("我喝一瓶水。") == "wǒ hē yì píng shuǐ"
        assert sentence_pinyin("我喝一水。") == "wǒ hē yì shuǐ"
        assert sentence_pinyin("一。") == "yī"

    def test_syllable_count_matches_han_count(self) -> None:
        for sentence in ["我要喝水。", "我不知道，他来了。", "火灭了。"]:
            syllables = sentence_pinyin(sentence).split()
            han_count = sum(1 for char in sentence if "\u4e00" <= char <= "\u9fff")
            assert len(syllables) == han_count


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
        pool = [_candidate(tatoeba_id=1, covered=frozenset({"水"}))]
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
        cards, selection = select_sentences(
            [_candidate()], pool_table, per_word=1, max_cards=10
        )
        report = build_pack_report({"total_linked": 1, "counts": {}}, selection)
        assert report["cards"] == len(cards) == 1
        assert report["coverage"]["with_card"] == 1
        assert report["form_coverage"]["total"] == 2
