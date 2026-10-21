"""Tests for the Cantonese frequency list, sentence rules, and selection."""

from pathlib import Path

import jieba
import pytest

from anki_tools.languages.cantonese.rank import load_frequency_data_yue
from anki_tools.ranking.rank import Sentence, yue_frequency_path
from anki_tools.languages.cantonese.frequency import (
    CifuEntry,
    FreqRow,
    YueFrequency,
    build_yue_frequency,
    build_yue_tokenizer,
    check_frequency,
    han_tokens,
    rank_change_rows,
    read_cifu,
    read_overrides,
    read_variants,
    write_freq_change_csv,
    write_frequency_tsv,
)
from anki_tools.languages.cantonese.select import (
    UNRANKED,
    Candidate,
    SelectionOptions,
    SelectionReport,
    check_sentence,
    greedy_select,
    read_names,
    select_yue_sentences,
    spoken_mass,
)

CIFU_HEADER = "Word\tJyutPing\tSpokenAdult\tWritten"


def write_cifu(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    lines = [CIFU_HEADER] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestCifuReader:
    def test_skips_header_and_reads_empty_counts_as_zero(self, tmp_path: Path) -> None:
        path = write_cifu(
            tmp_path / "cifu.tsv",
            [("係", "hai6", "10", "20"), ("嘅", "ge3", "", "7")],
        )
        entries = read_cifu(path)
        assert entries["係"] == CifuEntry("hai6", 10, 20)
        assert entries["嘅"] == CifuEntry("ge3", 0, 7)
        assert "Word" not in entries

    def test_sums_a_repeated_word(self, tmp_path: Path) -> None:
        path = write_cifu(
            tmp_path / "cifu.tsv", [("咁", "gam3", "4", "1"), ("咁", "gam2", "3", "2")]
        )
        assert read_cifu(path)["咁"].spoken == 7

    def test_missing_file_names_the_environment_variable(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="CIFU_PATH"):
            read_cifu(tmp_path / "absent.tsv")


CIFU = {
    "係": CifuEntry("hai6", 100, 10),
    "嘅": CifuEntry("ge3", 4, 400),
    "咁": CifuEntry("gam3", 50, 10),
    "噉": CifuEntry("gam2", 30, 0),
}
APPLE = {"係": 10, "嘅": 5000, "新聞": 9000, "咁": 20, "噉": 5}


class TestBlend:
    def test_count_below_five_uses_the_written_term(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, spoken_weight=1.0)
        row = next(r for r in data.rows if r.word == "嘅")
        assert row.blend_pm == pytest.approx(row.written_pm)

    def test_weight_changes_the_order(self) -> None:
        spoken = build_yue_frequency(CIFU, APPLE, spoken_weight=1.0)
        written = build_yue_frequency(CIFU, APPLE, spoken_weight=0.0)
        assert spoken.ranks["係"] < written.ranks["係"]
        assert written.ranks["嘅"] < written.ranks["係"]
        assert [r.word for r in spoken.rows] != [r.word for r in written.rows]

    def test_freq_written_ignores_spoken_counts(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, freq="written")
        assert data.ranks["新聞"] < data.ranks["係"]

    def test_unknown_source_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_yue_frequency(CIFU, APPLE, freq="loud")

    def test_ranks_are_contiguous_and_source_is_recorded(self) -> None:
        data = build_yue_frequency(CIFU, APPLE)
        assert [r.rank for r in data.rows] == list(range(1, len(data.rows) + 1))
        sources = {r.word: r.source for r in data.rows}
        assert sources["係"] == "cifu+ad"
        assert sources["新聞"] == "ad"


class TestVariantsAndOverrides:
    def test_variant_counts_merge_into_the_standard_form(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, variants={"噉": "咁"})
        words = {r.word: r for r in data.rows}
        assert "噉" not in words
        assert words["咁"].spoken_count == 80

    def test_lookup_maps_a_variant_without_changing_text(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, variants={"噉": "咁"})
        assert data.ranks["噉"] == data.ranks["咁"]
        sentence = "噉樣好"
        assert sentence == "噉樣好"

    def test_override_takes_the_value_of_the_named_word(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, overrides={"你好": "係"})
        row = next(r for r in data.rows if r.word == "你好")
        assert row.source == "override"
        assert abs(data.ranks["你好"] - data.ranks["係"]) <= 1

    def test_override_with_unknown_target_is_ignored(self) -> None:
        data = build_yue_frequency(CIFU, APPLE, overrides={"你好": "無此詞"})
        assert "你好" not in data.ranks

    def test_files_read_with_header_skipped(self, tmp_path: Path) -> None:
        variants = tmp_path / "v.tsv"
        variants.write_text("variant\tstandard\treason\n噉\t咁\tx\n", encoding="utf-8")
        overrides = tmp_path / "o.tsv"
        overrides.write_text("word\trank_like\treason\n你好\t多謝\tx\n", encoding="utf-8")
        assert read_variants(variants) == {"噉": "咁"}
        assert read_overrides(overrides) == {"你好": "多謝"}


class TestOutputAndGates:
    def test_tsv_has_one_row_per_word_sorted_by_rank(self, tmp_path: Path) -> None:
        data = build_yue_frequency(CIFU, APPLE)
        out = tmp_path / "list.tsv"
        write_frequency_tsv(data, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].split("\t")[0] == "word"
        assert len(lines) == len(data.rows) + 1
        assert lines[1].split("\t")[6] == "1"

    def test_small_list_fails_gate_g1(self) -> None:
        results = {gate: ok for gate, ok, _ in check_frequency(build_yue_frequency(CIFU, APPLE))}
        assert results["G1"] is False


class TestTokenizer:
    def test_global_jieba_result_is_unchanged(self, tmp_path: Path) -> None:
        text = "我愛北京天安門"
        before = jieba.lcut(text)
        entries = {"我哋": CifuEntry("ngo5 dei6", 100, 10), "北京天安門": CifuEntry("", 5, 5)}
        build_yue_tokenizer(entries, dict_path=tmp_path / "dict.txt")
        assert jieba.lcut(text) == before

    def test_splits_with_the_cifu_words_and_keeps_han_tokens(self, tmp_path: Path) -> None:
        entries = {
            "我哋": CifuEntry("ngo5 dei6", 100, 10),
            "鍾意": CifuEntry("zung1 ji3", 80, 5),
        }
        tokenizer = build_yue_tokenizer(entries, dict_path=tmp_path / "dict.txt")
        assert han_tokens(tokenizer, "我哋鍾意，") == ["我哋", "鍾意"]

    def test_override_words_join_the_dictionary(self, tmp_path: Path) -> None:
        entries = {"你": CifuEntry("", 50, 5), "好": CifuEntry("", 50, 5)}
        tokenizer = build_yue_tokenizer(
            entries, extra_words={"你好": "多謝"}, dict_path=tmp_path / "dict.txt"
        )
        assert han_tokens(tokenizer, "你好") == ["你好"]


class TestRankChangeReport:
    def test_reports_rows_past_the_threshold_sorted_by_move(self) -> None:
        before = [
            Sentence(note_id=n, text=f"s{n}", original_order=n) for n in (1, 2, 3)
        ]
        after = [before[2], before[0], before[1]]
        rows = rank_change_rows(before, after, threshold=0)
        assert [row["sentence"] for row in rows] == ["s3", "s1", "s2"]
        assert rows[0]["move"] == 2
        assert rows[1]["move"] == -1
        assert rows[2]["move"] == -1
        assert rank_change_rows(before, after, threshold=2) == []

    def test_writes_a_header_and_one_row_per_change(self, tmp_path: Path) -> None:
        rows = [{"sentence": "我明。", "rank_before": 5, "rank_after": 986, "move": -981}]
        out = tmp_path / "change.csv"
        write_freq_change_csv(rows, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "sentence,rank_before,rank_after,move"
        assert lines[1] == "我明。,5,986,-981"


def make_freq(words: list[str]) -> YueFrequency:
    rows = [
        FreqRow(word, "", 10, 1.0, 1.0, 1.0, rank=rank, source="cifu")
        for rank, word in enumerate(words, 1)
    ]
    return YueFrequency(rows)


def make_sentence(
    text: str, order: int, complexity: float = 0.0
) -> Sentence:
    return Sentence(
        note_id=order + 1,
        text=text,
        original_order=order,
        complexity_score=complexity,
    )


class StubTokenizer:
    def __init__(self, tokens: dict[str, list[str]]) -> None:
        self.tokens = tokens

    def lcut(self, text: str) -> list[str]:
        return self.tokens[text]


class TestSentenceRules:
    RANKS = {"我": 1, "食": 2, "飯": 3, "好": 4}
    OPTIONS = SelectionOptions()

    def check(self, text: str, tokens: list[str] | None = None) -> str:
        return check_sentence(text, tokens or [], self.RANKS, self.OPTIONS)

    def test_a_complete_sentence_passes(self) -> None:
        assert self.check("我食飯。") == ""

    def test_y1_length_window(self) -> None:
        assert self.check("食飯。") == "Y1"
        assert self.check("我哋今日去街市買嘢食好唔好。") == "Y1"
        assert self.check("我哋今日去街市買嘢食。") == ""

    def test_y2_latin_letter_or_digit(self) -> None:
        assert self.check("我食飯Tom。") == "Y2"
        assert self.check("我食飯3。") == "Y2"

    def test_y3_repeated_pattern(self) -> None:
        assert self.check("我食飯我食飯。") == "Y3"

    def test_y4_one_final_mark_at_the_end(self) -> None:
        assert self.check("我食飯") == "Y4"
        assert self.check("我食飯。？") == "Y4"
        assert self.check("我食飯。") == ""

    def test_y4_can_be_skipped(self) -> None:
        options = SelectionOptions(final_mark=False)
        assert check_sentence("我食飯", [], self.RANKS, options) == ""

    def test_y5_quotation_mark(self) -> None:
        assert self.check("「我食飯」。") == "Y5"

    def test_y6_at_most_one_internal_mark(self) -> None:
        assert self.check("我，食飯。") == ""
        assert self.check("我，食，飯。") == "Y6"

    def test_y7_unknown_token_budget(self) -> None:
        assert self.check("我食飯。", ["我", "食", "飯", "貓"]) == ""
        assert self.check("我食飯。", ["貓", "狗", "魚"]) == "Y7"

    def test_y8_foreign_name(self) -> None:
        options = SelectionOptions(names=("蕾拉",))
        assert check_sentence("蕾拉想走。", [], self.RANKS, options) == "Y8"
        assert check_sentence("森美想走。", [], self.RANKS, options) == ""
        assert check_sentence("蕾拉想走。", [], self.RANKS, self.OPTIONS) == ""

    def test_y8_comes_after_the_other_rules(self) -> None:
        options = SelectionOptions(names=("蕾拉",))
        assert check_sentence("蕾拉想走", [], self.RANKS, options) == "Y4"

    def test_rules_apply_in_order(self) -> None:
        assert self.check("「Tom」") == "Y1"


class TestNameFile:
    def test_reads_one_name_per_line_and_skips_the_header(self, tmp_path: Path) -> None:
        path = tmp_path / "names.tsv"
        path.write_text("name\treason\n蕾拉\tLayla\n\n森美\tSammy\n", encoding="utf-8")
        assert read_names(path) == ("蕾拉", "森美")

    def test_an_absent_file_gives_no_names(self, tmp_path: Path) -> None:
        assert read_names(tmp_path / "absent.tsv") == ()

    def test_the_repository_file_loads(self) -> None:
        assert "蕾拉" in read_names()


class TestGreedyCover:
    def test_a_duplicate_token_counts_once_and_per_word_stops_the_gain(self) -> None:
        ranks = {"甲": 1, "乙": 2, "丙": 500}
        candidates = [
            Candidate(make_sentence("甲甲。", 0), ["甲", "甲"], 2),
            Candidate(make_sentence("甲丙。", 1), ["甲", "丙"], 2),
            Candidate(make_sentence("乙。", 2), ["乙"], 2),
        ]
        result = greedy_select(candidates, ranks, SelectionOptions(per_word=1))
        assert [c.sentence.text for c in result.kept] == ["甲丙。", "乙。"]
        assert result.reasons[0] == "no_gain"

    def test_per_word_limits_the_cards_for_one_word(self) -> None:
        ranks = {"甲": 1}
        candidates = [
            Candidate(make_sentence(f"甲{word}。", i), ["甲", word], 2)
            for i, word in enumerate(("一", "二", "三"))
        ]
        result = greedy_select(candidates, ranks, SelectionOptions(per_word=2))
        assert len(result.kept) == 2
        assert sorted(result.reasons.values()) == ["no_gain"]
        assert result.word_counts["甲"] == 2

    def test_max_sentences_stops_the_selection(self) -> None:
        ranks = {"甲": 1, "乙": 2, "丙": 3}
        candidates = [
            Candidate(make_sentence("甲。", 0), ["甲"], 1),
            Candidate(make_sentence("乙。", 1), ["乙"], 1),
            Candidate(make_sentence("丙。", 2), ["丙"], 1),
        ]
        options = SelectionOptions(per_word=1, max_sentences=1)
        result = greedy_select(candidates, ranks, options)
        assert len(result.kept) == 1
        assert set(result.reasons.values()) == {"cap"}


class TestSimilarity:
    def test_same_token_set_is_a_duplicate(self) -> None:
        ranks = {"我": 1, "食": 2, "飯": 3}
        candidates = [
            Candidate(make_sentence("我食飯。", 0), ["我", "食", "飯"], 3),
            Candidate(make_sentence("飯食我。", 1), ["我", "食", "飯"], 3),
        ]
        result = greedy_select(candidates, ranks, SelectionOptions())
        assert len(result.kept) == 1
        assert result.reasons[1] == "similar"

    def test_same_unique_characters_with_a_high_jaccard_is_a_duplicate(self) -> None:
        ranks = {"我食": 1, "飯": 2, "我": 3, "食飯": 4}
        candidates = [
            Candidate(make_sentence("我食飯。", 0), ["我食", "飯"], 3),
            Candidate(make_sentence("我飯食。", 1), ["我", "食飯"], 3),
        ]
        result = greedy_select(candidates, ranks, SelectionOptions())
        assert len(result.kept) == 1
        assert result.reasons[1] == "similar"


class TestWordUseCap:
    def candidates(self, shared: str, words: list[str]) -> list[Candidate]:
        return [
            Candidate(make_sentence(f"{shared}{word}。", i), [shared, word], 2)
            for i, word in enumerate(words)
        ]

    def test_a_word_outside_the_top_100_stops_at_the_limit(self) -> None:
        ranks = {"甲": 2, "乙": 3, "丙": 4, "丁": 5, "學": 150}
        options = SelectionOptions(per_word=2, max_uses=2)
        result = greedy_select(self.candidates("學", ["甲", "乙", "丙", "丁"]), ranks, options)
        assert len(result.kept) == 2
        assert list(result.reasons.values()).count("use_cap") == 2
        assert result.word_counts["學"] == 2

    def test_a_top_100_word_is_exempt(self) -> None:
        ranks = {"甲": 2, "乙": 3, "丙": 4, "丁": 5, "我": 1}
        options = SelectionOptions(per_word=2, max_uses=2)
        result = greedy_select(self.candidates("我", ["甲", "乙", "丙", "丁"]), ranks, options)
        assert len(result.kept) == 4

    def test_a_word_twice_in_one_sentence_counts_once(self) -> None:
        ranks = {"甲": 2, "學": 150}
        candidates = [
            Candidate(make_sentence("學學。", 0), ["學", "學"], 2),
            Candidate(make_sentence("學甲。", 1), ["學", "甲"], 2),
        ]
        options = SelectionOptions(per_word=2, max_uses=2)
        result = greedy_select(candidates, ranks, options)
        assert len(result.kept) == 2
        assert result.word_counts["學"] == 2

    def test_zero_turns_the_limit_off(self) -> None:
        ranks = {"甲": 2, "乙": 3, "丙": 4, "丁": 5, "學": 150}
        options = SelectionOptions(per_word=2, max_uses=0)
        result = greedy_select(self.candidates("學", ["甲", "乙", "丙", "丁"]), ranks, options)
        assert len(result.kept) == 4

    def test_the_limit_must_be_at_least_per_word(self) -> None:
        with pytest.raises(ValueError):
            SelectionOptions(per_word=3, max_uses=2)


class TestSelectSentences:
    def report(self, order: str = "coverage") -> SelectionReport:
        freq = make_freq(["我", "食", "飯", "麵"])
        ranked = [
            make_sentence("我食飯。", 0),
            make_sentence("我食麵。", 1),
            make_sentence("飯食我。", 2),
        ]
        dropped = [make_sentence("我食飯我食飯。", 3)]
        tokenizer = StubTokenizer({
            "我食飯。": ["我", "食", "飯"],
            "我食麵。": ["我", "食", "麵"],
            "飯食我。": ["我", "食", "飯"],
            "我食飯我食飯。": ["我", "食", "飯"],
        })
        options = SelectionOptions(per_word=2, max_uses=0, order=order)
        return select_yue_sentences(ranked, dropped, freq, tokenizer, options)

    def test_marks_every_input_sentence_and_keeps_the_order(self) -> None:
        report = self.report()
        reasons = {row["sentence"]: row["reason"] for row in report.select_rows}
        assert len(report.select_rows) == 4
        assert reasons["我食飯。"] == ""
        assert reasons["飯食我。"] == "similar"
        assert reasons["我食飯我食飯。"] == "Y3"
        assert [sentence.text for sentence in report.ordered[:2]] == [
            "我食飯。",
            "我食麵。",
        ]

    def test_hardest_rank_does_not_decrease(self) -> None:
        report = self.report()
        kept = [sentence for sentence in report.ordered if sentence.kept]
        ranks = [sentence.hardest_rank for sentence in kept]
        assert ranks == sorted(ranks)
        assert kept[0].hardest_rank == 3
        assert kept[1].hardest_rank == 4

    def test_a_sentence_of_unranked_words_comes_last(self) -> None:
        freq = make_freq(["我", "食", "飯"])
        ranked = [
            make_sentence("我食飯。", 0),
            make_sentence("我食嘢。", 1),
        ]
        tokenizer = StubTokenizer({"我食飯。": ["我", "食", "飯"], "我食嘢。": ["我", "食", "嘢"]})
        report = select_yue_sentences(
            ranked, [], freq, tokenizer, SelectionOptions(per_word=2)
        )
        kept = [sentence for sentence in report.ordered if sentence.kept]
        assert [sentence.text for sentence in kept] == ["我食飯。", "我食嘢。"]
        assert kept[1].hardest_rank == UNRANKED

    def test_score_order_follows_the_ranker(self) -> None:
        report = self.report(order="score")
        kept = [sentence.text for sentence in report.ordered if sentence.kept]
        assert kept == ["我食飯。", "我食麵。"]

    def test_coverage_rows_have_a_status_for_each_top_word(self) -> None:
        report = self.report()
        rows = {row["word"]: row for row in report.coverage_rows}
        assert rows["我"]["status"] == "covered"
        assert rows["我"]["count"] == 2
        assert rows["麵"]["status"] == "partial"
        assert report.summary["kept"] == 2
        assert report.summary["top500"] == 4


class TestSpokenMass:
    def test_counts_the_share_of_spoken_tokens_in_the_kept_words(self) -> None:
        freq = YueFrequency([
            FreqRow("我", "", 75, 1.0, 1.0, 1.0, rank=1, source="cifu"),
            FreqRow("食", "", 25, 1.0, 1.0, 1.0, rank=2, source="cifu"),
        ])
        candidates = [Candidate(make_sentence("我。", 0), ["我"], 1)]
        assert spoken_mass(freq, candidates) == pytest.approx(75.0)


class TestWrittenRegression:
    @pytest.mark.skipif(
        not yue_frequency_path().exists(), reason="the Apple Daily cache is absent"
    )
    def test_written_source_does_not_read_cifu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the written source read Cifu")

        monkeypatch.setattr("anki_tools.languages.cantonese.frequency.load_yue_frequency", boom)
        data = load_frequency_data_yue(source="written", max_words=1000)
        assert data.word_freq
        assert data.word_freq.get("係", 0) > 0
