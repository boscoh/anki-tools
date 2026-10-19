"""Tests for the Cantonese frequency list, sentence rules, and selection."""

from pathlib import Path

import jieba
import pytest

from anki_tools.yue_frequency import (
    CifuEntry,
    build_yue_frequency,
    build_yue_tokenizer,
    check_frequency,
    han_tokens,
    read_cifu,
    read_overrides,
    read_variants,
    write_frequency_tsv,
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
