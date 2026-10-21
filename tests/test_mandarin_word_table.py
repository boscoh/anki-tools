"""Tests for the Mandarin Swadesh word table."""

import csv
from pathlib import Path

import pytest

from anki_tools.languages.mandarin.profile import (
    SWADESH_ROWS,
    WordForm,
    check_word_table,
    load_word_table,
    word_table_report,
)

WORD_FIELDS = ["english", "mandarin", "transliteration"]
REVIEW_HEADER = "lang,index,english,current,canonical,issue,fix,status\n"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _synthetic_rows() -> list[dict[str, str]]:
    return [
        {
            "english": f"concept {index}",
            "mandarin": chr(0x4E00 + index),
            "transliteration": "shuǐ",
        }
        for index in range(1, SWADESH_ROWS + 1)
    ]


class Fixture:
    """Paths and rows of a small or full word table fixture."""

    def __init__(
        self,
        tmp_path: Path,
        rows: list[dict[str, str]],
        review_rows: str = "",
        override_rows: list[dict[str, str]] | None = None,
    ) -> None:
        self.rows = rows
        self.csv_path = _write_csv(tmp_path / "words.csv", WORD_FIELDS, rows)
        self.swadesh_path = tmp_path / "swadesh.txt"
        self.write_labels([row["english"] for row in rows])
        self.overrides_path = _write_csv(
            tmp_path / "overrides.csv",
            ["mandarin", "pinyin", "reason"],
            override_rows or [],
        )
        self.review_path = tmp_path / "review.csv"
        self.review_path.write_text(REVIEW_HEADER + review_rows, encoding="utf-8")

    def write_labels(self, labels: list[str]) -> None:
        self.swadesh_path.write_text("\n".join(labels), encoding="utf-8")

    def load(self) -> list[WordForm]:
        return load_word_table(
            self.csv_path, self.swadesh_path, self.overrides_path
        )

    def check(self, table: list[WordForm]) -> list[str]:
        return check_word_table(
            table,
            csv_path=self.csv_path,
            swadesh_path=self.swadesh_path,
            overrides_path=self.overrides_path,
            review_path=self.review_path,
        )


@pytest.fixture
def full_fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path, _synthetic_rows())


@pytest.fixture
def small_fixture(tmp_path: Path) -> Fixture:
    rows = [
        {
            "english": "he/she/it/they (singular)",
            "mandarin": "他/她/它",
            "transliteration": "tā tā tā",
        },
        {"english": "old", "mandarin": "老/旧", "transliteration": "lǎo jiù"},
        {"english": "warm", "mandarin": "暖和", "transliteration": "nuǎn huo"},
    ]
    return Fixture(tmp_path, rows)


def test_slash_expansion_and_pinyin_alignment(small_fixture: Fixture) -> None:
    table = small_fixture.load()
    assert [(form.mandarin, form.pinyin) for form in table] == [
        ("他", "tā"),
        ("她", "tā"),
        ("它", "tā"),
        ("老", "lǎo"),
        ("旧", "jiù"),
        ("暖和", "nuǎn huo"),
    ]
    assert [form.variant for form in table] == [
        "primary",
        "alternate",
        "alternate",
        "primary",
        "alternate",
        "primary",
    ]
    assert [form.swadesh_index for form in table] == [1, 1, 1, 2, 2, 3]
    assert {form.english for form in table} == {
        "he/she/it/they (singular)",
        "old",
        "warm",
    }


def test_pinyin_count_mismatch_raises(tmp_path: Path) -> None:
    rows = [{"english": "old", "mandarin": "老/旧", "transliteration": "lǎo"}]
    fixture = Fixture(tmp_path, rows)
    with pytest.raises(ValueError, match="Han characters"):
        fixture.load()


def test_row_count_mismatch_raises(tmp_path: Path) -> None:
    rows = [{"english": "old", "mandarin": "老", "transliteration": "lǎo"}]
    fixture = Fixture(tmp_path, rows)
    fixture.write_labels(["old", "warm"])
    with pytest.raises(ValueError, match="Swadesh labels"):
        fixture.load()


def test_traditional_and_zipf_fields(small_fixture: Fixture) -> None:
    table = small_fixture.load()
    traditional = {form.mandarin: form.traditional for form in table}
    assert traditional["旧"] == "舊"
    assert traditional["老"] == "老"
    assert traditional["暖和"] == "暖和"
    for form in table:
        assert isinstance(form.zipf, float)
        assert form.zipf > 0.0
    assert table[0].candidate_count == 0
    assert table[0].tier == "core"


def test_clean_fixture_passes(full_fixture: Fixture) -> None:
    table = full_fixture.load()
    assert full_fixture.check(table) == []
    assert all(form.variant == "primary" for form in table)


def test_label_mismatch_detected(full_fixture: Fixture) -> None:
    table = full_fixture.load()
    labels = [row["english"] for row in full_fixture.rows]
    labels[4] = "wrong label"
    full_fixture.write_labels(labels)
    failures = full_fixture.check(table)
    assert len(failures) == 1
    assert "label 5" in failures[0]
    assert "'wrong label'" in failures[0]


def test_duplicate_form_detected(full_fixture: Fixture) -> None:
    full_fixture.rows[1]["mandarin"] = full_fixture.rows[0]["mandarin"]
    full_fixture.csv_path = _write_csv(
        full_fixture.csv_path, WORD_FIELDS, full_fixture.rows
    )
    table = full_fixture.load()
    failures = full_fixture.check(table)
    assert len(failures) == 1
    assert "duplicate form" in failures[0]


def test_review_file_gate(full_fixture: Fixture) -> None:
    table = full_fixture.load()
    full_fixture.review_path.write_text(
        REVIEW_HEADER + "cmn,7,concept 7,水 · shuǐ,水 · shuǐ,sense,,open\n",
        encoding="utf-8",
    )
    failures = full_fixture.check(table)
    assert len(failures) == 1
    assert "open row" in failures[0]

    full_fixture.review_path.unlink()
    failures = full_fixture.check(table)
    assert len(failures) == 1
    assert "missing" in failures[0]


def test_fixed_review_row_is_clean(full_fixture: Fixture) -> None:
    table = full_fixture.load()
    full_fixture.review_path.write_text(
        REVIEW_HEADER + "cmn,7,concept 7,水 · shuǐ,水 · shuǐ,sense,kept,kept\n",
        encoding="utf-8",
    )
    assert full_fixture.check(table) == []


def test_pinyin_override_is_accepted(tmp_path: Path) -> None:
    rows = [
        {"english": "water", "mandarin": "水", "transliteration": "shuǐ"},
        {"english": "fire", "mandarin": "火", "transliteration": "huǒ"},
    ]
    fixture = Fixture(
        tmp_path,
        rows,
        override_rows=[{"mandarin": "水", "pinyin": "shui", "reason": "test"}],
    )
    table = fixture.load()
    assert table[0].pinyin == "shui"
    assert table[1].pinyin == "huǒ"
    failures = check_word_table(
        table,
        csv_path=fixture.csv_path,
        swadesh_path=fixture.swadesh_path,
        overrides_path=fixture.overrides_path,
        review_path=fixture.review_path,
    )
    assert not any("pinyin" in failure for failure in failures)


def test_report_counts(full_fixture: Fixture) -> None:
    report = word_table_report(full_fixture.load())
    assert report["words"] == SWADESH_ROWS
    assert report["forms"] == SWADESH_ROWS
    assert report["unique_forms"] == SWADESH_ROWS
    assert report["uncovered"] == []
    assert report["zipf_below_3"] == len(report["low_zipf_forms"])
