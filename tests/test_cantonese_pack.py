"""Tests for the Cantonese Swadesh sentence pack (S1 to S5 and the gates)."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest

from anki_tools.sentence_pack import pool
from anki_tools.languages import cantonese
from anki_tools.sentence_pack.guards import check_sentence_shape


@pytest.fixture(autouse=True)
def _clear_hooks() -> Iterator[None]:
    """Clear the caches that the hooks share between tests."""
    cantonese.han_spans.cache_clear()
    cantonese.table_forms.cache_clear()
    yield
    cantonese.han_spans.cache_clear()
    cantonese.table_forms.cache_clear()


class FakeTokenizer:
    """Small tokenizer with the longest-word match of jieba."""

    def __init__(self, words: list[str]) -> None:
        self.words: list[str] = list(
            sorted(words, key=lambda word: len(word), reverse=True)
        )

    def tokenize(self, text: str) -> Iterator[tuple[str, int, int]]:
        position = 0
        while position < len(text):
            for word in self.words:
                if text.startswith(word, position):
                    yield word, position, position + len(word)
                    position += len(word)
                    break
            else:
                yield text[position], position, position + 1
                position += 1

    def lcut(self, text: str) -> list[str]:
        return [word for word, _, _ in self.tokenize(text)]


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small tokenizer and an empty form table in the tests."""
    words = ["我哋", "你哋", "佢哋", "食飯", "今日", "龍", "茶", "好", "食", "我"]
    monkeypatch.setattr(cantonese.profile, "tokenizer", lambda: FakeTokenizer(words))
    monkeypatch.setattr(
        cantonese.profile, "table_forms", lambda: frozenset({"我", "我哋"})
    )
    cantonese.han_spans.cache_clear()


def _write_tsv(path: Path, rows: list[tuple[object, ...]]) -> Path:
    text = "\n".join("\t".join(str(cell) for cell in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_reviewed_table_has_207_rows() -> None:
    table = cantonese.load_word_table()

    assert len(table) == 207
    assert len({form.cantonese for form in table}) == 207
    assert all(form.jyutping for form in table)


def test_sew_form_is_the_verb_to_sew() -> None:
    table = cantonese.load_word_table()
    rows = {form.swadesh_index: form for form in table}

    assert rows[138].english == "to sew"
    assert rows[138].cantonese == "縫"
    assert rows[138].jyutping == "fung4"


def _module_terms() -> dict[int, tuple[str, ...]]:
    return {
        form.swadesh_index: (form.cantonese,) for form in cantonese.load_word_table()
    }


def test_check_word_table_accepts_the_reviewed_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cantonese.profile, "module_terms", _module_terms)
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cantonese.WORD_REVIEW_FIELDS)
        writer.writeheader()
        for row in cantonese.read_word_review():
            writer.writerow(row)

    assert (
        cantonese.check_word_table(cantonese.load_word_table(), review_path=review)
        == []
    )


def test_check_word_table_marks_an_open_review_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cantonese.profile, "module_terms", _module_terms)
    review = tmp_path / "review.csv"
    review.write_text(
        "index,english,current,module_terms,chosen,issue,fix,status\n"
        "8,that,嗰個,嗰,,open question,,open\n",
        encoding="utf-8",
    )

    failures = cantonese.check_word_table(
        cantonese.load_word_table(), review_path=review
    )

    assert any("open row" in failure for failure in failures)


def test_prepare_text_converts_simplified_to_traditional() -> None:
    assert cantonese.prepare_text("我哋今日去咗街度食饭") == "我哋今日去咗街度食飯"


def test_jyutping_of_a_sentence() -> None:
    assert (
        cantonese.sentence_jyutping("我哋今日去咗街度食飯。")
        == "ngo5 dei6 gam1 jat6 heoi3 zo2 gaai1 dou6 sik6 faan6"
    )


def test_pronoun_pair_does_not_hit_inside_the_longer_word(
    fake_tokenizer: None,
) -> None:
    assert cantonese.form_offsets("我哋去街。", "我") == []
    assert cantonese.form_offsets("我去街。", "我") == [0]
    assert cantonese.form_offsets("我哋去街。", "我哋") == [0]


def test_verb_hits_inside_a_non_target_word(fake_tokenizer: None) -> None:
    assert cantonese.form_offsets("我食飯。", "食") == [1]
    assert cantonese.form_offsets("我食飯。", "食飯") == [1]


def test_false_friend_compound_does_not_hit(fake_tokenizer: None) -> None:
    assert cantonese.form_offsets("聽日係我生日。", "聽") == []
    assert cantonese.form_offsets("我聽音樂。", "聽") == [1]
    assert cantonese.form_offsets("一切都好。", "切") == []
    assert cantonese.form_offsets("我切菜。", "切") == [1]
    assert cantonese.form_offsets("佢靠彩數成功。", "數") == []
    assert cantonese.form_offsets("由左數起。", "數") == [2]
    assert cantonese.form_offsets("我發燒。", "燒") == []
    assert cantonese.form_offsets("我燒水。", "燒") == [1]


def test_sense_filter_rejects_false_friend_english() -> None:
    assert cantonese.sense_filter("揸", "Could you drive more slowly?")
    assert not cantonese.sense_filter("揸", "Hold the cup.")
    assert cantonese.sense_filter("燒", "Does she have a temperature?")
    assert not cantonese.sense_filter("燒", "My hand was burned.")
    assert cantonese.sense_filter("數", "I like math least of all.")
    assert not cantonese.sense_filter("數", "I counted to ten.")


def test_single_hits_skips_a_repeated_form(fake_tokenizer: None) -> None:
    forms = frozenset({"茶", "好"})

    assert cantonese.single_hits("我飲茶。", forms) == {"茶": 1}
    assert cantonese.single_hits("茶同茶。", forms) == {}


def test_unknown_token_count(fake_tokenizer: None) -> None:
    known = {"龍", "茶", "好"}

    assert cantonese.count_unknown("龍好。", known) == 0
    assert cantonese.count_unknown("龍茶好。", {"龍"}) == 2


def test_han_length_window_is_three_to_twelve() -> None:
    assert check_sentence_shape("我食飯。", cantonese.CANTONESE, 3, 12) is None
    assert check_sentence_shape("食。", cantonese.CANTONESE, 3, 12) == "F2"


def test_script_final_mark_and_quote_rules() -> None:
    assert check_sentence_shape("我食飯Tom。", cantonese.CANTONESE, 3, 12) == "F3"
    assert check_sentence_shape("我食飯", cantonese.CANTONESE, 3, 12) == "F5"
    assert check_sentence_shape("我食飯。好。", cantonese.CANTONESE, 3, 12) == "F6"
    assert check_sentence_shape("我食「飯」。", cantonese.CANTONESE, 3, 12) == "F7"
    assert check_sentence_shape("我，食飯：好。", cantonese.CANTONESE, 3, 12) == "F8"


def test_predicate_and_orphan_filters_are_off() -> None:
    assert cantonese.has_predicate("我食飯。")
    assert not cantonese.has_orphan_conjunction("因為我食飯。")


def test_highlight_wraps_the_whole_token(fake_tokenizer: None) -> None:
    assert cantonese.highlight("我哋同我去街。", "我") == "我哋同<b>我</b>去街。"
    assert cantonese.highlight("我食飯。", "食") == "我<b>食</b>飯。"


def test_gloss_and_card_fields() -> None:
    form = cantonese.WordForm(
        swadesh_index=1,
        english="I",
        cantonese="我",
        jyutping="ngo5",
        variant="primary",
        form_kind="single",
        zipf=6.9,
    )

    assert cantonese.glossary(form) == "I · ngo5"
    assert cantonese.card_fields({"jyutping": "ngo5 dei6"}, form) == {
        "Jyutping": "ngo5 dei6"
    }


def _pool_dir(tmp_path: Path) -> Path:
    data = tmp_path / "tatoeba"
    data.mkdir()
    sentences = [
        (1, "我食飯。"),
        (2, "我食飯"),
        (3, "我食飯。好。"),
        (4, "我食「飯」。"),
        (5, "我食Tom。"),
        (6, "龍食飯。"),
    ]
    _write_tsv(data / "yue_sentences.tsv", [(i, "yue", t) for i, t in sentences])
    _write_tsv(data / "yue-eng_links.tsv", [(i, 100 + i) for i, _ in sentences])
    _write_tsv(
        data / "eng_sentences.tsv",
        [(100 + i, "eng", "I eat rice.") for i, _ in sentences],
    )
    _write_tsv(
        data / "yue_sentences_detailed.tsv",
        [(i, "yue", t, "alice", "\\N", "2026-10-18") for i, t in sentences],
    )
    _write_tsv(data / "yue_sentences_with_audio.tsv", [])
    return data


@pytest.mark.usefixtures("fake_tokenizer")
def test_filter_set_f2_to_f12(tmp_path: Path) -> None:
    table = [
        cantonese.WordForm(
            swadesh_index=1,
            english="I",
            cantonese="我",
            jyutping="ngo5",
            variant="primary",
            form_kind="single",
            zipf=6.9,
        )
    ]
    data = _pool_dir(tmp_path)

    candidates, report = pool.build_candidate_pool(
        table,
        cantonese.CANTONESE,
        data_dir=data,
        known_words={"我", "食", "飯", "好", "龍"},
        make_candidate=cantonese.make_candidate,
    )
    texts = {candidate.sentence for candidate in candidates}

    assert texts == {"我食飯。"}
    assert report["counts"]["F2"] == 5
    assert report["counts"]["F3"] == 5
    assert report["counts"]["F5"] == 4
    assert report["counts"]["F6"] == 3
    assert report["counts"]["F7"] == 2
    for candidate in candidates:
        assert candidate.target == "我"


@pytest.mark.usefixtures("fake_tokenizer")
def test_candidate_holds_the_rank_key(tmp_path: Path) -> None:
    table = [
        cantonese.WordForm(
            swadesh_index=1,
            english="I",
            cantonese="我",
            jyutping="ngo5",
            variant="primary",
            form_kind="single",
            zipf=6.9,
        )
    ]
    data = _pool_dir(tmp_path)

    candidates, _ = pool.build_candidate_pool(
        table,
        cantonese.CANTONESE,
        data_dir=data,
        known_words={"我", "食", "飯", "好", "龍"},
        make_candidate=cantonese.make_candidate,
    )

    assert candidates
    assert len(candidates[0].rank_key.split("|")) == 5
    assert candidates[0].han_len == 3
