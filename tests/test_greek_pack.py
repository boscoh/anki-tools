"""Tests for the Greek Swadesh sentence pack (S1 to S5 and the gates)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.sentence_pack import audio, pack, pool, select
from anki_tools.sentence_pack.audit import AuditInputs, gate_target_in_table
from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.languages import greek


def _form(
    index: int,
    english: str,
    form: str,
    gender: str = "",
    variant: str = "primary",
) -> greek.WordForm:
    return greek.WordForm(
        swadesh_index=index,
        english=english,
        greek=form,
        gender=gender,
        variant=variant,
        form_kind=greek.form_kind(form),
        zipf=5.0,
    )


def _write_tsv(path: Path, rows: list[tuple[object, ...]]) -> Path:
    text = "\n".join("\t".join(str(cell) for cell in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _audit_inputs(
    tmp_path: Path, table: list[greek.WordForm], sentence: str
) -> AuditInputs:
    """Build audit inputs with a word table and one card row."""
    sentences = tmp_path / "sentences.csv"
    sentences.write_text(
        "target,sentence,english\nπόδι," + sentence + ",A leg.\n", encoding="utf-8"
    )
    review = tmp_path / "review.csv"
    review.write_text(
        "index,english,current,module_terms,chosen,issue,fix,status\n",
        encoding="utf-8",
    )
    return AuditInputs(
        greek.GREEK,
        apkg_path=tmp_path / "pack.apkg",
        sentences_csv=sentences,
        word_csv=tmp_path / "words.csv",
        review_csv=review,
        audio_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        attribution_path=tmp_path / "ATTRIBUTION.md",
        load_table=lambda **_: table,
    )


def test_note_removal_and_article_strip() -> None:
    assert greek.strip_note("εσύ (ενικός)") == "εσύ"
    assert greek.clean_form("το λίπος (ουσιαστικό)") == ("λίπος", "n")
    assert greek.clean_form("η γυναίκα") == ("γυναίκα", "f")
    assert greek.clean_form("ο άντρας") == ("άντρας", "m")


def test_one_word_form_keeps_its_article() -> None:
    assert greek.strip_article("ένας") == ("ένας", "")
    assert greek.clean_form("ένας") == ("ένας", "")


def test_reviewed_table_has_207_rows_and_206_forms() -> None:
    table = greek.load_word_table()

    assert len(table) == 207
    assert len({form.greek for form in table}) == 206
    assert greek.check_word_table(table) == []


def test_shared_foot_and_leg_row() -> None:
    table = greek.load_word_table()
    rows = {form.swadesh_index: form for form in table}

    assert rows[80].greek == "πόδι"
    assert rows[80].variant == "primary"
    assert rows[81].greek == "πόδι"
    assert rows[81].variant == "shared"
    targets = {form.greek for form in table} - {"πόδι"}
    assert greek.greek_excluded_words(table, targets) == ["πόδι"]


def test_gate_19_accepts_the_primary_foot_row_and_rejects_a_shared_only_target(
    tmp_path: Path,
) -> None:
    table = [
        _form(80, "foot", "πόδι", "n"),
        _form(81, "leg", "πόδι", "n", "shared"),
        _form(82, "knee", "γόνατο", "n"),
    ]
    inputs = _audit_inputs(tmp_path, table, "Θέλω το πόδι.")
    passed, message = gate_target_in_table(inputs, greek.GREEK)
    assert passed, message

    shared_only = [_form(81, "leg", "πόδι", "n", "shared")]
    inputs = _audit_inputs(tmp_path, shared_only, "Θέλω το πόδι.")
    passed, message = gate_target_in_table(inputs, greek.GREEK)
    assert not passed
    assert "shared form" in message


def test_final_sigma_and_surface_key() -> None:
    assert greek.surface_key("ΝΕΡΌΣ") == greek.surface_key("νερός")
    assert greek.script_problem("νερός") is None
    assert "final sigma" in (greek.script_problem("αςα") or "")
    assert greek.script_problem("νερό water") is not None


def test_semicolon_is_the_question_mark() -> None:
    assert check_sentence_shape("Θέλω να πίνω νερό;", greek.GREEK, 4, 10) is None
    two = "Θέλω να πίνω νερό; Ναι, θέλω;"
    assert check_sentence_shape(two, greek.GREEK, 4, 10) == "F6"


def test_ano_teleia_is_an_internal_mark() -> None:
    ok = "Θέλω νερό· γάλα και ψωμί."
    too_many = "Θέλω νερό· γάλα· ψωμί."
    assert check_sentence_shape(ok, greek.GREEK, 4, 10) is None
    assert check_sentence_shape(too_many, greek.GREEK, 4, 10) == "F8"


def test_lemma_match_for_pino() -> None:
    sentence = "Θέλω να πίνει νερό."

    assert greek.token_matches(sentence, "πίνω") == 1
    assert greek.hits(sentence, frozenset({"πίνω"})) == {"πίνω": 1}


def test_target_inside_a_longer_word_does_not_match() -> None:
    assert greek.token_matches("Το νερό είναι κρύο.", "νερό") == 1
    assert greek.token_matches("Ο νερόλακκος είναι βαθύς.", "νερό") == 0
    assert greek.token_matches("Τα νερά είναι κρύα.", "νερό") == 1


def test_sense_filter_rejects_false_friend_english() -> None:
    assert greek.sense_filter("γλώσσα", "I love this language.")
    assert not greek.sense_filter("γλώσσα", "I bit my tongue.")
    assert greek.sense_filter("νέος", "My father is young.")
    assert not greek.sense_filter("νέος", "This is a new car.")
    assert greek.sense_filter("χρόνος", "I don't have time.")
    assert not greek.sense_filter("χρόνος", "It is three years.")


def test_unknown_word_filter_with_the_top_1000_list() -> None:
    table = greek.load_word_table()
    known = greek.known_words(table)
    sentence = "Θέλω να πίνω νερό."

    assert greek.count_unknown(sentence, known) == 0
    assert greek.count_unknown(sentence, {"νερό"}) >= 1


def _pool_dir(tmp_path: Path) -> Path:
    data = tmp_path / "tatoeba"
    data.mkdir()
    sentences = [
        (1, "Θέλω να πίνω νερό."),
        (2, "Θέλω νερό;"),
        (3, "Θέλω να πίνω νερό; Ναι;"),
        (4, "Θέλω νερό, γάλα, ψωμί."),
        (5, "Θέλω «νερό»."),
        (6, "Θέλω water."),
        (7, "Και θέλω νερό."),
    ]
    _write_tsv(data / "ell_sentences.tsv", [(i, "ell", t) for i, t in sentences])
    _write_tsv(
        data / "ell-eng_links.tsv",
        [(i, 100 + i) for i, _ in sentences],
    )
    _write_tsv(
        data / "eng_sentences.tsv",
        [(100 + i, "eng", "I want to drink water.") for i, _ in sentences],
    )
    _write_tsv(
        data / "ell_sentences_detailed.tsv",
        [(i, "ell", t, "alice", "\\N", "2026-10-18") for i, t in sentences],
    )
    return data


def test_greek_filter_set_f2_to_f12(tmp_path: Path) -> None:
    table = [_form(1, "water", "νερό", "n"), _form(2, "to drink", "πίνω")]
    data = _pool_dir(tmp_path)
    known = {"Θέλω", "να", "πίνω", "νερό", "γάλα", "ψωμί", "Ναι", "Και", "είναι"}

    candidates, report = pool.build_candidate_pool(
        table,
        greek.GREEK,
        data_dir=data,
        known_words=known,
        make_candidate=greek.make_candidate,
    )
    counts = report["counts"]
    texts = {candidate.sentence for candidate in candidates}

    assert texts == {"Θέλω να πίνω νερό."}
    assert counts["F1"] == 7
    assert counts["F2"] < counts["F1"]
    assert counts["F12"] == 1
    assert counts["unknown"] == 1
    assert check_sentence_shape("Θέλω νερό;", greek.GREEK, 4, 10) == "F2"
    assert check_sentence_shape("Θέλω να πίνω water.", greek.GREEK, 4, 10) == "F3"
    assert check_sentence_shape("Θέλω «νερό» και γάλα.", greek.GREEK, 4, 10) == "F7"
    assert check_sentence_shape("Θέλω νερό· γάλα· ψωμί.", greek.GREEK, 4, 10) == "F8"
    assert greek.has_orphan("Και θέλω νερό.")
    for candidate in candidates:
        assert check_sentence_shape(
            candidate.sentence, greek.GREEK, 4, 10
        ) is None


def test_set_cover_on_a_small_fixture() -> None:
    table = [_form(1, "water", "νερό", "n"), _form(2, "to drink", "πίνω")]
    cards = [
        greek.make_candidate(
            pool.SentenceFacts(
                sentence="Θέλω να πίνω νερό.",
                text_original="Θέλω να πίνω νερό.",
                tatoeba_id=1,
                tatoeba_author="alice",
                english="I want to drink water.",
                english_id=101,
                audio_id=0,
                audio_license="",
                unknown_words=0,
                unit_len=5,
                zipf_mean=5.0,
                hits=("νερό", "πίνω"),
            ),
            "νερό",
            "core",
            frozenset({"νερό", "πίνω"}),
        ),
        greek.make_candidate(
            pool.SentenceFacts(
                sentence="Θέλω νερό και ψωμί.",
                text_original="Θέλω νερό και ψωμί.",
                tatoeba_id=2,
                tatoeba_author="bob",
                english="I want water and bread.",
                english_id=102,
                audio_id=0,
                audio_license="",
                unknown_words=0,
                unit_len=4,
                zipf_mean=5.0,
                hits=("νερό",),
            ),
            "νερό",
            "core",
            frozenset({"νερό"}),
        ),
    ]

    selected, report = select.select_sentences(
        cards, table, greek.GREEK, per_word=1, max_cards=10
    )

    assert len(selected) == 1
    assert selected[0].target == "νερό"
    assert selected[0].tatoeba_id == 1
    assert report["cards"] == 1


def test_audio_file_names_are_deterministic() -> None:
    first = audio.audio_hash("Θέλω νερό.", "gtts:el", "-10%")
    second = audio.audio_hash("Θέλω νερό.", "gtts:el", "-10%")

    assert first == second
    assert first != audio.audio_hash("Θέλω νερό.", "gtts:en", "-10%")
    assert audio.sentence_audio_name("Θέλω νερό.", "gtts:el", "-10%", "tts", 0) == (
        f"tts_{first}.mp3"
    )
    assert audio.word_audio_name("νερό", "gtts:el", "-10%").startswith("word_")


def test_manifest_counts(tmp_path: Path) -> None:
    table = [
        _form(1, "water", "νερό", "n"),
        _form(2, "to drink", "πίνω"),
        _form(81, "leg", "πόδι", "n", "shared"),
    ]
    rows = [
        {
            "target": "νερό",
            "sentence": "Θέλω νερό.",
            "english": "I want water.",
            "tatoeba_id": "1",
            "tatoeba_author": "alice",
            "audio_source": "tts",
            "tier": "core",
        }
    ]
    word_csv = tmp_path / "words.csv"
    word_csv.write_text("english,greek,gender\nwater,νερό,n\n", encoding="utf-8")
    csv_path = tmp_path / "sentences.csv"
    csv_path.write_text("target,sentence\nνερό,Θέλω νερό.\n", encoding="utf-8")
    index = pack.AudioIndex(sentences={"Θέλω νερό.": "tts_x.mp3"}, words={}, voice="gtts:el")

    manifest = pack.build_manifest(
        greek.GREEK, table, rows, index, word_csv=word_csv, csv_path=csv_path
    )

    assert manifest["counts"]["words"] == 3
    assert manifest["counts"]["cards"] == 1
    assert manifest["counts"]["tts_audio"] == 1
    assert manifest["counts"]["native_audio"] == 0
    assert manifest["coverage"]["core_with_card"] == round(1 / 207, 4)
    assert manifest["voice"] == "gtts:el"
    assert manifest["native_audio_available"] is False
    assert manifest["excluded_words"] == ["πίνω", "πόδι"]
    assert manifest["deck"] == greek.GREEK.deck_name
    assert manifest["model"] == greek.GREEK.model_name


def test_card_cap_holds_the_greek_deck_small() -> None:
    table = [_form(1, "water", "νερό", "n")]
    facts = pool.SentenceFacts(
        sentence="Θέλω να πίνω νερό.",
        text_original="Θέλω να πίνω νερό.",
        tatoeba_id=1,
        tatoeba_author="alice",
        english="I want to drink water.",
        english_id=101,
        audio_id=0,
        audio_license="",
        unknown_words=0,
        unit_len=5,
        zipf_mean=5.0,
        hits=("νερό",),
    )
    sentences = [
        "Θέλω να πίνω νερό.",
        "Θέλω νερό και ψωμί.",
        "Δώσε μου λίγο νερό.",
        "Πίνω νερό κάθε πρωί.",
        "Το νερό είναι κρύο.",
    ]
    cards = [
        greek.make_candidate(
            replace(facts, tatoeba_id=number, sentence=sentence),
            "νερό",
            "core",
            frozenset({"νερό"}),
        )
        for number, sentence in enumerate(sentences, start=1)
    ]

    selected, report = select.select_sentences(
        cards, table, greek.GREEK, per_word=5, max_cards=3
    )

    assert len(selected) == 3
    assert report["max_cards"] == 3


def test_greek_never_uses_the_native_audio_path() -> None:
    row = {"audio_id": "123", "audio_license": "CC BY 2.0 FR"}

    with pytest.raises(ValueError):
        audio.choose_source(row, "auto", "public", greek.GREEK)
    with pytest.raises(ValueError):
        audio.choose_source(row, "only", "public", greek.GREEK)
    assert audio.choose_source(row, "none", "public", greek.GREEK) == "tts"
    assert greek.GREEK.native_audio is False


def test_sentence_csv_columns() -> None:
    with greek.SENTENCES_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert header == greek.SENTENCE_FIELDS
