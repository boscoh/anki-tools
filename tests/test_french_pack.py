"""Tests for the French Swadesh sentence pack (S1 to S5 and the gates)."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from anki_tools.sentence_pack import audio, pack, pool, select
from anki_tools.sentence_pack.audit import AuditInputs, gate_proper_sentences
from anki_tools.sentence_pack.pool import check_sentence_shape
from anki_tools.languages.french import profile as french

MANDARIN_SHA256 = "862e07a6b5c7baa2766b9729734ddc72821496e89cdb76e3156f39245157b581"
TAB = "\t"
NOUN_ROWS = 84


def _form(
    index: int,
    english: str,
    form: str,
    gender: str = "",
    variant: str = "primary",
) -> french.WordForm:
    return french.WordForm(
        swadesh_index=index,
        english=english,
        french=form,
        gender=gender,
        variant=variant,
        form_kind=french.form_kind(form),
        zipf=5.0,
    )


def _write_tsv(path: Path, rows: list[tuple[object, ...]]) -> Path:
    text = "\n".join(TAB.join(str(cell) for cell in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _audit_inputs(
    tmp_path: Path,
    table: list[french.WordForm],
    rows: list[dict[str, str]],
    manifest: dict[str, object] | None = None,
) -> AuditInputs:
    sentences = tmp_path / "sentences.csv"
    lines = ["target,sentence,english,tier,audio_source,audio_license,tatoeba_author"]
    for row in rows:
        lines.append(
            ",".join(
                row.get(field, "")
                for field in (
                    "target",
                    "sentence",
                    "english",
                    "tier",
                    "audio_source",
                    "audio_license",
                    "tatoeba_author",
                )
            )
        )
    sentences.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = tmp_path / "review.csv"
    review.write_text(
        "index,english,current,module_terms,chosen,issue,fix,status\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest or {}), encoding="utf-8")
    return AuditInputs(
        french.FRENCH,
        apkg_path=tmp_path / "pack.apkg",
        sentences_csv=sentences,
        word_csv=tmp_path / "words.csv",
        review_csv=review,
        audio_dir=tmp_path,
        manifest_path=manifest_path,
        attribution_path=tmp_path / "ATTRIBUTION.md",
        load_table=lambda **_: table,
    )


def test_reviewed_table_has_207_rows_and_207_forms() -> None:
    table = french.load_word_table()

    assert len(table) == 207
    assert len({form.french for form in table}) == 207
    assert french.check_word_table(table) == []


def test_review_file_holds_the_nine_reviewed_rows() -> None:
    review = french.read_word_review()
    reviewed = {
        int(row["index"]): row
        for row in review
        if row["issue"] in ("shared_form", "duplicate_form", "two_terms")
    }

    assert sorted(reviewed) == [2, 36, 37, 38, 40, 41, 86, 179, 183]
    assert reviewed[2]["chosen"] == "tu"
    assert reviewed[38]["chosen"] == "être humain"
    assert reviewed[40]["chosen"] == "épouse"
    assert reviewed[41]["chosen"] == "mari"
    assert reviewed[86]["chosen"] == "intestins"
    assert reviewed[179]["chosen"] == "année"
    assert reviewed[183]["chosen"] == "nouveau"
    assert all(row["status"] != "open" for row in review)
    gender_rows = [row for row in review if row["issue"] == "gender"]
    assert len(gender_rows) == NOUN_ROWS
    assert len({row["index"] for row in gender_rows}) == NOUN_ROWS
    zipf_rows = [row for row in review if row["issue"] == "zipf"]
    assert [row["index"] for row in zipf_rows] == ["192"]
    assert zipf_rows[0]["status"] == "kept"
    assert len(review) == len(reviewed) + NOUN_ROWS + 1


def test_module_cleanup() -> None:
    assert french.clean_term("[[ne]] [[pas]]") == "ne pas"
    assert french.clean_term("se battre") == "se battre"
    assert french.clean_term("εσύ (ενικός)") == "εσύ"

    text = 'm[1] = {{term = "je"}}\nm[2] = {\n    {term = "tu"},\n    {term = "vous"}\n}\n'
    assert french.parse_module_terms(text) == {1: ("je",), 2: ("tu", "vous")}
    assert len(french.module_terms()) == 207


def test_form_kinds_and_multiword_forms() -> None:
    table = {form.swadesh_index: form for form in french.load_word_table()}

    assert table[16].french == "ne pas"
    assert table[16].form_kind == "discontinuous"
    assert table[111].french == "se battre"
    assert table[111].form_kind == "pair"
    assert table[125].french == "se lever"
    assert table[125].form_kind == "pair"
    assert table[206].french == "parce que"
    assert table[206].form_kind == "pair"
    assert table[150].form_kind == "single"


def test_whitespace_normalization() -> None:
    raw = "Bonjour\u202f! Ça va\u00a0? Oui\u202f!"

    assert french.prepare_text(raw) == "Bonjour ! Ça va ? Oui !"
    assert "\u202f" not in french.prepare_text(raw)
    assert "\u00a0" not in french.prepare_text(raw)
    assert french.prepare_text("Écoute\u00a0: le chien.") == "Écoute : le chien."
    assert french.unit_count(french.prepare_text("Ça va\u00a0?")) == 2


def test_form_key_rule_with_pluie_manger_and_elision() -> None:
    assert french.hits("La pluie tombe depuis hier.", frozenset({"pluie"})) == {
        "pluie": 1
    }
    assert "plui" in french.form_keys("pluie")
    assert french.hits("Je mange du pain.", frozenset({"manger"})) == {"manger": 1}
    assert french.token_matches("Elle a mangé du pain.", "manger") == 1
    assert french.hits("Je voudrais un verre d'eau.", frozenset({"eau"})) == {"eau": 1}
    assert french.token_matches("La voiture est en panne.", "eau") == 0


def test_sense_filter_rejects_false_friend_english() -> None:
    assert french.sense_filter("queue", "I didn't wait in line.")
    assert not french.sense_filter("queue", "The dog wagged its tail.")
    assert french.sense_filter("sentir", "I feel fine.")
    assert not french.sense_filter("sentir", "I smell smoke.")
    assert french.sense_filter("voler", "I've been robbed.")
    assert not french.sense_filter("voler", "The birds were flying.")


def test_form_key_rule_for_pairs_and_elided_clitics() -> None:
    assert french.token_matches("Ils se battent dans la rue.", "se battre") == 1
    assert french.token_matches("Il aime battre le tambour.", "se battre") == 0
    assert french.token_matches("Je ne veux pas partir.", "ne pas") == 1
    assert french.token_matches("Il n'est pas venu.", "ne pas") == 1
    assert french.token_matches("Je viens parce qu'il est là.", "parce que") == 1
    assert french.token_matches("Je viens.", "parce que") == 0


def test_highlight_marks_the_matched_token() -> None:
    assert french.highlight("Je voudrais un verre d'eau.", "eau") == (
        "Je voudrais un verre d'<b>eau</b>."
    )
    assert french.highlight("Je mange du pain.", "manger") == (
        "Je <b>mange</b> du pain."
    )
    assert french.highlight("Ils se battent ici.", "se battre") == (
        "Ils <b>se battent</b> ici."
    )


def test_unknown_word_filter_with_lemma_matching() -> None:
    table = french.load_word_table()
    known = french.known_words(table)
    sentence = "Je voudrais un verre d'eau."

    assert french.count_unknown(sentence, known) == 2
    assert french.count_unknown(sentence, {"eau"}) >= 1
    assert french.count_unknown("Les oiseaux mangent des graines.", known) == 0
    assert french.count_unknown(
        "Elle a mangé du pain.", {"elle", "a", "du", "pain", "manger"}
    ) == 0
    assert french.count_unknown("Elle a mangé du pain.", {"elle", "a", "du", "pain"}) == 1


def test_proper_sentence_filters_f2_to_f10() -> None:
    assert check_sentence_shape("Le chien.", french.FRENCH, 4, 10) == "F2"
    assert check_sentence_shape("Le chien mange vite.", french.FRENCH, 4, 10) is None
    assert (
        check_sentence_shape(
            french.prepare_text("Le chien mange vite\u00a0!"), french.FRENCH, 4, 10
        )
        is None
    )
    assert (
        check_sentence_shape("Le chien mange 42 fois ici.", french.FRENCH, 4, 10)
        == "F3"
    )
    assert check_sentence_shape("Oui oui oui oui.", french.FRENCH, 4, 10) == "F4"
    assert check_sentence_shape("Le chien mange vite", french.FRENCH, 4, 10) == "F5"
    assert (
        check_sentence_shape("Le chien mange. Il aboie.", french.FRENCH, 4, 10)
        == "F6"
    )
    assert (
        check_sentence_shape('Il dit "oui" au chien.', french.FRENCH, 4, 10) == "F7"
    )
    assert (
        check_sentence_shape("Le chien, le chat, l'oiseau.", french.FRENCH, 4, 10)
        == "F8"
    )
    assert french.has_predicate("Le chien mange vite.") is True
    assert french.has_predicate("Trois chiens.") is False
    assert french.has_predicate("Il y a un chien.") is True
    assert french.has_orphan("Et il y a un chien.") is True
    assert french.has_orphan("Et il y a un chien, dit-il.") is False


def _pool_dir(tmp_path: Path) -> Path:
    data = tmp_path / "tatoeba"
    data.mkdir()
    sentences = [
        (1, "Je voudrais un verre d'eau."),
        (2, "Le chien."),
        (3, "Le chien mange vite, il aboie."),
        (4, "Le chien, le chat, l'oiseau."),
        (5, "Il dit « oui » au chien."),
        (6, "Le chien mange 42 fois ici."),
        (7, "Et le chien mange vite."),
    ]
    _write_tsv(data / "fra_sentences.tsv", [(i, "fra", text) for i, text in sentences])
    _write_tsv(data / "fra-eng_links.tsv", [(i, 100 + i) for i, _ in sentences])
    _write_tsv(
        data / "eng_sentences.tsv",
        [(100 + i, "eng", "The dog runs to the water.") for i, _ in sentences],
    )
    _write_tsv(
        data / "fra_sentences_detailed.tsv",
        [(i, "fra", text, "alice", "\\N", "2026-10-19") for i, text in sentences],
    )
    _write_tsv(
        data / "fra_sentences_with_audio.tsv", [(1, 9001, "bob", "CC BY-SA 4.0")]
    )
    return data


def test_pool_uses_the_french_filters(tmp_path: Path) -> None:
    table = [_form(150, "water", "eau", "f"), _form(47, "dog", "chien", "m")]
    data = _pool_dir(tmp_path)
    known = {
        "je",
        "voudrais",
        "un",
        "verre",
        "de",
        "le",
        "chien",
        "mange",
        "vite",
        "il",
        "aboie",
        "eau",
    }

    candidates, report = pool.build_candidate_pool(
        table,
        french.FRENCH,
        data_dir=data,
        known_words=known,
        make_candidate=french.make_candidate,
    )
    counts = report["counts"]
    texts = {candidate.sentence for candidate in candidates}

    assert texts == {"Je voudrais un verre d'eau.", "Le chien mange vite, il aboie."}
    assert counts["F1"] == 7
    assert counts["F2"] == 6
    assert counts["F3"] == 5
    assert counts["F7"] == 4
    assert counts["F8"] == 3
    assert counts["F10"] == 2
    assert counts["F11"] == 2
    assert counts["F12"] == 2
    assert report["forms_with_core_candidate"] == 2
    assert report["pool_candidates"] == 2


def test_set_cover_on_a_small_fixture() -> None:
    table = [_form(1, "water", "eau", "f"), _form(2, "to drink", "boire")]
    facts = pool.SentenceFacts(
        sentence="Je voudrais un verre d'eau.",
        text_original="Je voudrais un verre d'eau.",
        tatoeba_id=1,
        tatoeba_author="alice",
        english="I would like a glass of water.",
        english_id=101,
        audio_id=0,
        audio_license="",
        unknown_words=0,
        unit_len=6,
        zipf_mean=5.0,
        hits=("eau",),
    )
    cards = [
        french.make_candidate(facts, "eau", "core", frozenset({"eau"})),
        french.make_candidate(
            replace(facts, tatoeba_id=2, sentence="Il boit de l'eau."),
            "eau",
            "core",
            frozenset({"eau"}),
        ),
    ]

    selected, report = select.select_sentences(
        cards, table, french.FRENCH, per_word=1, max_cards=10
    )

    assert len(selected) == 1
    assert selected[0].target == "eau"
    assert selected[0].tatoeba_id == 1
    assert report["cards"] == 1
    assert report["forms"]["with_card"] == 1


def test_audio_file_names_are_deterministic() -> None:
    first = audio.audio_hash("Je voudrais un verre d'eau.", "gtts:fr", "-10%")
    second = audio.audio_hash("Je voudrais un verre d'eau.", "gtts:fr", "-10%")

    assert first == second
    assert first != audio.audio_hash("Je voudrais un verre d'eau.", "gtts:en", "-10%")
    assert audio.sentence_audio_name(
        "Je voudrais un verre d'eau.", "gtts:fr", "-10%", "tts", 0
    ) == (f"tts_{first}.mp3")
    assert audio.word_audio_name("eau", "gtts:fr", "-10%").startswith("word_")
    assert audio.choose_source({}, "none", "public", french.FRENCH) == "tts"
    assert (
        audio.choose_source(
            {"audio_id": "9", "audio_license": "CC BY-NC 4.0"},
            "auto",
            "public",
            french.FRENCH,
        )
        == "tts"
    )


def test_scaffold_cards_take_the_g14_exemption(tmp_path: Path) -> None:
    table = [_form(47, "dog", "chien", "m")]
    rows = [
        {
            "target": "chien",
            "sentence": "Le chien mange le chat ? Il mange ?",
            "english": "The dog eats the cat? He eats?",
            "tier": "scaffold",
            "audio_source": "tts",
            "tatoeba_author": "alice",
        }
    ]
    inputs = _audit_inputs(tmp_path, table, rows)
    passed, message = gate_proper_sentences(inputs, french.FRENCH)
    assert passed, message

    core = [dict(rows[0], tier="core")]
    inputs = _audit_inputs(tmp_path, table, core)
    passed, message = gate_proper_sentences(inputs, french.FRENCH)
    assert not passed
    assert "F6" in message


def test_gate_19_rejects_a_discontinuous_target_without_both_parts(
    tmp_path: Path,
) -> None:
    table = [_form(16, "not", "ne pas")]
    ok = [
        {
            "target": "ne pas",
            "sentence": "Je ne veux pas partir.",
            "english": "I do not want to leave.",
            "tier": "core",
        }
    ]
    inputs = _audit_inputs(tmp_path, table, ok)
    passed, message = french.gate_discontinuous(inputs, french.FRENCH)
    assert passed, message

    bad = [dict(ok[0], sentence="Je ne veux partir.")]
    inputs = _audit_inputs(tmp_path, table, bad)
    passed, _ = french.gate_discontinuous(inputs, french.FRENCH)
    assert not passed


def test_gate_20_reports_the_noun_gender_count(tmp_path: Path) -> None:
    table = [_form(150, "water", "eau", "f"), _form(151, "rain", "pluie")]
    rows = [
        {
            "target": "eau",
            "sentence": "Je voudrais de l'eau.",
            "english": "I would like water.",
            "tier": "core",
        }
    ]
    inputs = _audit_inputs(tmp_path, table, rows, {"noun_gender_missing": 1})
    passed, message = french.gate_noun_gender(inputs, french.FRENCH)
    assert passed
    assert "pluie" in message

    full = [_form(150, "water", "eau", "f"), _form(151, "rain", "pluie", "f")]
    inputs = _audit_inputs(tmp_path, full, rows, {"noun_gender_missing": 0})
    passed, message = french.gate_noun_gender(inputs, french.FRENCH)
    assert passed
    assert "2 noun row(s)" in message


def test_manifest_counts(tmp_path: Path) -> None:
    table = [
        _form(1, "I", "je"),
        _form(150, "water", "eau", "f"),
        _form(151, "rain", "pluie", "f"),
    ]
    rows = [
        {
            "target": "eau",
            "sentence": "Je voudrais de l'eau.",
            "english": "I would like water.",
            "tatoeba_id": "1",
            "tatoeba_author": "alice",
            "audio_source": "tts",
            "tier": "core",
        }
    ]
    word_csv = tmp_path / "words.csv"
    word_csv.write_text("english,french,gender\nwater,eau,f\n", encoding="utf-8")
    csv_path = tmp_path / "sentences.csv"
    csv_path.write_text("target,sentence\neau,Je voudrais de l'eau.\n", encoding="utf-8")
    index = pack.AudioIndex(
        sentences={"Je voudrais de l'eau.": "tts_x.mp3"}, words={}, voice="gtts:fr"
    )

    manifest = pack.build_manifest(
        french.FRENCH, table, rows, index, word_csv=word_csv, csv_path=csv_path
    )

    assert manifest["counts"]["words"] == 3
    assert manifest["counts"]["cards"] == 1
    assert manifest["counts"]["tts_audio"] == 1
    assert manifest["counts"]["native_audio"] == 0
    assert manifest["coverage"]["core_with_card"] == round(1 / 207, 4)
    assert manifest["voice"] == "gtts:fr"
    assert manifest["native_audio_available"] is True
    assert manifest["noun_rows"] == NOUN_ROWS
    assert manifest["noun_gender_missing"] == 0
    assert manifest["excluded_words"] == ["je", "pluie"]


def test_sentence_csv_columns() -> None:
    with french.SENTENCES_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert header == french.SENTENCE_FIELDS


def test_mandarin_sentences_csv_matches_the_snapshot() -> None:
    digest = hashlib.sha256(
        Path("vocab/mandarin_sentences.csv").read_bytes()
    ).hexdigest()

    assert digest == MANDARIN_SHA256
