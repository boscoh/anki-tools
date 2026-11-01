"""Tests for the French Swadesh sentence pack (S1 to S5 and the gates)."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from anki_tools.languages.french import profile as french
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack import audio, pack, pool, select
from anki_tools.sentence_pack.audit import AuditInputs, gate_filter_recheck
from anki_tools.sentence_pack.english_sense import sense_failure, sense_row
from tests.tatoeba_fixture import PoolRow, write_tatoeba_dir

AUDIT_FIELDS = (
    "target",
    "sentence",
    "english",
    "tier",
    "audio_source",
    "audio_license",
    "tatoeba_author",
)


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


def _nouns() -> list[french.WordForm]:
    return [
        form
        for form in french.load_word_table()
        if form.swadesh_index in french.NOUN_INDICES
    ]


def _audit_inputs(
    tmp_path: Path, table: list[french.WordForm], rows: list[dict[str, str]]
) -> AuditInputs:
    sentences = tmp_path / "sentences.csv"
    lines: list[str] = [",".join(AUDIT_FIELDS)]
    lines.extend(",".join(row.get(field, "") for field in AUDIT_FIELDS) for row in rows)
    sentences.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = tmp_path / "review.csv"
    review.write_text(
        "index,english,current,module_terms,chosen,issue,fix,status\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({}), encoding="utf-8")
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


def test_the_reviewed_table_passes_its_checker() -> None:
    table = french.load_word_table()

    assert len({form.french for form in table}) == len(table)
    assert french.check_word_table(table) == []


def test_the_review_file_is_closed_and_agrees_with_the_table() -> None:
    review = french.read_word_review()
    primary = {
        form.swadesh_index: form.french
        for form in french.load_word_table()
        if form.variant == "primary"
    }
    choices = [
        row
        for row in review
        if row["issue"] in ("shared_form", "duplicate_form", "two_terms")
    ]
    gender = [int(row["index"]) for row in review if row["issue"] == "gender"]

    assert all(row["status"] != "open" for row in review)
    assert choices
    assert all(primary[int(row["index"])] == row["chosen"] for row in choices)
    assert sorted(gender) == sorted({form.swadesh_index for form in _nouns()})


@pytest.mark.parametrize(
    ("sentence", "form", "expected"),
    [
        pytest.param(
            "Je voudrais un verre d'eau.",
            "eau",
            "Je voudrais un verre d'<b>eau</b>.",
            id="elision",
        ),
        pytest.param(
            "Je mange du pain.", "manger", "Je <b>mange</b> du pain.", id="inflection"
        ),
        pytest.param(
            "Ils se battent ici.", "se battre", "Ils <b>se battent</b> ici.", id="pair"
        ),
        pytest.param(
            "Je ne parle pas.", "ne", "Je <b>ne</b> parle <b>pas</b>.", id="negative"
        ),
    ],
)
def test_highlight_marks_the_matched_token(
    sentence: str, form: str, expected: str
) -> None:
    assert french.highlight(sentence, form) == expected


def test_module_cleanup() -> None:
    assert french.clean_term("[[ne]] [[pas]]") == "ne pas"
    assert french.clean_term("se battre") == "se battre"
    assert french.clean_term("εσύ (ενικός)") == "εσύ"

    text = (
        'm[1] = {{term = "je"}}\nm[2] = {\n    {term = "tu"},\n    {term = "vous"}\n}\n'
    )
    assert french.parse_module_terms(text) == {1: ("je",), 2: ("tu", "vous")}
    assert len(french.module_terms()) == len(french.load_word_table())


def test_whitespace_normalization() -> None:
    assert (
        french.prepare_text("Bonjour\u202f! Ça va\u00a0? Oui\u202f!")
        == "Bonjour ! Ça va ? Oui !"
    )
    assert french.prepare_text("Écoute\u00a0: le chien.") == "Écoute : le chien."
    assert french.unit_count(french.prepare_text("Ça va\u00a0?")) == 2


@pytest.mark.parametrize(
    ("sentence", "form", "expected"),
    [
        pytest.param("La pluie tombe depuis hier.", "pluie", 1, id="plain"),
        pytest.param("Elle a mangé du pain.", "manger", 1, id="inflected"),
        pytest.param("Je voudrais un verre d'eau.", "eau", 1, id="elided"),
        pytest.param("La voiture est en panne.", "eau", 0, id="absent"),
        pytest.param("Ils se battent dans la rue.", "se battre", 1, id="pair"),
        pytest.param(
            "Il aime battre le tambour.", "se battre", 0, id="pair needs the clitic"
        ),
        pytest.param(
            "Il se dit de battre le tambour.", "se battre", 0, id="pair is adjacent"
        ),
        pytest.param("Je ne veux pas partir.", "ne pas", 1, id="both parts"),
        pytest.param("Il n'est pas venu.", "ne pas", 1, id="elided ne"),
        pytest.param("Je viens parce qu'il est là.", "parce que", 1, id="elided que"),
        pytest.param("Je viens.", "parce que", 0, id="no match"),
    ],
)
def test_form_key_rule(sentence: str, form: str, expected: int) -> None:
    assert french.token_matches(sentence, form) == expected


@pytest.mark.parametrize(
    ("form", "english", "expected"),
    [
        ("queue", "I didn't wait in line.", True),
        ("queue", "The dog wagged its tail.", False),
        ("sentir", "I feel fine.", True),
        ("sentir", "I smell smoke.", False),
        ("voler", "I've been robbed.", True),
        ("voler", "The birds were flying.", False),
    ],
)
def test_sense_rows_reject_false_friend_english(
    form: str, english: str, expected: bool
) -> None:
    row = sense_row(french.SENSE_PATH, "S", form)

    assert (sense_failure(english, row, "") == "false_friend") == expected


def test_unknown_word_filter_with_lemma_matching() -> None:
    known = french.known_words(french.load_word_table())
    mange = {"elle", "a", "du", "pain"}

    assert french.count_unknown("Je voudrais un verre d'eau.", known) == 2
    assert french.count_unknown("Les oiseaux mangent des graines.", known) == 0
    assert french.count_unknown("Elle a mangé du pain.", mange | {"manger"}) == 0
    assert french.count_unknown("Elle a mangé du pain.", mange) == 1


def test_pool_uses_the_french_filters(tmp_path: Path) -> None:
    rows = [
        PoolRow(
            1,
            "Je voudrais un verre d'eau.",
            101,
            "The dog runs to the water.",
            audio=(9001, "bob", "CC BY-SA 4.0"),
        ),
        PoolRow(2, "Le chien.", 102, "The dog runs to the water."),
        PoolRow(3, "Le chien mange vite, il aboie.", 103, "The dog runs to the water."),
        PoolRow(4, "Le chien, le chat, l'oiseau.", 104, "The dog runs to the water."),
        PoolRow(5, "Il dit « oui » au chien.", 105, "The dog runs to the water."),
        PoolRow(6, "Le chien mange 42 fois ici.", 106, "The dog runs to the water."),
        PoolRow(7, "Et le chien mange vite.", 107, "The dog runs to the water."),
    ]
    table = [_form(150, "water", "eau", "f"), _form(47, "dog", "chien", "m")]
    known = set("je voudrais un verre de le chien mange vite il aboie eau".split())

    candidates, report = pool.build_candidate_pool(
        table,
        french.FRENCH,
        data_dir=write_tatoeba_dir(tmp_path / "tatoeba", "fra", rows),
        known_words=known,
        make_candidate=french.make_candidate,
    )
    counts = report["counts"]

    assert {candidate.sentence for candidate in candidates} == {
        "Je voudrais un verre d'eau.",
        "Le chien mange vite, il aboie.",
        "Et le chien mange vite.",
    }
    assert report["english_paired"] == 7
    assert counts["length_window"] == 6
    assert counts["script"] == 5
    assert counts["marks"] == 3
    assert counts["unique"] == 3
    assert report["forms_with_core_candidate"] == 2
    assert report["pool_candidates"] == 3


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

    assert [(card.target, card.tatoeba_id) for card in selected] == [("eau", 1)]
    assert report["cards"] == 1
    assert report["forms"]["with_card"] == 1


def test_audio_file_names_are_deterministic() -> None:
    sentence = "Je voudrais un verre d'eau."
    first = audio.audio_hash(sentence, "gtts:fr", "-10%")

    assert first == audio.audio_hash(sentence, "gtts:fr", "-10%")
    assert first != audio.audio_hash(sentence, "gtts:en", "-10%")
    assert (
        audio.sentence_audio_name(sentence, "gtts:fr", "-10%", "tts", 0)
        == f"tts_{first}.mp3"
    )
    assert audio.word_audio_name("eau", "gtts:fr", "-10%").startswith("word_")
    assert audio.choose_source({}, "none", french.FRENCH) == "tts"
    noncommercial = {"audio_id": "9", "audio_license": "CC BY-NC 4.0"}
    assert audio.choose_source(noncommercial, "auto", french.FRENCH) == "tts"


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
    passed, message = gate_filter_recheck(
        _audit_inputs(tmp_path, table, rows), french.FRENCH
    )
    assert passed, message

    core = [dict(rows[0], tier="core")]
    passed, message = gate_filter_recheck(
        _audit_inputs(tmp_path, table, core), french.FRENCH
    )
    assert not passed
    assert "marks" in message


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
    passed, message = french.gate_discontinuous(
        _audit_inputs(tmp_path, table, ok), french.FRENCH
    )
    assert passed, message

    bad = [dict(ok[0], sentence="Je ne veux partir.")]
    passed, _ = french.gate_discontinuous(
        _audit_inputs(tmp_path, table, bad), french.FRENCH
    )
    assert not passed


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
    csv_path.write_text(
        "target,sentence\neau,Je voudrais de l'eau.\n", encoding="utf-8"
    )
    index = pack.AudioIndex(
        sentences={"Je voudrais de l'eau.": "tts_x.mp3"}, words={}, voice="gtts:fr"
    )

    manifest = pack.build_manifest(
        french.FRENCH,
        table,
        rows,
        index,
        identity=DeckIdentity(default_output("french", "pack")),
        word_csv=word_csv,
        csv_path=csv_path,
    )

    assert manifest["counts"]["words"] == 3
    assert manifest["counts"]["cards"] == 1
    assert manifest["counts"]["tts_audio"] == 1
    assert manifest["counts"]["native_audio"] == 0
    assert manifest["coverage"]["core_with_card"] == round(
        1 / french.FRENCH.core_concepts, 4
    )
    assert manifest["voice"] == "gtts:fr"
    assert manifest["native_audio_available"] is True
    assert manifest["noun_rows"] == len(_nouns())
    assert manifest["noun_gender_missing"] == 0
    assert manifest["excluded_words"] == ["je", "pluie"]


def test_sentence_csv_columns() -> None:
    with french.SENTENCES_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert header == french.SENTENCE_FIELDS
