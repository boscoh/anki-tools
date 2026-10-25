"""Tests for the French 1000-card cull (spec sections 10 and 11).

Every test is offline. The token lists are built by hand, so no test loads
spaCy, ffprobe, or the Tatoeba corpus.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Collection

import pytest

from anki_tools.languages.french import cull as cull_fr
from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking import coreflow, corelist, cullflow, cullgates
from anki_tools.ranking.coverage import write_rows
from anki_tools.sentence_pack.audit import sha256_value

ENGLISH = "A short English line."
BASE_TOKENS = [("le", "DET"), ("chat", "NOUN"), ("mange", "VERB"), ("vite", "ADV")]
MIN_WORDS = 3
MAX_WORDS = 12
DECK_NAME = "French 1000 Core Sentences"
DECK_FIELDS = ["French", "sound", "Back", "freq ranks"]


def _card(
    sentence: str,
    index: int = 0,
    english: str = ENGLISH,
    audio: str = "a.mp3",
) -> cull_fr.InputCard:
    return cull_fr.InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def _target(lemma: str, rank: int) -> corelist.TargetWord:
    return corelist.TargetWord(
        lemma=lemma,
        rank=rank,
        deck_count=3,
        content_share=1.0,
        weight=1 / math.log2(rank + 1),
    )


def _freq_row(
    word: str,
    count: int = 10,
    pm: float = 5.0,
    written: float = 5.0,
) -> cullflow.FrequencyRow:
    return cullflow.FrequencyRow(
        word=word,
        tatoeba_count=count,
        tatoeba_pm=pm,
        written_pm=written,
        zipf=4.0,
    )


class TestFrequency:
    def test_the_spoken_weight_changes_the_order(self) -> None:
        rows = [
            _freq_row("parle", pm=100.0, written=1.0),
            _freq_row("ecrit", pm=1.0, written=100.0),
        ]

        assert cull_fr.frequency_ranks(rows, spoken_weight=0.0) == {
            "ecrit": 1,
            "parle": 2,
        }
        assert cull_fr.frequency_ranks(rows, spoken_weight=1.0) == {
            "parle": 1,
            "ecrit": 2,
        }

    def test_the_loader_builds_the_file_when_it_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frequency_path = tmp_path / "french_frequency.tsv"
        tatoeba_path = tmp_path / "fra_sentences.tsv"
        tatoeba_path.write_text("1\tfra\tLe chat mange.\n", encoding="utf-8")
        calls: list[Path] = []

        def fake_build(tatoeba: Path, out_path: Path, profile: object) -> int:
            calls.append(out_path)
            with out_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    ["word", "tatoeba_count", "tatoeba_pm", "written_pm", "zipf"]
                )
                writer.writerow(["chat", 10, "5.000000", "5.000000", "4.0"])
            return 1

        monkeypatch.setattr(cull_fr.cullflow, "build_frequency_cache", fake_build)

        rows = cull_fr.load_frequency(frequency_path, tatoeba_path)
        again = cull_fr.load_frequency(frequency_path, tatoeba_path)

        assert [row.word for row in rows] == ["chat"]
        assert again == rows
        assert frequency_path.exists()
        assert calls == [frequency_path]


class TestNames:
    def test_read_names_skips_blanks_and_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "names.txt"
        path.write_text("# a comment\n\nParis\n  zoe \n", encoding="utf-8")

        assert cull_fr.read_names(path) == {"paris", "zoe"}


@pytest.mark.parametrize(
    ("rule", "sentence", "english", "tokens"),
    [
        ("F1", "Le chat mange vite.", "Hi", BASE_TOKENS),
        ("F2", "Chat mange.", ENGLISH, [("chat", "NOUN"), ("mange", "VERB")]),
        ("F3", "Le chat a 3 ans.", ENGLISH, BASE_TOKENS),
        ("F3", "Le chat [noir] mange.", ENGLISH, BASE_TOKENS),
        ("F3", "Voir www.exemple.fr.", ENGLISH, BASE_TOKENS),
        ("F3", "Le chat mange...", ENGLISH, BASE_TOKENS),
        ("F4", "Le chat mange vite", ENGLISH, BASE_TOKENS),
        ("F5", "Le chat mange. Le chien dort.", ENGLISH, BASE_TOKENS),
        ("F6", "Le chat « mange » vite.", ENGLISH, BASE_TOKENS),
        ("F6", "— Le chat mange vite.", ENGLISH, BASE_TOKENS),
        ("F7", "Le chat, le chien, et l'oiseau mangent.", ENGLISH, BASE_TOKENS),
        (
            "F8",
            "Paris est une ville.",
            ENGLISH,
            [("paris", "PROPN"), ("etre", "AUX"), ("ville", "NOUN")],
        ),
    ],
)
def test_one_card_per_filter_rule(
    rule: str, sentence: str, english: str, tokens: list[tuple[str, str]]
) -> None:
    card = _card(sentence, english=english)

    assert cull_fr._filter_reason(card, tokens, MIN_WORDS, MAX_WORDS) == rule


def test_a_clean_card_passes_every_rule() -> None:
    card = _card("Le chat mange vite.")

    assert cull_fr._filter_reason(card, BASE_TOKENS, MIN_WORDS, MAX_WORDS) == ""


def test_filter_cards_rejects_a_duplicate_text() -> None:
    cards = [
        _card("Le chat mange vite.", index=0),
        _card("Le chien dort bien.", index=1),
        _card("Le chat mange vite.", index=2),
    ]
    tokens = [BASE_TOKENS, BASE_TOKENS, BASE_TOKENS]

    passed, reasons = cull_fr.filter_cards(cards, tokens, MIN_WORDS, MAX_WORDS)

    assert passed == [0, 1]
    assert reasons == ["", "", "F9"]


def test_manifest_has_params_counts_files_and_kept_order(tmp_path: Path) -> None:
    path = tmp_path / "run.manifest.json"

    coreflow.write_manifest(
        path,
        source={"input": "in.apkg", "deck": DECK_NAME, "model": cull_fr.MODEL_NAME},
        params={"max_cards": 2},
        counts={"cards": 2},
        frequency={"spoken_weight": 0.5},
        files={"cull_csv": "sha256:0"},
        kept_order="sha256:1",
        licenses=cull_fr.LICENSES,
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["source"]["model"] == cull_fr.MODEL_NAME
    assert data["params"] == {"max_cards": 2}
    assert data["counts"] == {"cards": 2}
    assert data["frequency"] == {"spoken_weight": 0.5}
    assert data["files"] == {"cull_csv": "sha256:0"}
    assert data["licenses"] == cull_fr.LICENSES
    assert data["kept_order"] == "sha256:1"
    assert data["build_date"]


DECK_SENTENCES = [
    "Le chat aime la fille.",
    "Le chat voit le chien.",
    "Le chien aime la maison.",
]
DECK_TOKENS = {
    DECK_SENTENCES[0]: [("chat", "NOUN"), ("boire", "VERB"), ("fille", "NOUN")],
    DECK_SENTENCES[1]: [("chat", "NOUN"), ("lait", "NOUN"), ("chien", "NOUN")],
    DECK_SENTENCES[2]: [("boire", "VERB"), ("lait", "NOUN"), ("maison", "NOUN")],
}
DECK_ENGLISH = [
    "The cat likes the girl.",
    "The cat sees the dog.",
    "The dog likes the house.",
]


def _tiny_deck(tmp_path: Path) -> Path:
    media = []
    for index in range(len(DECK_SENTENCES)):
        path = tmp_path / f"a{index}.mp3"
        path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
        media.append(str(path))
    cards = [
        {
            "French": sentence,
            "sound": f"[sound:a{index}.mp3]",
            "Back": english,
            "freq ranks": "",
        }
        for index, (sentence, english) in enumerate(
            zip(DECK_SENTENCES, DECK_ENGLISH, strict=True)
        )
    ]
    source = tmp_path / "input.apkg"
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=DECK_FIELDS,
        cards=cards,
        media_files=media,
        model_name=cull_fr.MODEL_NAME,
    )
    return source


def test_write_deck_keeps_the_notes_and_their_media(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "output.apkg"

    counts = cull_fr.write_deck(source, cards, [0, 2], output, DECK_NAME)

    with AnkiPackage(output) as package:
        notes = package.get_notes()
        media = set(package.get_media_mapping().values())
        decks = {deck.get("name") for deck in package.get_decks().values()}
        models = {model.get("name") for model in package.get_models().values()}

    assert counts == {"notes": 2, "media": 2, "items": 2}
    assert len(notes) == 2
    assert media == {"a0.mp3", "a2.mp3"}
    assert decks == {DECK_NAME}
    assert models == {cull_fr.MODEL_NAME}
    assert [card.sentence for card in cull_fr.load_input_deck(output)] == [
        DECK_SENTENCES[0],
        DECK_SENTENCES[2],
    ]


def test_write_deck_sets_the_swadesh_templates(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "templated.apkg"

    cull_fr.write_deck(source, cards, [0], output, DECK_NAME)

    with AnkiPackage(output) as package:
        models = package.get_models(include_templates=True)
    model = next(
        item for item in models.values() if item.get("name") == cull_fr.MODEL_NAME
    )
    templates = [tuple(template[k] for k in ("qfmt", "afmt")) for template in model["tmpls"]]

    assert templates == [cull_fr.CARD_TEMPLATES[0]]
    assert ".sentence" in model["css"]
    assert ".english" in model["css"]
    assert ".hint" in model["css"]


def _add_second_template(path: Path) -> None:
    """Add a copy of the first template and its cards to a deck."""
    with AnkiPackage(path) as package:
        conn = package.conn
        assert conn is not None
        cursor = conn.cursor()
        models = json.loads(cursor.execute("SELECT models FROM col").fetchone()[0])
        model_id = next(iter(models))
        templates = models[model_id]["tmpls"]
        extra = dict(templates[0])
        extra.update({"name": "Card 2", "ord": len(templates)})
        templates.append(extra)
        cursor.execute("UPDATE col SET models = ? WHERE id = 1", (json.dumps(models),))
        rows = cursor.execute(
            "SELECT * FROM cards WHERE ord = 0 AND nid IN "
            "(SELECT id FROM notes WHERE mid = ?)",
            (int(model_id),),
        ).fetchall()
        columns = [row[1] for row in cursor.execute("PRAGMA table_info(cards)")]
        next_card = int(cursor.execute("SELECT MAX(id) FROM cards").fetchone()[0]) + 1
        for offset, row in enumerate(rows):
            record = {name: row[name] for name in row.keys()}
            record.update(id=next_card + offset, ord=1)
            cursor.execute(
                f"INSERT INTO cards ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})",
                [record[column] for column in columns],
            )
        conn.commit()
        package._modified = True
        package.save()


def test_write_deck_removes_the_second_card_template(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    _add_second_template(source)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "one-card.apkg"

    cull_fr.write_deck(source, cards, [0, 1], output, DECK_NAME)

    with AnkiPackage(output) as package:
        models = package.get_models(include_templates=True)
        rows = package.get_cards()
    model = next(
        item for item in models.values() if item.get("name") == cull_fr.MODEL_NAME
    )

    assert [template["name"] for template in model["tmpls"]] == ["Card 1"]
    assert len(rows) == 2
    assert {row["ord"] for row in rows} == {0}


def test_gate_card_templates_fails_on_a_second_template(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "two-templates.apkg"
    cull_fr.write_deck(source, cards, [0], output, DECK_NAME)
    _add_second_template(output)

    rows = cullgates.run_gates(
        cullgates.cull_files(output, cull_fr.FRENCH_CULL_PROFILE)
    )
    message = dict((gate, passed) for gate, passed, _ in rows)["G15"]

    assert message is False


def test_gate_card_templates_fails_on_a_wrong_front(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "wrong.apkg"
    cull_fr.write_deck(source, cards, [0], output, DECK_NAME)

    with AnkiPackage(output) as package:
        package.update_model_template(
            qfmt="{{French}}", afmt=cull_fr.CARD_TEMPLATES[0][1]
        )
        package.save()

    rows = cullgates.run_gates(
        cullgates.cull_files(output, cull_fr.FRENCH_CULL_PROFILE)
    )
    message = dict((gate, passed) for gate, passed, _ in rows)["G15"]

    assert message is False


def test_coverage_gates_read_the_profile_values(tmp_path: Path) -> None:
    profile = replace(
        cull_fr.FRENCH_CULL_PROFILE,
        target_coverage_gate=0.7,
        double_coverage_gate=0.5,
    )
    files = cullgates.cull_files(tmp_path / "gated.apkg", profile)
    write_rows(
        files.coverage_csv,
        cullflow.COVERAGE_COLUMNS,
        [
            {"word": "chat", "rank": "1", "count": "2", "status": "covered"},
            {"word": "mange", "rank": "2", "count": "2", "status": "covered"},
            {"word": "vite", "rank": "3", "count": "1", "status": "partial"},
            {"word": "lent", "rank": "4", "count": "0", "status": "absent"},
        ],
    )

    passed, message = cullgates.gate_target_coverage(files)
    twice_passed, twice_message = cullgates.gate_double_coverage(files)
    french = cullgates.cull_files(tmp_path / "gated.apkg", cull_fr.FRENCH_CULL_PROFILE)
    french_passed, french_message = cullgates.gate_target_coverage(french)

    assert passed is True
    assert message == "3/4 targets covered (0.750, gate 0.7)"
    assert twice_passed is True
    assert twice_message == "2/4 targets with 2 cards (0.500, gate 0.5)"
    assert french_passed is False
    assert french_message == "3/4 targets covered (0.750, gate 0.98)"


def _fake_tokens(self: cullgates.CullFiles) -> list[list[tuple[str, str]]]:
    return [DECK_TOKENS[card.sentence] for card in self.output()["cards"]]


def _fake_probe(
    _temp_dir: str | None, _media: dict[str, str], names: Collection[str]
) -> dict[str, float]:
    return {name: 1.0 for name in names}


def test_run_gates_passes_on_a_tiny_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    output = tmp_path / "gated.apkg"
    ordered = [0, 1, 2]
    cull_fr.write_deck(source, cards, ordered, output, DECK_NAME)

    targets = [_target("chat", 1), _target("boire", 2), _target("lait", 3)]
    selection = cullflow.SelectionResult(
        kept=ordered,
        uses=Counter({"chat": 2, "boire": 2, "lait": 2}),
        gain={index: 0.5 for index in ordered},
        exempt=set(),
        reasons={},
    )
    cull_csv, coverage_csv, manifest_path = cullflow.side_paths(output)
    cullflow.write_cull_csv(cards, ["", "", ""], selection, ordered, cull_csv)
    cullflow.write_coverage_csv(targets, selection, coverage_csv)
    coreflow.write_manifest(
        manifest_path,
        source={"input": str(source), "deck": DECK_NAME, "model": cull_fr.MODEL_NAME},
        params={
            "max_cards": 3,
            "target_words": 3,
            "max_uses": 4,
            "exempt_top": 0,
            "spoken_weight": 0.5,
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "similarity": 0.67,
            "order": "coverage",
            "name": DECK_NAME,
        },
        counts={"input": 3, "passed": 3, "cards": 3},
        frequency={"path": str(cull_fr.FREQUENCY_PATH), "spoken_weight": 0.5},
        files={
            "cull_csv": sha256_value(cull_csv),
            "coverage_csv": sha256_value(coverage_csv),
        },
        kept_order=cullflow.kept_order_hash(cards, ordered),
    )
    monkeypatch.setattr(cullgates.CullFiles, "tokens", _fake_tokens)
    monkeypatch.setattr(cullgates, "probe_audio", _fake_probe)

    rows = cullgates.run_gates(
        cullgates.cull_files(output, cull_fr.FRENCH_CULL_PROFILE)
    )

    assert [gate for gate, _passed, _message in rows] == [
        f"G{number}" for number in range(1, 16)
    ]
    for gate, passed, message in rows:
        assert passed, f"{gate}: {message}"


SWADESH_FIELDS = [
    "Audio",
    "Sentence",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
]


def _tiny_swadesh_deck(tmp_path: Path) -> Path:
    audio = tmp_path / "tts_reserve.mp3"
    audio.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
    word = tmp_path / "word_reserve.mp3"
    word.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(40))
    source = tmp_path / "swadesh.apkg"
    AnkiPackage.create(
        source,
        "Swadesh Source",
        fields=SWADESH_FIELDS,
        cards=[
            {
                "Audio": "[sound:tts_reserve.mp3]",
                "Sentence": "<b>Je</b> suis ici.",
                "English": "I am here.",
                "Target": "je",
                "Gloss": "I",
                "WordAudio": "[sound:word_reserve.mp3]",
                "Source": "Tatoeba #1",
            }
        ],
        media_files=[str(audio), str(word)],
        model_name="Swadesh French Sentences",
    )
    return source


def test_load_reserve_reads_the_plain_and_bold_sentence(tmp_path: Path) -> None:
    reserve = cull_fr.load_reserve(_tiny_swadesh_deck(tmp_path))

    assert len(reserve) == 1
    assert reserve[0].sentence == "Je suis ici."
    assert reserve[0].html == "<b>Je</b> suis ici."
    assert reserve[0].target == "je"
    assert reserve[0].audio == "tts_reserve.mp3"


def test_write_deck_inserts_the_reserve_card(tmp_path: Path) -> None:
    source = _tiny_deck(tmp_path)
    cards = cull_fr.load_input_deck(source)
    swadesh = _tiny_swadesh_deck(tmp_path)
    reserve = cull_fr.load_reserve(swadesh)
    output = tmp_path / "merged.apkg"

    counts = cull_fr.write_deck(
        source,
        cards,
        [0, 1],
        output,
        DECK_NAME,
        reserve=reserve,
        reserve_pack_path=swadesh,
    )

    with AnkiPackage(output) as package:
        notes = [dict(note) for note in package.get_notes()]
        media = set(package.get_media_mapping().values())
        models = {model.get("name") for model in package.get_models().values()}

    assert counts == {"notes": 3, "media": 3, "items": 3}
    assert media == {"a0.mp3", "a1.mp3", "tts_reserve.mp3"}
    assert models == {cull_fr.MODEL_NAME}
    fields = [str(note["flds"]).split("\x1f") for note in notes]
    assert any("<b>Je</b> suis ici." in row[0] for row in fields)
    reserve_row = next(row for row in fields if "<b>Je</b>" in row[0])
    assert reserve_row[1] == "[sound:tts_reserve.mp3]"
    assert reserve_row[2] == "I am here. — je (I)"


def test_swadesh_coverage_counts_a_word_from_another_card() -> None:
    sentence = "Un nuage flottait dans le ciel."
    tokens = cull_fr.tokenize_texts([sentence])

    covered, total = cull_fr.swadesh_coverage(
        ["nuage", "intestins"], [sentence], tokens
    )

    assert covered == 1
    assert total == cull_fr.SWADESH_TOTAL
