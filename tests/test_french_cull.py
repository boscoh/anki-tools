"""Tests for the French 1000-card cull (spec sections 10 and 11).

Every test is offline. The token lists are built by hand, so no test loads
spaCy, ffprobe, or the Tatoeba corpus.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Collection

import pytest

from anki_tools.languages.french import cull as cull_fr
from anki_tools.apkg.package import AnkiPackage
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


def _target(lemma: str, rank: int) -> cull_fr.TargetWord:
    return cull_fr.TargetWord(
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
) -> cull_fr.FrequencyRow:
    return cull_fr.FrequencyRow(
        word=word,
        tatoeba_count=count,
        tatoeba_pm=pm,
        written_pm=written,
        zipf=4.0,
    )


class TestFrequency:
    def test_a_count_below_five_uses_the_written_term(self) -> None:
        row = _freq_row("rare", count=4, pm=0.0, written=12.5)

        assert cull_fr.blend_pm(row, spoken_weight=1.0) == pytest.approx(12.5)
        assert cull_fr.blend_pm(row, spoken_weight=0.0) == pytest.approx(12.5)

    def test_the_middle_weight_uses_both_terms(self) -> None:
        row = _freq_row("chat", count=10, pm=20.0, written=10.0)

        assert cull_fr.blend_pm(row, spoken_weight=0.5) == pytest.approx(15.0)

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


def _target_fixture() -> tuple[
    list[cull_fr.InputCard], list[list[tuple[str, str]]], dict[str, int]
]:
    pairs = (
        [("mange", "NOUN")] * 3
        + [("le", "DET")] * 5
        + [("table", "NOUN")] * 2
        + [("fait", "VERB")] * 3
        + [("fait", "AUX")] * 4
        + [("a", "VERB")] * 3
        + [("paris", "PROPN")] * 3
        + [("chien", "NOUN")] * 4
    )
    tokens = [pairs[start : start + 3] for start in range(0, len(pairs), 3)]
    cards = [_card("Le chat mange ici.", index=index) for index in range(len(tokens))]
    return cards, tokens, {"mange": 10, "chien": 2, "le": 1}


class TestNames:
    def test_read_names_skips_blanks_and_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "names.txt"
        path.write_text("# a comment\n\nParis\n  zoe \n", encoding="utf-8")

        assert cull_fr.read_names(path) == {"paris", "zoe"}


class TestTargets:
    def test_the_target_rules_keep_two_of_the_candidates(self) -> None:
        cards, tokens, ranks = _target_fixture()

        targets, content = cull_fr.build_targets(
            cards, tokens, ranks, {"paris"}, target_words=300
        )

        assert [target.lemma for target in targets] == ["chien", "mange"]
        assert content == 2
        assert targets[0].rank == 2
        assert targets[1].rank == 10

    def test_the_target_cut_keeps_the_best_rank(self) -> None:
        cards, tokens, ranks = _target_fixture()

        targets, content = cull_fr.build_targets(
            cards, tokens, ranks, {"paris"}, target_words=1
        )

        assert [target.lemma for target in targets] == ["chien"]
        assert content == 2

    def test_a_mismatched_token_list_raises(self) -> None:
        cards, tokens, ranks = _target_fixture()

        with pytest.raises(ValueError):
            cull_fr.build_targets(cards[:-1], tokens, ranks, set(), target_words=3)


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


def _cap_fixture() -> tuple[
    list[cull_fr.InputCard],
    list[list[tuple[str, str]]],
    list[cull_fr.TargetWord],
    dict[str, int],
]:
    sentences = [
        "Le chat est ici.",
        "Le chat est la.",
        "Le chat est seul.",
        "Le chat est noir.",
        "Le chat est grand.",
    ]
    cards = [_card(sentence, index=index) for index, sentence in enumerate(sentences)]
    tokens = [
        [("chat", "NOUN"), (f"f{index}", "NOUN")] for index in range(len(sentences))
    ]
    return cards, tokens, [_target("chat", 1)], {"chat": 1}


class TestSelection:
    def test_a_rare_word_wins_over_a_repeated_word(self) -> None:
        cards = [
            _card("Une phrase rare ici.", index=0),
            _card("Une phrase commune ici.", index=1),
            _card("Une autre phrase commune.", index=2),
        ]
        tokens = [
            [("rare", "NOUN"), ("f1", "NOUN")],
            [("commune", "NOUN"), ("f2", "NOUN")],
            [("commune", "NOUN"), ("f3", "NOUN")],
        ]
        targets = [_target("rare", 1), _target("commune", 50)]
        ranks = {"rare": 1, "commune": 50}

        result = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=4, exempt_top=0
        )

        assert result.kept == [0, 1, 2]
        assert result.gain[1] > result.gain[2]

    def test_a_non_exempt_word_stops_at_the_cap(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=2, exempt_top=0
        )

        assert len(result.kept) == 2
        assert result.uses["chat"] == 2

    def test_a_card_with_a_capped_word_is_skipped(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=2, exempt_top=0
        )

        rejected = [index for index in range(len(cards)) if index not in result.kept]
        assert [result.reasons[index] for index in rejected] == ["use_cap"] * 3

    def test_an_exempt_word_passes_the_cap(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        result = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=2, exempt_top=1
        )

        assert len(result.kept) == len(cards)
        assert result.uses["chat"] == len(cards)

    def test_a_lemma_jaccard_of_067_is_duplicate(self) -> None:
        shared = [f"s{index}" for index in range(67)]
        first = shared + [f"a{index}" for index in range(20)]
        second = shared + [f"b{index}" for index in range(13)]
        cards = [
            _card("Le chat est ici.", index=0),
            _card("Le chat est la.", index=1),
        ]
        tokens = [
            [(lemma, "NOUN") for lemma in first],
            [(lemma, "NOUN") for lemma in second],
        ]
        ranks = {lemma: 100 for lemma in shared}

        result = cull_fr.select_cards(
            cards, tokens, [], ranks, max_cards=10, max_uses=4, exempt_top=0
        )

        assert len(result.kept) == 1
        assert result.reasons == {1: "similar"}

    def test_two_runs_keep_the_same_order(self) -> None:
        cards, tokens, targets, ranks = _cap_fixture()

        first = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=2, exempt_top=0
        )
        second = cull_fr.select_cards(
            cards, tokens, targets, ranks, max_cards=10, max_uses=2, exempt_top=0
        )

        assert first.kept == second.kept


ORDER_TOKENS = [
    [("maison", "NOUN"), ("grande", "ADJ")],
    [("chat", "NOUN"), ("noir", "ADJ")],
    [("et", "DET")],
]
ORDER_RANKS = {"maison": 5, "grande": 40, "chat": 2, "noir": 900}


class TestOrder:
    def test_coverage_gives_a_non_decreasing_hardest_rank(self) -> None:
        ordered = cull_fr.order_cards([0, 1, 2], ORDER_TOKENS, ORDER_RANKS)

        hardest = [
            cull_fr.hardest_rank(ORDER_TOKENS[index], ORDER_RANKS) for index in ordered
        ]
        assert ordered == [0, 1, 2]
        assert hardest == [40, 900, cull_fr.UNRANKED_ORDER]

    def test_input_keeps_the_given_order(self) -> None:
        ordered = cull_fr.order_cards(
            [2, 0, 1], ORDER_TOKENS, ORDER_RANKS, order="input"
        )

        assert ordered == [2, 0, 1]

    def test_an_unknown_order_raises(self) -> None:
        with pytest.raises(ValueError):
            cull_fr.order_cards([0], ORDER_TOKENS, ORDER_RANKS, order="random")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def test_cull_csv_columns_and_reasons(tmp_path: Path) -> None:
    cards = [
        _card("Le chat mange vite.", index=0),
        _card("Le chien dort bien.", index=1),
        _card("Le chat mange vite.", index=2),
    ]
    selection = cull_fr.SelectionResult(
        kept=[1],
        uses=Counter({"chien": 1}),
        gain={1: 0.5},
        exempt=set(),
        reasons={2: "similar"},
        card_targets={1: ("chien",)},
        card_hardest={1: 3},
    )
    path = tmp_path / "run.cull.csv"

    cull_fr.write_cull_csv(cards, ["F7", "", ""], selection, [1], path)
    header, rows = _read_csv(path)
    raw = path.read_bytes()
    assert raw.split(b"\n")[0] == ",".join(cull_fr.CULL_COLUMNS).encode()
    assert b"\r" not in raw

    assert header == list(cull_fr.CULL_COLUMNS)
    assert len(rows) == 3
    assert [row["sentence"] for row in rows] == [
        cards[1].sentence,
        cards[0].sentence,
        cards[2].sentence,
    ]
    assert [row["kept"] for row in rows] == ["1", "0", "0"]
    assert [row["reason"] for row in rows] == ["", "F7", "similar"]
    assert rows[0]["gain"] == "0.500000"
    assert rows[0]["targets"] == "chien"
    assert rows[0]["hardest_rank"] == "3"
    assert rows[1]["gain"] == ""


def test_coverage_csv_statuses(tmp_path: Path) -> None:
    targets = [_target("chat", 1), _target("boire", 2), _target("lait", 3)]
    selection = cull_fr.SelectionResult(
        kept=[0, 1],
        uses=Counter({"chat": 2, "boire": 1}),
        gain={0: 0.5, 1: 0.4},
        exempt=set(),
        reasons={},
    )
    path = tmp_path / "run.coverage.csv"

    cull_fr.write_coverage_csv(targets, selection, path)
    header, rows = _read_csv(path)

    assert header == list(cull_fr.COVERAGE_COLUMNS)
    assert [row["word"] for row in rows] == ["chat", "boire", "lait"]
    assert [row["count"] for row in rows] == ["2", "1", "0"]
    assert [row["status"] for row in rows] == ["covered", "partial", "absent"]


def test_manifest_has_params_counts_files_and_kept_order(tmp_path: Path) -> None:
    path = tmp_path / "run.manifest.json"

    cull_fr.write_manifest(
        path,
        source={"input": "in.apkg", "deck": DECK_NAME, "model": cull_fr.MODEL_NAME},
        params={"max_cards": 2},
        counts={"cards": 2},
        frequency={"spoken_weight": 0.5},
        files={"cull_csv": "sha256:0"},
        kept_order="sha256:1",
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

    rows = cull_fr.run_gates(cull_fr.gate_inputs(output))
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

    rows = cull_fr.run_gates(cull_fr.gate_inputs(output))
    message = dict((gate, passed) for gate, passed, _ in rows)["G15"]

    assert message is False


def _fake_tokens(self: cull_fr.GateInputs) -> list[list[tuple[str, str]]]:
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
    selection = cull_fr.SelectionResult(
        kept=ordered,
        uses=Counter({"chat": 2, "boire": 2, "lait": 2}),
        gain={index: 0.5 for index in ordered},
        exempt=set(),
        reasons={},
    )
    cull_csv, coverage_csv, manifest_path = cull_fr.side_paths(output)
    cull_fr.write_cull_csv(cards, ["", "", ""], selection, ordered, cull_csv)
    cull_fr.write_coverage_csv(targets, selection, coverage_csv)
    cull_fr.write_manifest(
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
        kept_order=cull_fr.kept_order_hash(cards, ordered),
    )
    monkeypatch.setattr(cull_fr.GateInputs, "tokens", _fake_tokens)
    monkeypatch.setattr(cull_fr, "_probe_audio", _fake_probe)

    rows = cull_fr.run_gates(cull_fr.gate_inputs(output))

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
        model_name="French Swadesh Sentence",
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


def test_dedupe_reserve_drops_an_internal_duplicate() -> None:
    reserve = [
        cull_fr.ReserveCard(
            sentence="Le chat est ici.",
            html="Le <b>chat</b> est ici.",
            english="The cat is here.",
            target="chat",
            gloss="cat",
            audio="a.mp3",
        ),
        cull_fr.ReserveCard(
            sentence="Le chat est ici aussi.",
            html="Le <b>chat</b> est ici aussi.",
            english="The cat is here too.",
            target="chat",
            gloss="cat",
            audio="b.mp3",
        ),
    ]
    sets = [
        {"chat", "ici", "aussi"},
        {"chat", "ici", "aussi", "le"},
    ]

    kept, kept_sets = cull_fr.dedupe_reserve_sets(reserve, sets, [], [])

    assert len(kept) == 1
    assert kept_sets == [{"chat", "ici", "aussi"}]
