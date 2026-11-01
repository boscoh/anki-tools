"""Tests for the sense recheck gate of the cull (G18) and the core (K22)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from anki_tools.ranking import coregates, cullgates, gates

FIELDS = ["Audio", "Sentence", "English", "Core", "CoreGloss"]
SENSE_TSV = (
    "word\tsenses\tfalse_friends\tcompounds\ttier\n"
    "pas\tnot\tstep\t\tF\n"
    "tete\thead\tleader\t\tS\n"
)


def _files(
    tmp_path: Path, cards: list[tuple[str, str, str, str]], **extra: Any
) -> Any:
    """Build a fake reader. A card is ``(sentence, english, core, gloss)``."""
    path = tmp_path / "senses.tsv"
    path.write_text(SENSE_TSV, encoding="utf-8")
    notes = {
        index: ["", sentence, english, core, gloss]
        for index, (sentence, english, core, gloss) in enumerate(cards)
    }
    output = {
        "cards": [
            SimpleNamespace(note_id=index, sentence=card[0], english=card[1])
            for index, card in enumerate(cards)
        ]
    }
    return SimpleNamespace(
        profile=SimpleNamespace(sense_path=path),
        notes=lambda: notes,
        fields=lambda: FIELDS,
        output=lambda: output,
        **extra,
    )


def test_the_shared_gate_passes_a_sense_and_an_unknown_word(tmp_path: Path) -> None:
    files = _files(
        tmp_path,
        [("a", "I do not know.", "pas", "not"), ("b", "Anything.", "inconnu", "")],
    )
    ok, _ = gates.gate_sense(files, files.profile.sense_path, lambda *_: "F")
    assert ok


def test_the_shared_gate_fails_a_tier_f_word_without_a_sense_word(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path, [("a", "Take a step.", "pas", "not")])
    ok, message = gates.gate_sense(files, files.profile.sense_path, lambda *_: "F")
    assert not ok
    assert "pas" in message


def test_the_shared_gate_reads_the_tier_from_the_caller(tmp_path: Path) -> None:
    files = _files(tmp_path, [("a", "The leader spoke.", "tete", "head")])
    path = files.profile.sense_path
    assert not gates.gate_sense(files, path, lambda *_: "S")[0]
    assert gates.gate_sense(files, path, lambda *_: "F")[0]


def test_the_cull_gate_takes_tier_s_from_the_swadesh_source(tmp_path: Path) -> None:
    cards = [("Il a une tete.", "The leader spoke.", "tete", "head")]
    swadesh = _files(
        tmp_path, cards, rows=lambda: [{"source": "swadesh", "sentence": cards[0][0]}]
    )
    frequency = _files(
        tmp_path, cards, rows=lambda: [{"source": "frequency", "sentence": cards[0][0]}]
    )
    assert not cullgates.gate_sense(swadesh)[0]
    assert cullgates.gate_sense(frequency)[0]


def test_the_core_gate_takes_the_tier_from_the_coverage_row(tmp_path: Path) -> None:
    cards = [("Il a une tete.", "The leader spoke.", "tete", "head")]
    tier_s = _files(
        tmp_path,
        cards,
        coverage=lambda: [{"word": "tete", "gloss": "head", "tier": "S"}],
    )
    assert not coregates.gate_sense(tier_s)[0]
    unknown = _files(tmp_path, cards, coverage=lambda: [])
    assert coregates.gate_sense(unknown)[0]


def test_each_flow_registers_the_gate() -> None:
    assert cullgates.GATES[-1].name == "G18"
    assert cullgates.GATES[-1].check is cullgates.gate_sense
    assert coregates.GATES[-1].name == "K22"
    assert coregates.GATES[-1].check is coregates.gate_sense
