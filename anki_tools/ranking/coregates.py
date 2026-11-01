"""Shared gates K5, K6, K13, and K23 of the core deck. The ids K1 to K22 are retired.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies the
:class:`anki_tools.ranking.scan.CoreProfile` hooks, and every gate reads them
from the :class:`CoreFiles` reader.

Usage: ``anki fix <lang> --core --check`` calls :func:`check_core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anki_tools.gate_files import GateFileReader, read_notes

from anki_tools.ranking.core import SIMILARITY_LIMIT
from anki_tools.ranking.coreflow import (
    OUTPUT_CORE_FIELD,
    TARGET_SEPARATOR,
    filter_core_cards,
    read_output,
)
from anki_tools.ranking.corelist import CONTENT_POS, lemma_sets
from anki_tools.ranking.gates import (
    Gate,
    GateResult,
    gate_similarity as _gate_similarity,
    read_csv,
    read_json,
    report,
    run_gates,
)
from anki_tools.ranking.gates import sample as _sample
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.scan import CoreProfile

__all__ = [
    "CoreFiles",
    "GATES",
    "Gate",
    "GateResult",
    "check_core",
    "gate_coverage_floor",
    "gate_filter_recheck",
    "gate_target_in_sentence",
    "gate_similarity",
    "run_gates",
]

@dataclass
class CoreFiles(GateFileReader):
    """Lazy reader for the four files of a core run.

    :param apkg_path: Output APKG file.
    :param profile: Language hooks.
    """

    apkg_path: Path
    profile: CoreProfile

    def __post_init__(self) -> None:
        """Set the paths and the loader table."""
        self.apkg_path = Path(self.apkg_path)
        self.core_csv, self.coverage_csv, self.manifest_path, self.rank_csv = (
            DeckIdentity(self.apkg_path).core_sides()
        )
        super().__init__(
            {
                "rows": lambda: read_csv(self.core_csv),
                "coverage": lambda: read_csv(self.coverage_csv),
                "manifest": lambda: read_json(self.manifest_path),
                "output": lambda: read_output(self.apkg_path),
                "notes": lambda: read_notes(self.apkg_path),
                "fields": lambda: list(self.output()["fields"]),
                "media": lambda: set(self.output()["media"]),
                "tokens": lambda: self.profile.tokenize_deck(
                    self.output()["cards"]
                ),
            }
        )



def _kept(files: CoreFiles) -> list[dict[str, str]]:
    """Give the rows of the kept cards."""
    return [row for row in files.rows() if row["kept"] == "1"]


def _output_by_note(files: CoreFiles) -> dict[int, Any]:
    """Give the output card per note id."""
    return {card.note_id: card for card in files.output()["cards"]}


def gate_filter_recheck(files: CoreFiles) -> tuple[bool, str]:
    """K5: every kept card passes the shared filter rules."""
    output = files.output()
    cards = output["cards"]
    kept, _reasons = filter_core_cards(cards, files.profile)
    if len(kept) != len(cards):
        failed = [
            cards[index].sentence
            for index in range(len(cards))
            if index not in set(kept)
        ]
        return False, f"{len(failed)} card(s) fail a rule: {_sample(failed)}"
    by_note = _output_by_note(files)
    mismatched = [
        row["sentence"]
        for row in _kept(files)
        if int(row["note_id"]) in by_note
        and by_note[int(row["note_id"])].sentence != row["sentence"]
    ]
    if mismatched:
        return False, f"CSV text differs on {_sample(mismatched)}"
    return True, f"{len(cards)} card(s) pass the rules"


def gate_similarity(files: CoreFiles) -> tuple[bool, str]:
    """K6: no kept pair reaches the lemma Jaccard limit."""
    sets = lemma_sets(files.tokens(), CONTENT_POS)
    return _gate_similarity(files, sets, SIMILARITY_LIMIT)



def _key_of(row: dict[str, str]) -> str:
    """Give the target key of a coverage row."""
    return row["key"]


def gate_target_in_sentence(files: CoreFiles) -> tuple[bool, str]:
    """K13: each card holds the form or lemma of its targets and of its Core word.

    The Core word is the word that the card shows in the ``Core`` field. A card
    with an empty ``Core`` field is skipped for this part. The Core word is
    tested like a target of its own tier, so a swap of the Core word between two
    cards fails.
    """
    coverage = files.coverage()
    form = {_key_of(row): (row["tier"], row["word"]) for row in coverage}
    tier_of = {row["word"]: row["tier"] for row in reversed(coverage)}
    core_index = files.fields().index(OUTPUT_CORE_FIELD)
    notes = files.notes()
    by_note = _output_by_note(files)
    note_tokens = {
        card.note_id: card_tokens
        for card, card_tokens in zip(files.output()["cards"], files.tokens())
    }
    bad: list[str] = []
    for row in _kept(files):
        note = int(row["note_id"])
        card = by_note.get(note)
        if card is None:
            bad.append(row["sentence"])
            continue
        lemmas = {
            lemma
            for lemma, pos in note_tokens.get(note, ())
            if pos in CONTENT_POS
        }
        held = [
            form.get(key, ("?", ""))
            for key in row["targets"].split(TARGET_SEPARATOR)
            if key
        ]
        shown = notes[note][core_index] if note in notes else ""
        if shown:
            held.append((tier_of.get(shown, "?"), shown))
        for tier, word in dict.fromkeys(held):
            ok = (
                bool(files.profile.hits(card.sentence, {word}))
                if tier == "S"
                else word in lemmas
            )
            if not ok:
                bad.append(f"{word} in {card.sentence}")
    return not bad, f"{len(bad)} bad hit(s): {_sample(bad)}"

def gate_coverage_floor(files: CoreFiles) -> tuple[bool, str]:
    """K23: the share of coverage rows with a kept card reaches the floor.

    The floor is the ``coverage_floor`` of the language profile. It is a
    ratchet: a drop below it means the selection or the input got worse. An
    absent target counts as not covered.
    """
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 1)
    floor = files.profile.coverage_floor
    share = covered / len(coverage) if coverage else 0.0
    return (
        share >= floor,
        f"{covered} of {len(coverage)} target(s) covered, share {share:.3f}, floor {floor:.2f}",
    )


GATES: tuple[Gate, ...] = (
    Gate("K5", gate_filter_recheck),
    Gate("K6", gate_similarity),
    Gate("K13", gate_target_in_sentence),
    Gate("K23", gate_coverage_floor),
)


def check_core(output_path: Path, profile: CoreProfile) -> int:
    """Run gates K5, K6, K13, and K23 on the files of a core output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CoreFiles(Path(output_path), profile)), "Core deck")
