"""Shared gates G1 to G18 of the cull deck, without the retired G3 and G4.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies a
:class:`anki_tools.ranking.cullflow.CullProfile`, and every gate reads its
values from the file reader.

Usage: ``anki fix <lang> --select --check`` calls :func:`check_cull`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anki_tools.gate_files import GateFileReader, read_notes

from anki_tools.ranking.coreflow import InputCard, read_output
from anki_tools.ranking.cullflow import (
    FREQUENCY_SOURCE,
    SWADESH_SOURCE,
    CullProfile,
    coverage_status,
    filter_reason,
    frequency_senses,
    lemma_sets,
    sense_lemma_sets,
)
from anki_tools.ranking.gates import (
    Gate,
    gate_audio as _gate_audio,
    gate_counts as _gate_counts,
    gate_core_word,
    gate_identity,
    gate_kept_order as _gate_kept_order,
    gate_manifest_hashes as _gate_manifest_hashes,
    gate_output_model,
    gate_sense as _gate_sense,
    gate_similarity as _gate_similarity,
    gate_templates as _gate_templates,
    read_csv,
    read_json,
    report,
    run_gates,
)
from anki_tools.ranking.gates import sample as _sample
from anki_tools.ranking.identity import DeckIdentity

CARD_CLASSES = ("sentence", "english", "hint")

@dataclass
class CullFiles(GateFileReader):
    """Lazy reader for the APKG, the CSVs, and the manifest of a cull run.

    :param apkg_path: Output APKG file.
    :param profile: Language hooks.
    """

    apkg_path: Path
    profile: CullProfile

    def __post_init__(self) -> None:
        """Set the side paths and the loader table."""
        self.apkg_path = Path(self.apkg_path)
        self.cull_csv, self.coverage_csv, self.manifest_path = (
            DeckIdentity(self.apkg_path).cull_sides()
        )
        super().__init__(
            {
                "rows": lambda: read_csv(self.cull_csv),
                "coverage": lambda: read_csv(self.coverage_csv),
                "manifest": lambda: read_json(self.manifest_path),
                "output": lambda: read_output(self.apkg_path),
                "notes": lambda: read_notes(self.apkg_path),
                "fields": lambda: list(self.output()["fields"]),
                "tokens": lambda: self.profile.tokenize_texts(
                    _sentences(self.output())
                ),
            }
        )

def _sentences(output: dict[str, Any]) -> list[str]:
    """Give the sentence of every card of the output deck."""
    return [card.sentence for card in output["cards"]]


def cull_files(apkg_path: Path, profile: CullProfile) -> CullFiles:
    """Build the file reader of an output APKG.

    :param apkg_path: Output APKG file.
    :param profile: Language hooks.
    :returns: A lazy reader for the APKG and its three side files.
    """
    return CullFiles(apkg_path, profile)




def _kept_rows(files: CullFiles) -> list[dict[str, str]]:
    """Read the kept rows of the cull CSV."""
    return [row for row in files.rows() if row.get("kept") == "1"]


def _exempt_lemmas(files: CullFiles) -> set[str]:
    """Read the target lemmas that are exempt from the card cap."""
    top = int(files.params().get("exempt_top", 0))
    return {row["word"] for row in files.coverage()[:top]}


def gate_card_cap(files: CullFiles) -> tuple[bool, str]:
    """G1: the card count is the cap, or the pool has no more cards.

    A kept card that holds no core word leaves the deck, so the count before
    that drop is the value that the gate compares.
    """
    kept = len(_kept_rows(files))
    counts = files.manifest().get("counts", {})
    dropped = int(counts.get("dropped_no_core", 0))
    cap = int(files.params().get("max_cards", 0))
    pooled = int(counts.get("passed", 0))
    total = kept + dropped
    if total == cap:
        return True, f"{kept} kept + {dropped} dropped, the cap"
    if total == pooled:
        return True, f"{kept} kept + {dropped} dropped, the pool has no more"
    return False, (
        f"{kept} kept + {dropped} dropped, not the cap {cap} "
        f"and below the pool {pooled}"
    )


def gate_use_cap(files: CullFiles) -> tuple[bool, str]:
    """G2: no non-exempt lemma is in more than the cap of kept cards.

    A lemma that fails the sense test on a card is not a use of that card.
    """
    limit = int(files.params().get("max_uses", 0))
    exempt = _exempt_lemmas(files)
    frequency_sentences = {
        row["sentence"]
        for row in files.rows()
        if row.get("source") == FREQUENCY_SOURCE and row.get("kept") == "1"
    }
    cards = files.output()["cards"]
    sets = sense_lemma_sets(
        [card.english for card in cards],
        files.tokens(),
        files.profile.content_pos,
        frequency_senses(files.profile.sense_path),
    )
    uses: Counter[str] = Counter()
    for card, lemma_set in zip(cards, sets):
        if card.sentence in frequency_sentences:
            uses.update(lemma_set)
    over = {
        lemma: count
        for lemma, count in uses.items()
        if lemma not in exempt and count > limit
    }
    if over:
        names = [f"{lemma} {count}" for lemma, count in sorted(over.items())]
        return False, f"{len(over)} lemma(s) over {limit}: {_sample(names)}"
    most = max(
        (count for lemma, count in uses.items() if lemma not in exempt), default=0
    )
    return True, f"max {most} uses outside the exempt {len(exempt)}"


def gate_filter_recheck(files: CullFiles) -> tuple[bool, str]:
    """G5: every kept card passes the filter rules."""
    rows = _kept_rows(files)
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        card = InputCard(
            note_id=0,
            sentence=row["sentence"],
            english=row["english"],
            audio="",
            index=0,
        )
        reason = filter_reason(card, files.profile)
        key = files.profile.dedupe_key(card.sentence)
        if not reason and key in seen:
            reason = "unique"
        seen.add(key)
        if reason:
            problems.append(f"{reason} {card.sentence[:40]}")
    if problems:
        return False, f"{len(problems)} kept card(s) fail: {_sample(problems)}"
    return True, f"{len(rows)} kept cards pass the shared filter rules"


def gate_similarity(files: CullFiles) -> tuple[bool, str]:
    """G6: no kept card pair reaches the lemma Jaccard limit."""
    limit = float(files.params().get("similarity", 0.67))
    sets = lemma_sets(files.tokens(), files.profile.content_pos)
    return _gate_similarity(files, sets, limit)

def gate_audio(files: CullFiles) -> tuple[bool, str]:
    """G7: every kept card has a valid MP3 file."""
    return _gate_audio(files)

def gate_note_count(files: CullFiles) -> tuple[bool, str]:
    """G8: the note count matches the CSV and the manifest."""
    return _gate_counts(files)

def gate_media(files: CullFiles) -> tuple[bool, str]:
    """G9: the APKG holds exactly the media files of the kept cards."""
    output = files.output()
    media = output["media"]
    used = {card.audio for card in output["cards"] if card.audio}
    extra = sorted(media - used)
    missing = sorted(used - media)
    if extra or missing:
        return (
            False,
            f"{len(extra)} extra, {len(missing)} missing: {_sample(extra + missing)}",
        )
    return True, f"{len(media)} media files, all used by the kept cards"




def gate_kept_order(files: CullFiles) -> tuple[bool, str]:
    """G11: the kept set and order match the manifest of the run."""
    return _gate_kept_order(files, files.rows())

def gate_coverage_shape(files: CullFiles) -> tuple[bool, str]:
    """G12: the coverage CSV has one row per target word."""
    rows = files.coverage()
    wanted = int(files.params().get("target_words", 0))
    if len(rows) != wanted:
        return False, f"{len(rows)} coverage rows, not {wanted}"
    words = [row.get("word", "") for row in rows]
    if len(set(words)) != len(words):
        return False, f"{len(words) - len(set(words))} duplicate target word(s)"
    wrong = [
        row["word"]
        for row in rows
        if row.get("status") != coverage_status(int(row.get("count", 0)))
    ]
    if wrong:
        return False, f"{len(wrong)} row(s) with a wrong status: {_sample(wrong)}"
    return True, f"{len(rows)} coverage rows, one per target word"


def gate_manifest_hashes(files: CullFiles) -> tuple[bool, str]:
    """G13: the manifest hashes match the two CSV files."""
    file_keys = {"cull_csv": files.cull_csv, "coverage_csv": files.coverage_csv}
    return _gate_manifest_hashes(files, file_keys)


def _css_class_problems(css: str) -> list[str]:
    """Report the card classes that the CSS does not declare."""
    return [
        f"the CSS has no .{name} class"
        for name in CARD_CLASSES
        if f".{name}" not in css
    ]


def gate_card_templates(files: CullFiles) -> tuple[bool, str]:
    """G15: the deck model holds the Swadesh-style template and CSS."""
    return _gate_templates(files, _css_class_problems)


def gate_sense(files: CullFiles) -> tuple[bool, str]:
    """G18: the core word of every kept card passes the sense test."""
    swadesh = {
        row["sentence"] for row in files.rows() if row.get("source") == SWADESH_SOURCE
    }
    return _gate_sense(
        files,
        files.profile.sense_path,
        lambda _word, _gloss, sentence: "S" if sentence in swadesh else "F",
    )


GATES: tuple[Gate, ...] = (
    Gate("G1", gate_card_cap),
    Gate("G2", gate_use_cap),
    Gate("G5", gate_filter_recheck),
    Gate("G6", gate_similarity),
    Gate("G7", gate_audio),
    Gate("G8", gate_note_count),
    Gate("G9", gate_media),
    Gate("G10", gate_identity),
    Gate("G11", gate_kept_order),
    Gate("G12", gate_coverage_shape),
    Gate("G13", gate_manifest_hashes),
    Gate("G15", gate_card_templates),
    Gate("G16", gate_output_model),
    Gate("G17", gate_core_word),
    Gate("G18", gate_sense),
)


def check_cull(output_path: Path, profile: CullProfile) -> int:
    """Run gates G1 to G18, without the retired G3 and G4, on a cull output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CullFiles(Path(output_path), profile)), "Cull deck")
