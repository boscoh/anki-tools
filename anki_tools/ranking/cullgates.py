"""Shared gates C1, C2, C5, and C6 of the cull deck. The other ids C3, C4, and C7 to C18 are retired.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies a
:class:`anki_tools.ranking.cullflow.CullProfile`, and every gate reads its
values from the file reader.

Usage: ``anki fix <lang> --select --check`` calls :func:`check_cull`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from anki_tools.gate_files import GateFileReader, read_notes

from anki_tools.ranking.coreflow import InputCard, read_output
from anki_tools.ranking.cullflow import (
    DEFAULT_MAX_USES,
    EXEMPT_TOP,
    FREQUENCY_SOURCE,
    SIMILARITY,
    CullProfile,
    filter_cards,
    filter_reason,
    frequency_senses,
    sense_lemma_sets,
)
from anki_tools.ranking.corelist import lemma_sets
from anki_tools.ranking.gates import (
    Gate,
    gate_similarity as _gate_similarity,
    read_csv,
    read_json,
    report,
    run_gates,
)
from anki_tools.ranking.gates import sample as _sample
from anki_tools.ranking.identity import DeckIdentity

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





def _kept_rows(files: CullFiles) -> list[dict[str, str]]:
    """Read the kept rows of the cull CSV."""
    return [row for row in files.rows() if row.get("kept") == "1"]


def _exempt_lemmas(files: CullFiles) -> set[str]:
    """Read the target lemmas that are exempt from the card cap."""
    return {row["word"] for row in files.coverage()[:EXEMPT_TOP]}


def gate_card_cap(files: CullFiles) -> tuple[bool, str]:
    """C1: the card count is the cap, or the pool has no more cards.

    A kept card that holds no core word leaves the deck, so the count before
    that drop is the value that the gate compares. The pool has no more cards
    when the selection rejected every other passed card by a rule (``similar``,
    ``use_cap``, ``no_gain``). A card with the reason ``max_cards`` was still
    available, so the selection stopped early.

    The cap comes from the manifest, because the user sets it with ``--cards``.
    """
    kept = len(_kept_rows(files))
    dropped = int(files.manifest().get("counts", {}).get("dropped_no_core", 0))
    cap = int(files.params().get("max_cards", 0))
    total = kept + dropped
    if total == cap:
        return True, f"{kept} kept + {dropped} dropped, the cap"
    left = sum(1 for row in files.rows() if row.get("reason") == "max_cards")
    if left == 0:
        return True, f"{kept} kept + {dropped} dropped, the pool has no more"
    return False, (
        f"{kept} kept + {dropped} dropped, not the cap {cap} "
        f"and {left} card(s) left in the pool"
    )


def gate_use_cap(files: CullFiles) -> tuple[bool, str]:
    """C2: no non-exempt lemma is in more than the cap of kept cards.

    A lemma that fails the sense test on a card is not a use of that card.
    """
    limit = DEFAULT_MAX_USES
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
    """C5: every kept frequency card passes the filter rules.

    A reserve card has its own length window, so the gate skips it.
    """
    rows = [row for row in _kept_rows(files) if row.get("source") == FREQUENCY_SOURCE]
    cards = [
        InputCard(
            note_id=0,
            sentence=row["sentence"],
            english=row["english"],
            audio="",
            index=index,
        )
        for index, row in enumerate(rows)
    ]
    _passed, reasons = filter_cards(
        cards,
        card_rule=partial(filter_reason, profile=files.profile),
        dedupe_key=files.profile.dedupe_key,
    )
    problems = [
        f"{reason} {card.sentence[:40]}"
        for card, reason in zip(cards, reasons)
        if reason
    ]
    if problems:
        return False, f"{len(problems)} kept card(s) fail: {_sample(problems)}"
    return True, f"{len(rows)} kept cards pass the shared filter rules"


def gate_similarity(files: CullFiles) -> tuple[bool, str]:
    """C6: no kept card pair reaches the lemma Jaccard limit."""
    sets = lemma_sets(files.tokens(), files.profile.content_pos)
    return _gate_similarity(files, sets, SIMILARITY)


GATES: tuple[Gate, ...] = (
    Gate("C1", gate_card_cap),
    Gate("C2", gate_use_cap),
    Gate("C5", gate_filter_recheck),
    Gate("C6", gate_similarity),
)


def check_cull(output_path: Path, profile: CullProfile) -> int:
    """Run gates C1, C2, C5, and C6 on a cull output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CullFiles(Path(output_path), profile)), "Cull deck")
