"""Shared gates K1 to K22 of the core deck.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies the
:class:`anki_tools.ranking.scan.CoreProfile` hooks, and every gate reads them
from the :class:`CoreFiles` reader.

Usage: ``anki fix <lang> --core --check`` calls :func:`check_core`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anki_tools.gate_files import GateFileReader, read_notes

from anki_tools.ranking.core import SIMILARITY_LIMIT
from anki_tools.ranking.coreflow import (
    OUTPUT_SENTENCE_FIELD,
    STATUSES,
    TARGET_SEPARATOR,
    UNCOVERABLE_STATUSES,
    coverage_glosses,
    filter_core_cards,
    read_loanwords,
    read_output,
)
from anki_tools.ranking.corelist import CONTENT_POS
from anki_tools.ranking.gates import (
    Gate,
    GateResult,
    core_word_problems,
    gate_audio as _gate_audio,
    gate_counts as _gate_counts,
    gate_identity,
    gate_kept_order as _gate_kept_order,
    gate_manifest_hashes as _gate_manifest_hashes,
    gate_output_model,
    gate_sense as _gate_sense,
    gate_similarity as _gate_similarity,
    gate_templates as _gate_templates,
    output_model_problems,
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
    "core_word_problems",
    "gate_audio",
    "gate_card_count",
    "gate_core_word",
    "gate_counts",
    "gate_coverage_agreement",
    "gate_covered_targets",
    "gate_filter_recheck",
    "gate_hashes",
    "gate_target_in_sentence",
    "gate_identity",
    "gate_order",
    "gate_outside",
    "gate_output_model",
    "gate_sense",
    "gate_unchecked_sense",
    "gate_similarity",
    "gate_tail",
    "gate_templates",
    "gate_unranked",
    "output_model_problems",
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
                "rank": lambda: read_csv(self.rank_csv),
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


def gate_card_count(files: CoreFiles) -> tuple[bool, str]:
    """K1: the card count agrees in the CSV, the APKG, and the manifest."""
    counts = files.manifest()["counts"]
    kept = _kept(files)
    cards = len(files.output()["cards"])
    ok = counts["cards"] == len(kept) == cards
    return ok, f"manifest {counts['cards']}, CSV {len(kept)}, APKG {cards}"


def gate_covered_targets(files: CoreFiles) -> tuple[bool, str]:
    """K2: every coverable target has a card."""
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 1)
    targets = sum(1 for row in coverage if row["status"] not in UNCOVERABLE_STATUSES)
    ok = covered == targets
    return ok, f"covered {covered} of {targets}"



def gate_outside(files: CoreFiles) -> tuple[bool, str]:
    """K4: the outside set and the rare outside words stay small."""
    counts = files.manifest()["counts"]
    targets = max(1, counts["targets"])
    outside = counts["outside"]
    profile = files.profile
    ok_set = outside <= profile.outside_set_ratio * targets
    ok_rare = counts["outside_rare"] <= profile.outside_rare_ratio * max(1, outside)
    return (
        ok_set and ok_rare,
        f"outside {outside} of {targets} targets, rare {counts['outside_rare']}",
    )


def gate_filter_recheck(files: CoreFiles) -> tuple[bool, str]:
    """K5: every kept card passes the shared filter rules and C3 to C5."""
    output = files.output()
    cards = output["cards"]
    profile = files.profile
    kept, _reasons = filter_core_cards(
        cards,
        files.tokens(),
        set(output["media"]),
        profile.read_names(),
        read_loanwords(profile.loan_path),
        min_units=profile.min_units,
        max_units=profile.max_units,
        shape=profile,
    )
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
    sets = [
        {lemma for lemma, pos in card_tokens if pos in CONTENT_POS}
        for card_tokens in files.tokens()
    ]
    return _gate_similarity(files, sets, SIMILARITY_LIMIT)

def gate_audio(files: CoreFiles) -> tuple[bool, str]:
    """K7: every kept card has a valid MP3 file."""
    return _gate_audio(files)

def gate_counts(files: CoreFiles) -> tuple[bool, str]:
    """K8: the note, media, and manifest counts agree."""
    return _gate_counts(files, check_media=True)



def gate_order(files: CoreFiles) -> tuple[bool, str]:
    """K10: the kept order is the recorded order."""
    return _gate_kept_order(files, files.rank_rows())

def _key_of(row: dict[str, str]) -> str:
    """Give the target key of a coverage row."""
    return row["key"]


def gate_coverage_agreement(files: CoreFiles) -> tuple[bool, str]:
    """K11: the coverage CSV agrees with the card CSV."""
    coverage = files.coverage()
    kept = _kept(files)
    used = Counter(
        key
        for row in kept
        for key in row["targets"].split(TARGET_SEPARATOR)
        if key
    )
    problems: list[str] = []
    positions = [int(row["position"]) for row in coverage]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        problems.append("positions")
    for row in coverage:
        if row["status"] not in STATUSES:
            problems.append(f"status {row['status']}")
        if int(row["count"]) != used.get(_key_of(row), 0):
            problems.append(f"count of {row['word']}")
    by_target = {_key_of(row): row for row in coverage}
    for key, count in used.items():
        row = by_target.get(key)
        if row is None or int(row["count"]) != count:
            problems.append(f"missing target {key}")
    return not problems, f"{len(coverage)} rows, issues {_sample(problems)}"


def gate_hashes(files: CoreFiles) -> tuple[bool, str]:
    """K12: the manifest hashes match the two CSV files."""
    file_keys = {"core_csv": files.core_csv, "coverage_csv": files.coverage_csv}
    return _gate_manifest_hashes(files, file_keys)

def gate_target_in_sentence(files: CoreFiles) -> tuple[bool, str]:
    """K13: each card holds the form or lemma of its targets."""
    coverage = files.coverage()
    form = {_key_of(row): (row["tier"], row["word"]) for row in coverage}
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
        for key in row["targets"].split(TARGET_SEPARATOR):
            if not key:
                continue
            tier, word = form.get(key, ("?", ""))
            ok = (
                bool(files.profile.hits(card.sentence, {word}))
                if tier == "S"
                else word in lemmas
            )
            if not ok:
                bad.append(f"{word} in {card.sentence}")
    return not bad, f"{len(bad)} bad hit(s): {_sample(bad)}"


def gate_templates(files: CoreFiles) -> tuple[bool, str]:
    """K14: the templates are the core templates and the CSS is the cull CSS."""
    wanted = files.profile.card_css()

    def css_problems(css: str) -> list[str]:
        """Report a CSS value that is not the profile CSS."""
        return [] if css == wanted else ["the CSS is not the profile CSS"]

    return _gate_templates(files, css_problems)

def _strip_bold(text: str) -> str:
    """Remove the bold markup of a field value."""
    return text.replace("<b>", "").replace("</b>", "")


def _bold_text(text: str) -> str:
    """Give the text between the first pair of bold tags."""
    start = text.find("<b>")
    end = text.find("</b>", start)
    if start < 0 or end < 0:
        return ""
    return text[start + 3 : end]


def _bold_target_problems(files: CoreFiles) -> tuple[list[str], int]:
    """Give the bold-target problems of the kept notes and the bold count.

    :param files: Reader of the output files.
    :returns: One problem line per fault, and the count of notes with a bold.
    """
    manifest = files.manifest()
    params = manifest["params"]
    notes = files.notes()
    names = files.fields()
    sentence_index = names.index(OUTPUT_SENTENCE_FIELD)
    bad: list[str] = []
    bolded = 0
    form = {_key_of(row): row["word"] for row in files.coverage()}
    for row in _kept(files):
        fields = notes.get(int(row["note_id"]), [])
        text = fields[sentence_index] if sentence_index < len(fields) else ""
        plain = _strip_bold(text)
        if plain != row["sentence"]:
            bad.append(f"text of {row['sentence']}")
            continue
        if "<b>" not in text:
            bad.append(f"no bold: {row['sentence']}")
            continue
        bolded += 1
        target = row["assigned"]
        if not target:
            bad.append(f"bold without a target: {row['sentence']}")
            continue
        bold = _bold_text(text)
        if not bold or bold not in row["sentence"]:
            bad.append(f"bold {bold} is not in {form.get(target, target)}")
    if (
        params.get("bold_target")
        and bolded + manifest["counts"].get("unbolded", 0) != len(_kept(files))
    ):
        bad.append("bold count")
    return bad, bolded


def gate_core_word(files: CoreFiles) -> tuple[bool, str]:
    """K19: every note holds a core word, its coverage gloss, and the bold target."""
    problems, filled = core_word_problems(
        files.notes(), files.fields(), coverage_glosses(files.coverage())
    )
    bold_problems, bolded = _bold_target_problems(files)
    problems += bold_problems
    message = f"{filled} core word(s), {bolded} bold card(s), {len(problems)} bad"
    return not problems, f"{message}: {_sample(problems)}"

def gate_unchecked_sense(files: CoreFiles) -> tuple[bool, str]:
    """K16: the unchecked targets stay inside the limit."""
    coverage = files.coverage()
    unchecked = sum(1 for row in coverage if row["status"] == "unchecked")
    limit = int(files.manifest()["params"].get("max_unchecked_sense", 0))
    return unchecked <= limit, f"{unchecked} unchecked, limit {limit}"


def gate_sense(files: CoreFiles) -> tuple[bool, str]:
    """K22: the core word of every kept card passes the sense test."""
    tiers = {(row["word"], row["gloss"]): row["tier"] for row in files.coverage()}
    return _gate_sense(
        files,
        files.profile.sense_path,
        lambda word, gloss, _sentence: tiers.get((word, gloss), "F"),
    )


def gate_unranked(files: CoreFiles) -> tuple[bool, str]:
    """K20: the unranked tier F targets stay inside the limit."""
    count = int(files.manifest()["counts"].get("unranked", 0))
    limit = int(files.manifest()["params"].get("max_unranked", 0))
    return count <= limit, f"{count} unranked, limit {limit}"



def gate_tail(files: CoreFiles) -> tuple[bool, str]:
    """K18: every tail card brings a new content lemma."""
    tail = [
        row for row in files.rows() if row["pass"] == "tail" and row["kept"] == "1"
    ]
    bad = [row["sentence"] for row in tail if int(row["new_lemmas"]) < 1]
    return not bad, f"{len(tail)} tail card(s), {len(bad)} without a new lemma"



GATES: tuple[Gate, ...] = (
    Gate("K1", gate_card_count),
    Gate("K2", gate_covered_targets),
    Gate("K4", gate_outside),
    Gate("K5", gate_filter_recheck),
    Gate("K6", gate_similarity),
    Gate("K7", gate_audio),
    Gate("K8", gate_counts),
    Gate("K9", gate_identity),
    Gate("K10", gate_order),
    Gate("K11", gate_coverage_agreement),
    Gate("K12", gate_hashes),
    Gate("K13", gate_target_in_sentence),
    Gate("K14", gate_templates),
    Gate("K16", gate_unchecked_sense),
    Gate("K18", gate_tail),
    Gate("K19", gate_core_word),
    Gate("K20", gate_unranked),
    Gate("K21", gate_output_model),
    Gate("K22", gate_sense),
)


def check_core(output_path: Path, profile: CoreProfile) -> int:
    """Run gates K1 to K22 on the files of a core output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CoreFiles(Path(output_path), profile)), "Core deck")
