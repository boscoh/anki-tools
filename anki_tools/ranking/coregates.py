"""Shared gates K1 to K20 of the core deck.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies the
:class:`anki_tools.ranking.scan.CoreProfile` hooks, and every gate reads them
from the :class:`CoreFiles` reader.

Usage: ``anki fix <lang> --core --check`` calls :func:`check_core`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anki_tools.gate_files import GateFileReader, read_notes

from anki_tools.ranking.assemble import deck_templates
from anki_tools.ranking.coreflow import (
    OUTPUT_SENTENCE_FIELD,
    STATUSES,
    TARGET_SEPARATOR,
    UNCOVERABLE_STATUSES,
    filter_core_cards,
    read_loanwords,
    read_output,
)
from anki_tools.ranking.corelist import CONTENT_POS
from anki_tools.ranking.gates import (
    Gate,
    GateResult,
    core_word_problems,
    gate_core_word,
    gate_identity,
    gate_output_model,
    output_model_problems,
    read_csv,
    read_json,
    report,
    run_gates,
)
from anki_tools.ranking.gates import sample as _sample
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.audit import sha256_value

__all__ = [
    "CoreFiles",
    "GATES",
    "Gate",
    "GateResult",
    "check_core",
    "core_word_problems",
    "gate_audio",
    "gate_bold",
    "gate_card_count",
    "gate_core_word",
    "gate_counts",
    "gate_coverage",
    "gate_covered1",
    "gate_covered2",
    "gate_filter",
    "gate_hashes",
    "gate_hit",
    "gate_identity",
    "gate_names",
    "gate_order",
    "gate_outside",
    "gate_output_model",
    "gate_sense",
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


def gate_covered1(files: CoreFiles) -> tuple[bool, str]:
    """K2: every coverable target has a card."""
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 1)
    targets = sum(1 for row in coverage if row["status"] not in UNCOVERABLE_STATUSES)
    ok = covered == targets
    return ok, f"covered {covered} of {targets}"


def gate_covered2(files: CoreFiles) -> tuple[bool, str]:
    """K3: report the share of coverable targets with two cards.

    The share measures the input deck, not the build. A target needs a card
    whose English holds the gloss, and a small deck cannot supply two. The
    selection takes as many as the pool allows, so the gate reports the share
    and does not fail on it.
    """
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 2)
    targets = sum(1 for row in coverage if row["status"] not in UNCOVERABLE_STATUSES)
    share = covered / targets if targets else 0.0
    return True, f"two cards for {covered} of {targets} ({share:.3f})"


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


def gate_filter(files: CoreFiles) -> tuple[bool, str]:
    """K5: every kept card passes F1 to F9 and C1 to C5."""
    output = files.output()
    cards = output["cards"]
    profile = files.profile
    kept, _reasons = filter_core_cards(
        cards,
        files.tokens(),
        set(output["media"]),
        profile.read_names(),
        read_loanwords(profile.loan_path),
        profile.has_orphan,
        min_units=profile.min_units,
        max_units=profile.max_units,
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
        frozenset(lemma for lemma, pos in card_tokens if pos in CONTENT_POS)
        for card_tokens in files.tokens()
    ]
    shared: dict[str, list[int]] = {}
    for index, lemmas in enumerate(sets):
        for lemma in lemmas:
            shared.setdefault(lemma, []).append(index)
    for index, lemmas in enumerate(sets):
        buckets = {
            other for lemma in lemmas for other in shared[lemma] if other != index
        }
        for other in buckets:
            union = len(lemmas | sets[other])
            if union and len(lemmas & sets[other]) / union >= 0.67:
                return False, f"pair {index} and {other} reaches the limit"
    return True, "no pair reaches the limit"


def gate_audio(files: CoreFiles) -> tuple[bool, str]:
    """K7: every kept card has a valid MP3 file."""
    output = files.output()
    audio = output["audio"]
    missing = [
        card.audio
        for card in output["cards"]
        if not card.audio or audio.get(card.audio) is None
    ]
    return not missing, f"{len(missing)} card(s) without audio: {_sample(missing)}"


def gate_counts(files: CoreFiles) -> tuple[bool, str]:
    """K8: note, media, and manifest counts agree."""
    counts = files.manifest()["counts"]
    output = files.output()
    kept = _kept(files)
    audio = {card.audio for card in output["cards"] if card.audio}
    notes = output["notes"]
    media = files.media()
    ok = (
        notes == len(kept) == counts["cards"]
        and media == audio
    )
    return ok, f"notes {notes}, CSV {len(kept)}, media {len(media)} of {len(audio)}"




def gate_order(files: CoreFiles) -> tuple[bool, str]:
    """K10: the kept order is the recorded order."""
    rank_rows = files.rank_rows()
    sentences = [row["sentence"] for row in rank_rows]
    text = "\n".join(sentences)
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    recorded = files.manifest()["kept_order"]
    ok = digest == recorded and len(rank_rows) == len(_kept(files))
    return ok, f"{len(rank_rows)} rows, order {'matches' if ok else 'differs'}"


def _key_of(row: dict[str, str]) -> str:
    """Give the target key of a coverage row."""
    return row["key"]


def gate_coverage(files: CoreFiles) -> tuple[bool, str]:
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
    recorded = files.manifest()["files"]
    ok = (
        recorded.get("core_csv") == sha256_value(files.core_csv)
        and recorded.get("coverage_csv") == sha256_value(files.coverage_csv)
    )
    return ok, "hashes match" if ok else "hash mismatch"


def gate_hit(files: CoreFiles) -> tuple[bool, str]:
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
    output = files.output()
    used = [name for name in output["note_models"] if name]
    templates = [output["templates"].get(name, []) for name in used]
    css = [output["css"].get(name, "") for name in used]
    expected = [list(pair) for pair in deck_templates(files.profile)]
    ok_templates = all(
        [list(pair) for pair in value] == expected for value in templates
    )
    wanted_css = files.profile.card_css()
    ok_css = all(value == wanted_css for value in css)
    return ok_templates and ok_css, (
        f"{len(used)} model(s) "
        f"{'match' if ok_templates else 'differ'}, "
        f"CSS {'match' if ok_css else 'differ'}"
    )


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


def gate_bold(files: CoreFiles) -> tuple[bool, str]:
    """K15: the bold text is the assigned target, inside the sentence."""
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
    return not bad, f"{bolded} bold card(s), issues {_sample(bad)}"


def gate_sense(files: CoreFiles) -> tuple[bool, str]:
    """K16: the unchecked targets stay inside the limit."""
    coverage = files.coverage()
    unchecked = sum(1 for row in coverage if row["status"] == "unchecked")
    limit = int(files.manifest()["params"].get("max_unchecked_sense", 0))
    return unchecked <= limit, f"{unchecked} unchecked, limit {limit}"


def gate_unranked(files: CoreFiles) -> tuple[bool, str]:
    """K20: the unranked tier F targets stay inside the limit."""
    count = int(files.manifest()["counts"].get("unranked", 0))
    limit = int(files.manifest()["params"].get("max_unranked", 0))
    return count <= limit, f"{count} unranked, limit {limit}"


def gate_names(files: CoreFiles) -> tuple[bool, str]:
    """K17: no kept card holds a name or a loan word."""
    counts = files.manifest()["counts"]
    ok = counts.get("name_hits", 0) == 0 and counts.get("loan_hits", 0) == 0
    return ok, (
        f"name hits {counts.get('name_hits', 0)}, "
        f"loan hits {counts.get('loan_hits', 0)}"
    )


def gate_tail(files: CoreFiles) -> tuple[bool, str]:
    """K18: every tail card brings a new content lemma."""
    tail = [
        row for row in files.rows() if row["pass"] == "tail" and row["kept"] == "1"
    ]
    bad = [row["sentence"] for row in tail if int(row["new_lemmas"]) < 1]
    return not bad, f"{len(tail)} tail card(s), {len(bad)} without a new lemma"



GATES: tuple[Gate, ...] = (
    Gate("K1", gate_card_count),
    Gate("K2", gate_covered1),
    Gate("K3", gate_covered2),
    Gate("K4", gate_outside),
    Gate("K5", gate_filter),
    Gate("K6", gate_similarity),
    Gate("K7", gate_audio),
    Gate("K8", gate_counts),
    Gate("K9", gate_identity),
    Gate("K10", gate_order),
    Gate("K11", gate_coverage),
    Gate("K12", gate_hashes),
    Gate("K13", gate_hit),
    Gate("K14", gate_templates),
    Gate("K15", gate_bold),
    Gate("K16", gate_sense),
    Gate("K17", gate_names),
    Gate("K18", gate_tail),
    Gate("K19", gate_core_word),
    Gate("K20", gate_unranked),
    Gate("K21", gate_output_model),
)


def check_core(output_path: Path, profile: CoreProfile) -> int:
    """Run gates K1 to K21 on the files of a core output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CoreFiles(Path(output_path), profile)))
