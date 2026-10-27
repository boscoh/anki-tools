"""Shared gates G1 to G15 of the cull deck.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. A language package supplies a
:class:`anki_tools.ranking.cullflow.CullProfile`, and every gate reads its
values from the file reader.

Usage: ``anki fix <lang> --select --check`` calls :func:`check_cull`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.coreflow import (
    InputCard,
    deck_templates,
    read_output,
)
from anki_tools.ranking.corelist import is_content
from anki_tools.ranking.cullflow import (
    FREQUENCY_SOURCE,
    SWADESH_SOURCE,
    CullProfile,
    _is_similar,
    coverage_status,
    filter_reason,
    lemma_sets,
)
from anki_tools.ranking.gates import (
    Gate,
    gate_core_word,
    gate_identity,
    gate_output_model,
    read_csv,
    read_json,
    report,
    run_gates,
)
from anki_tools.ranking.gates import sample as _sample
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.sentence_pack.audit import sha256_value

CARD_CLASSES = ("sentence", "english", "hint")

@dataclass
class CullFiles:
    """Lazy reader for the APKG, the CSVs, and the manifest of a cull run.

    A read error is cached and re-raised for each gate that needs the file, so
    one missing input marks only the dependent gates as failed.

    :param apkg_path: Output APKG file.
    :param profile: Language hooks.
    """

    apkg_path: Path
    profile: CullProfile
    _cache: dict[str, tuple[bool, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set the side paths and start an empty cache."""
        self.apkg_path = Path(self.apkg_path)
        self.cull_csv, self.coverage_csv, self.manifest_path = (
            DeckIdentity(self.apkg_path).cull_sides()
        )

    def _load(self, name: str, loader: Callable[[], Any]) -> Any:
        """Load one value once and cache the value or the error."""
        if name not in self._cache:
            try:
                self._cache[name] = (True, loader())
            except Exception as error:  # noqa: BLE001 - cached for every gate
                self._cache[name] = (False, error)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    def rows(self) -> list[dict[str, str]]:
        """Read the cull CSV rows."""
        return self._load("rows", lambda: read_csv(self.cull_csv))

    def coverage(self) -> list[dict[str, str]]:
        """Read the coverage CSV rows."""
        return self._load("coverage", lambda: read_csv(self.coverage_csv))

    def manifest(self) -> dict[str, Any]:
        """Read the manifest JSON object."""
        return self._load("manifest", lambda: read_json(self.manifest_path))
    def params(self) -> dict[str, Any]:
        """Read the run options of the manifest."""
        params = self.manifest().get("params", {})
        return dict(params)

    def output(self) -> dict[str, Any]:
        """Read the output APKG and probe the duration of its audio."""
        return self._load("output", lambda: read_output(self.apkg_path))

    def notes(self) -> dict[int, list[str]]:
        """Read the note fields of the output deck, keyed by note id."""

        def load() -> dict[int, list[str]]:
            with AnkiPackage(self.apkg_path) as package:
                return {
                    int(row["id"]): str(row["flds"]).split("\x1f")
                    for row in package.get_notes()
                }

        return self._load("notes", load)

    def tokens(self) -> list[list[tuple[str, str]]]:
        """Tokenize the sentences of the output deck."""
        return self._load(
            "tokens", lambda: self.profile.tokenize_texts(_sentences(self.output()))
        )

    def fields(self) -> list[str]:
        """Read the field names of the only note type."""
        return self._load("fields", lambda: list(self.output()["fields"]))


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


def gate_card_count(files: CullFiles) -> tuple[bool, str]:
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
    """G2: no non-exempt lemma is in more than the cap of kept cards."""
    limit = int(files.params().get("max_uses", 0))
    exempt = _exempt_lemmas(files)
    frequency_sentences = {
        row["sentence"]
        for row in files.rows()
        if row.get("source") == FREQUENCY_SOURCE and row.get("kept") == "1"
    }
    uses: Counter[str] = Counter()
    for card, card_tokens in zip(files.output()["cards"], files.tokens()):
        if card.sentence not in frequency_sentences:
            continue
        uses.update(
            {
                lemma
                for lemma, pos in card_tokens
                if is_content(pos, files.profile.content_pos)
            }
        )
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


def gate_target_coverage(files: CullFiles) -> tuple[bool, str]:
    """G3: target coverage reaches the gate share."""
    rows = files.coverage()
    covered = sum(1 for row in rows if int(row.get("count", 0)) >= 1)
    share = covered / len(rows) if rows else 0.0
    gate = files.profile.target_coverage_gate
    message = f"{covered}/{len(rows)} targets covered ({share:.3f}"
    message += f", gate {gate})"
    return bool(rows) and share >= gate, message


def gate_double_coverage(files: CullFiles) -> tuple[bool, str]:
    """G4: the share of targets with two cards reaches the gate share."""
    rows = files.coverage()
    twice = sum(1 for row in rows if int(row.get("count", 0)) >= 2)
    share = twice / len(rows) if rows else 0.0
    gate = files.profile.double_coverage_gate
    message = f"{twice}/{len(rows)} targets with 2 cards ({share:.3f}"
    message += f", gate {gate})"
    return bool(rows) and share >= gate, message


def gate_filter(files: CullFiles) -> tuple[bool, str]:
    """G5: every kept card passes the filter rules."""
    min_words = int(files.params().get("min_words", 0))
    max_words = int(files.params().get("max_words", 0))
    rows = _kept_rows(files)
    tokens = files.profile.tokenize_texts([row["sentence"] for row in rows])
    problems: list[str] = []
    seen: set[str] = set()
    for row, card_tokens in zip(rows, tokens, strict=True):
        card = InputCard(
            note_id=0,
            sentence=row["sentence"],
            english=row["english"],
            audio="",
            index=0,
        )
        reason = filter_reason(card, card_tokens, min_words, max_words, files.profile)
        if reason == "F8" and row.get("source") == SWADESH_SOURCE:
            reason = ""
        if not reason and card.sentence in seen:
            reason = "F9"
        seen.add(card.sentence)
        if reason:
            problems.append(f"{reason} {card.sentence[:40]}")
    if problems:
        return False, f"{len(problems)} kept card(s) fail: {_sample(problems)}"
    return True, f"{len(rows)} kept cards pass F1 to F9"


def gate_similarity(files: CullFiles) -> tuple[bool, str]:
    """G6: no kept card pair reaches the lemma Jaccard limit."""
    limit = float(files.params().get("similarity", 0.67))
    cards = files.output()["cards"]
    sets = lemma_sets(files.tokens(), files.profile.content_pos)
    shared: dict[str, list[int]] = {}
    problems: list[str] = []
    for index, lemma_set in enumerate(sets):
        if _is_similar(lemma_set, sets, shared, limit):
            problems.append(cards[index].sentence[:40])
        for lemma in lemma_set:
            shared.setdefault(lemma, []).append(index)
    if problems:
        return False, f"{len(problems)} duplicate card(s): {_sample(problems)}"
    return True, f"{len(sets)} kept cards below the Jaccard limit {limit}"


def gate_audio(files: CullFiles) -> tuple[bool, str]:
    """G7: every kept card has a valid MP3 file."""
    output = files.output()
    media = output["media"]
    durations = output["audio"]
    problems: list[str] = []
    for card in output["cards"]:
        if not card.audio:
            problems.append(f"no sound: {card.sentence[:30]}")
        elif card.audio not in media:
            problems.append(f"missing media: {card.audio}")
        elif not durations.get(card.audio):
            problems.append(f"bad MP3: {card.audio}")
    if problems:
        return False, f"{len(problems)} card(s): {_sample(problems)}"
    return True, f"{len(output['cards'])} cards have a valid MP3 file"


def gate_note_count(files: CullFiles) -> tuple[bool, str]:
    """G8: the note count matches the CSV and the manifest."""
    notes = int(files.output()["notes"])
    kept = len(_kept_rows(files))
    cards = int(files.manifest().get("counts", {}).get("cards", 0))
    message = f"{notes} notes, {kept} kept rows, manifest {cards}"
    return notes == kept == cards, message


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
    rows = files.rows()
    kept = _kept_rows(files)
    if rows[: len(kept)] != kept:
        return False, "the kept rows are not the first rows of the cull CSV"
    text = "\n".join(row["sentence"] for row in kept)
    digest = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    expected = files.manifest().get("kept_order")
    if digest != expected:
        return False, f"kept order {digest} is not the manifest {expected}"
    return True, f"{len(kept)} kept cards in the manifest order"


def gate_coverage_rows(files: CullFiles) -> tuple[bool, str]:
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
    recorded = files.manifest().get("files", {})
    problems: list[str] = []
    for key, path in (
        ("cull_csv", files.cull_csv),
        ("coverage_csv", files.coverage_csv),
    ):
        actual = sha256_value(path)
        if recorded.get(key) != actual:
            problems.append(f"{key} manifest {recorded.get(key)} != {actual}")
    if problems:
        return False, f"{len(problems)} hash(es) differ: {_sample(problems)}"
    return True, "2 file hashes match"


def gate_swadesh(files: CullFiles) -> tuple[bool, str]:
    """G14: the merged deck covers the Swadesh concepts, minus the exceptions."""
    counts = files.manifest().get("counts", {})
    covered = int(counts.get("swadesh_covered", 0))
    total = int(counts.get("swadesh_total", 0))
    exceptions = files.manifest().get("swadesh_exceptions", [])
    needed = total - len(exceptions)
    if covered < needed:
        return False, f"Swadesh {covered}/{total}, need {needed}"
    return True, f"Swadesh {covered}/{total}, {len(exceptions)} exception(s)"


def gate_card_templates(files: CullFiles) -> tuple[bool, str]:
    """G15: the deck model holds the Swadesh-style template and CSS."""
    output = files.output()
    model_name = str(files.manifest().get("source", {}).get("model"))
    templates = output["templates"].get(model_name, [])
    if not templates:
        return False, f"the model {model_name!r} has no card template"
    expected = list(deck_templates(files.profile))
    problems = [
        f"card {index + 1} is {qfmt!r} / {afmt!r}"
        for index, (qfmt, afmt) in enumerate(templates)
        if index >= len(expected) or (qfmt, afmt) != expected[index]
    ]
    if len(templates) != len(expected):
        problems.append(f"{len(templates)} templates, not {len(expected)}")
    css = output["css"].get(model_name, "")
    for name in CARD_CLASSES:
        if f".{name}" not in css:
            problems.append(f"the CSS has no .{name} class")
    if problems:
        return False, f"{len(problems)} template problem(s): {_sample(problems)}"
    return True, f"{len(templates)} card template and the CSS classes match"



GATES: tuple[Gate, ...] = (
    Gate("G1", gate_card_count),
    Gate("G2", gate_use_cap),
    Gate("G3", gate_target_coverage),
    Gate("G4", gate_double_coverage),
    Gate("G5", gate_filter),
    Gate("G6", gate_similarity),
    Gate("G7", gate_audio),
    Gate("G8", gate_note_count),
    Gate("G9", gate_media),
    Gate("G10", gate_identity),
    Gate("G11", gate_kept_order),
    Gate("G12", gate_coverage_rows),
    Gate("G13", gate_manifest_hashes),
    Gate("G14", gate_swadesh),
    Gate("G15", gate_card_templates),
    Gate("G16", gate_output_model),
    Gate("G17", gate_core_word),
)


def check_cull(output_path: Path, profile: CullProfile) -> int:
    """Run gates G1 to G17 on the files of a cull output.

    :param output_path: Output APKG file.
    :param profile: Language hooks.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return report(run_gates(GATES, CullFiles(Path(output_path), profile)))
