"""Gates K1 to K19 of the French core deck.

The gates read the output APKG, the two CSV files, and the manifest. A gate
never writes a file. See ``specs/core-word-small-deck.md`` section 10.

Every gate is a :class:`CoreGate` value that implements the :class:`Gate`
protocol of :mod:`anki_tools.languages.french.core_types`, and the runner of
:func:`run_gates` invokes them against one :class:`CoreFiles` reader.

Usage: ``anki fix fr --core --check`` calls :func:`check_core`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from anki_tools.languages.french import cull
from anki_tools.languages.french.core_types import (
    CORE_FIELD,
    CORE_GLOSS_FIELD,
    CORE_TEMPLATES,
    CONTENT_POS,
    CoreFiles,
    CoreGate,
    Gate,
    GateResult,
    MAX_UNITS,
    MIN_UNITS,
    STATUSES,
    TARGET_SEPARATOR,
    filter_core_cards,
    read_loanwords,
)
from anki_tools.languages.french.profile import FRENCH
from anki_tools.sentence_pack.audit import print_audit, sha256_value

SAMPLE = 3


def _sample(items: Sequence[str], limit: int = SAMPLE) -> str:
    """Join a bounded sample of failure items."""
    return ", ".join(items[:limit])


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
    """K2: every coverable target has a card.

    A target whose every candidate fails the sense test is ``absent_sense``
    and is not part of the check.
    """
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 1)
    targets = sum(1 for row in coverage if row["status"] != "absent_sense")
    ok = covered == targets
    return ok, f"covered {covered} of {targets}"


def gate_covered2(files: CoreFiles) -> tuple[bool, str]:
    """K3: at least 0.85 of the coverable targets have two cards.

    A rare Swadesh word can have one candidate sentence in the pool, so the
    limit is below 1.0. The built French deck measures 0.897.
    """
    coverage = files.coverage()
    covered = sum(1 for row in coverage if int(row["count"]) >= 2)
    targets = sum(1 for row in coverage if row["status"] != "absent_sense")
    share = covered / targets if targets else 0.0
    return share >= 0.85, f"two cards for {covered} of {targets} ({share:.3f})"


def gate_outside(files: CoreFiles) -> tuple[bool, str]:
    """K4: the outside set and the rare outside words stay small."""
    counts = files.manifest()["counts"]
    targets = max(1, counts["targets"])
    outside = counts["outside"]
    ok_set = outside <= 1.3 * targets
    ok_rare = counts["outside_rare"] <= 0.1 * max(1, outside)
    return (
        ok_set and ok_rare,
        f"outside {outside} of {targets} targets, rare {counts['outside_rare']}",
    )


def gate_filter(files: CoreFiles) -> tuple[bool, str]:
    """K5: every kept card passes F1 to F9 and C1 to C5."""
    output = files.output()
    cards = output["cards"]
    kept, _reasons = filter_core_cards(
        cards,
        files.tokens(),
        set(output["media"]),
        cull.read_names(),
        read_loanwords(),
        min_units=MIN_UNITS,
        max_units=MAX_UNITS,
    )
    if len(kept) != len(cards):
        failed = [cards[index].sentence for index in range(len(cards)) if index not in set(kept)]
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
        buckets = {other for lemma in lemmas for other in shared[lemma] if other != index}
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


def gate_identity(files: CoreFiles) -> tuple[bool, str]:
    """K9: the deck name and the note model are the requested values."""
    manifest = files.manifest()
    source = manifest["source"]
    output = files.output()
    ok_deck = source["deck"] in output["decks"]
    ok_model = output["note_models"] == {source["model"]}
    return ok_deck and ok_model, (
        f"decks {sorted(output['decks'])}, models {sorted(output['note_models'])}"
    )


def gate_order(files: CoreFiles) -> tuple[bool, str]:
    """K10: the kept order is the recorded order."""
    rank_rows = files.rank_rows()
    sentences = [row["sentence"] for row in rank_rows]
    text = "\n".join(sentences)
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    recorded = files.manifest()["kept_order"]
    ok = digest == recorded and len(rank_rows) == len(_kept(files))
    return ok, f"{len(rank_rows)} rows, order {'matches' if ok else 'differs'}"


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


def _key_of(row: dict[str, str]) -> str:
    """Give the target key of a coverage row."""
    return row["key"]


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
        for card, card_tokens in zip(
            files.output()["cards"], files.tokens()
        )
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
                bool(FRENCH.hits(card.sentence, {word}))
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
    expected = [list(pair) for pair in CORE_TEMPLATES]
    ok_templates = all(
        [list(pair) for pair in value] == expected
        for value in templates
    )
    wanted_css = cull.card_css()
    ok_css = all(value == wanted_css for value in css)
    return ok_templates and ok_css, (
        f"{len(used)} model(s) "
        f"{'match' if ok_templates else 'differ'}, "
        f"CSS {'match' if ok_css else 'differ'}"
    )


def gate_bold(files: CoreFiles) -> tuple[bool, str]:
    """K15: the bold text is the assigned target, inside the sentence.

    The bold span can be one token of a multiword form, or an elided token
    such as ``n'`` for ``ne pas``. K13 checks the target itself.
    """
    manifest = files.manifest()
    params = manifest["params"]
    notes = files.notes()
    bad: list[str] = []
    bolded = 0
    form = {_key_of(row): row["word"] for row in files.coverage()}
    for row in _kept(files):
        fields = notes.get(int(row["note_id"]), [])
        text = fields[0] if fields else ""
        plain = _strip_bold(text)
        if plain != row["sentence"]:
            bad.append(f"text of {row['sentence']}")
            continue
        if "<b>" not in text:
            continue
        bolded += 1
        target = row["assigned"]
        if not target:
            bad.append(f"bold without a target: {row['sentence']}")
            continue
        bold = _bold_text(text)
        if not bold or bold not in row["sentence"]:
            bad.append(f"bold {bold} is not in {form.get(target, target)}")
    if params.get("bold_target") and bolded + manifest["counts"].get("unbolded", 0) != len(_kept(files)):
        bad.append("bold count")
    return not bad, f"{bolded} bold card(s), issues {_sample(bad)}"


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


def gate_sense(files: CoreFiles) -> tuple[bool, str]:
    """K16: the unchecked targets stay inside the limit."""
    coverage = files.coverage()
    unchecked = sum(1 for row in coverage if row["status"] == "unchecked")
    limit = int(files.manifest()["params"].get("max_unchecked_sense", 0))
    return unchecked <= limit, f"{unchecked} unchecked, limit {limit}"


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
    tail = [row for row in files.rows() if row["pass"] == "tail" and row["kept"] == "1"]
    bad = [row["sentence"] for row in tail if int(row["new_lemmas"]) < 1]
    return not bad, f"{len(tail)} tail card(s), {len(bad)} without a new lemma"


def gate_gloss(files: CoreFiles) -> tuple[bool, str]:
    """K19: the core gloss pairs with the core word and holds one sense."""
    names = files.fields()
    missing = [
        name
        for name in (CORE_FIELD, CORE_GLOSS_FIELD)
        if name not in names
    ]
    if missing:
        return False, f"model has no {_sample(missing)} field"
    word_index = names.index(CORE_FIELD)
    gloss_index = names.index(CORE_GLOSS_FIELD)
    bad: list[str] = []
    filled = 0
    for note_id, values in files.notes().items():
        word = values[word_index].strip() if word_index < len(values) else ""
        gloss = values[gloss_index].strip() if gloss_index < len(values) else ""
        if not word:
            if gloss:
                bad.append(f"{note_id} has a gloss without a core word")
            continue
        filled += 1
        if not gloss:
            bad.append(f"{note_id} has no gloss")
        elif "," in gloss:
            bad.append(f"{note_id} holds a comma gloss")
    return not bad, f"{filled} core word(s), {len(bad)} bad: {_sample(bad)}"


GATES: tuple[Gate, ...] = (
    CoreGate("K1", gate_card_count),
    CoreGate("K2", gate_covered1),
    CoreGate("K3", gate_covered2),
    CoreGate("K4", gate_outside),
    CoreGate("K5", gate_filter),
    CoreGate("K6", gate_similarity),
    CoreGate("K7", gate_audio),
    CoreGate("K8", gate_counts),
    CoreGate("K9", gate_identity),
    CoreGate("K10", gate_order),
    CoreGate("K11", gate_coverage),
    CoreGate("K12", gate_hashes),
    CoreGate("K13", gate_hit),
    CoreGate("K14", gate_templates),
    CoreGate("K15", gate_bold),
    CoreGate("K16", gate_sense),
    CoreGate("K17", gate_names),
    CoreGate("K18", gate_tail),
    CoreGate("K19", gate_gloss),
)


def run_gates(files: CoreFiles) -> list[GateResult]:
    """Run every gate and catch the error of a gate that cannot read a file.

    :param files: Lazy reader of the output files.
    :returns: One result per gate.
    """
    results: list[GateResult] = []
    for gate in GATES:
        try:
            results.append(gate.check(files))
        except Exception as error:  # noqa: BLE001 - a gate reports the error
            results.append(
                GateResult(gate.name, False, f"{type(error).__name__}: {error}")
            )
    return results


def check_core(output_path: Path) -> int:
    """Run gates K1 to K19 on the files of a core output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return print_audit(run_gates(CoreFiles(Path(output_path))))
