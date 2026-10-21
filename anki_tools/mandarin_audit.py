"""Mandarin caller of the shared quality gates (G1 to G18).

The shared module holds the gate runner and the language-neutral gates. This
module adds the Mandarin word-table gate, the pinyin gate, and the native-audio
attribution gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from anki_tools.sentence_pack.audit import (
    AuditInputs,
    GateRow,
    gate_apkg,
    gate_coverage,
    gate_manifest_counts,
    gate_media_count,
    gate_proper_sentences,
    gate_review_file,
    gate_sentence_audio,
    gate_similarity,
    gate_target_once,
    gate_targets_in_table,
    gate_unique_forms,
    gate_unique_tatoeba_ids,
    gate_word_audio,
    print_audit,
    run_gates,
    sample,
)
from anki_tools.sentence_pack.audio import probe_duration
from anki_tools.sentence_pack.languages.mandarin import (
    MANDARIN,
    check_word_table,
    load_word_table,
    false_friend_spans,
    sense_filter,
)
from anki_tools.mandarin_deck import audio_author_map
from anki_tools.rank import get_chinese_chars
from anki_tools.sentence_pack.audio import AUDIO_INDEX_NAME

OUTPUT_PATH = MANDARIN.output_apkg
SENTENCES_PATH = MANDARIN.sentence_csv
WORD_CSV_PATH = MANDARIN.word_csv
REVIEW_PATH = MANDARIN.word_review_csv
AUDIO_DIR = MANDARIN.audio_dir
MANIFEST_PATH = MANDARIN.manifest_path
ATTRIBUTION_PATH = Path("ATTRIBUTION.md")

OPEN_STATUS = "open"
FAILURE_SAMPLE = 3
DETAIL_WIDTH = 110

GateCheck = Callable[[AuditInputs, Any], tuple[bool, str]]


def _gate_g1(inputs: AuditInputs, profile: Any) -> tuple[bool, str]:
    table = inputs.table()
    failures = check_word_table(
        table, csv_path=inputs.word_csv, review_path=inputs.review_csv
    )
    concepts = len({form.swadesh_index for form in table})
    if failures:
        return False, sample(failures)
    return True, f"{concepts} concepts, {len(table)} forms"


def _gate_g9(inputs: AuditInputs, profile: Any) -> tuple[bool, str]:
    rows = inputs.rows()
    bad = []
    for row in rows:
        han_len = len(get_chinese_chars(row["sentence"]))
        syllables = len(row.get("pinyin", "").split())
        if han_len != syllables:
            bad.append(f"{row['sentence']}: {han_len} Han, {syllables} syllables")
    if bad:
        return False, f"{len(bad)} card(s) mismatch pinyin: {sample(bad)}"
    return True, f"{len(rows)} cards match pinyin syllables"


def _audio_id(row: dict[str, str]) -> int:
    try:
        return int(str(row.get("audio_id", "")).strip() or 0)
    except ValueError:
        return 0


def _gate_g12(inputs: AuditInputs, profile: Any) -> tuple[bool, str]:
    path = inputs.attribution_path
    if not path.is_file():
        return False, f"attribution file missing: {path}"
    text = path.read_text(encoding="utf-8")
    native = [
        row for row in inputs.rows() if row.get("audio_source") == "native"
    ]
    if not native:
        return True, "no native audio; the attribution file exists"
    authors = audio_author_map()
    names = {
        authors.get(_audio_id(row)) or row.get("tatoeba_author", "").strip()
        for row in native
    }
    missing = sorted(name for name in names if name and name not in text)
    if missing:
        return False, (
            f"{len(missing)} native audio author(s) not in {path}: {sample(missing)}"
        )
    return True, f"{len(names)} native audio author(s) named in {path}"


def _gate_g17(inputs: AuditInputs, profile: Any) -> tuple[bool, str]:
    """Gate G17: no card uses its target in a different sense.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        f"{row.get('target', '')} in {row.get('sentence', '')}"
        for row in rows
        if false_friend_spans(row.get("sentence", ""), row.get("target", ""))
    ]
    if bad:
        return False, (
            f"{len(bad)} card(s) use the target in a different sense: "
            f"{sample(bad)}"
        )
    return True, f"{len(rows)} cards use each target in its Swadesh sense"


def _gate_g18(inputs: AuditInputs, profile: Any) -> tuple[bool, str]:
    """Gate G18: no card English holds a false-friend sense.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        f"{row.get('target', '')} in {row.get('sentence', '')}"
        for row in rows
        if sense_filter(row.get("target", ""), row.get("english", ""))
    ]
    if bad:
        return False, (
            f"{len(bad)} card(s) use the target in a different sense: "
            f"{sample(bad)}"
        )
    return True, f"{len(rows)} cards use each target in its Swadesh sense"


GATES: tuple[tuple[str, GateCheck], ...] = (
    ("G1", _gate_g1),
    ("G2", gate_unique_forms),
    ("G3", gate_targets_in_table),
    ("G4", gate_unique_tatoeba_ids),
    ("G5", gate_target_once),
    ("G6", gate_similarity),
    ("G7", gate_sentence_audio),
    ("G8", gate_media_count),
    ("G9", _gate_g9),
    ("G10", gate_coverage),
    ("G11", gate_manifest_counts),
    ("G12", _gate_g12),
    ("G13", gate_apkg),
    ("G14", gate_proper_sentences),
    ("G15", gate_review_file),
    ("G16", gate_word_audio),
    ("G17", _gate_g17),
    ("G18", _gate_g18),
)


def run_audit(
    apkg_path: Path = OUTPUT_PATH,
    sentences_csv: Path = SENTENCES_PATH,
    word_csv: Path = WORD_CSV_PATH,
    review_csv: Path = REVIEW_PATH,
    audio_dir: Path = AUDIO_DIR,
    manifest_path: Path = MANIFEST_PATH,
    attribution_path: Path = ATTRIBUTION_PATH,
) -> list[GateRow]:
    """Run the quality gates G1 to G18.
    :param apkg_path: APKG file of the pack.
    :param sentences_csv: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param review_csv: Word review CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param manifest_path: Manifest JSON file.
    :param attribution_path: Attribution Markdown file.
    :returns: One ``(gate, passed, message)`` row per gate, G1 first.
    """
    inputs = AuditInputs(
        MANDARIN,
        apkg_path=apkg_path,
        sentences_csv=sentences_csv,
        word_csv=word_csv,
        review_csv=review_csv,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        attribution_path=attribution_path,
        load_table=load_word_table,
        probe=probe_duration,
    )
    return run_gates(GATES, inputs, MANDARIN)


__all__ = [
    "AUDIO_INDEX_NAME",
    "DETAIL_WIDTH",
    "FAILURE_SAMPLE",
    "GATES",
    "OPEN_STATUS",
    "print_audit",
    "run_audit",
]
