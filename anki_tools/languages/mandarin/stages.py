"""Mandarin bindings of the shared sentence-pack stages.

The shared stage modules hold the logic: :mod:`anki_tools.sentence_pack` for the
pool, the selection, the audio, the APKG, the audit, and the Tatoeba cache. This
module binds the Mandarin profile to each stage and keeps the Mandarin gate
functions and the module-level patch points that the tests use.

A binding carries the Mandarin name, for example
:func:`select_mandarin_sentences` or :func:`mandarin_card`, so one name never
resolves to two modules. A stage that takes no Mandarin value is imported from
its shared module where the caller lives.

The stages are S0 (fetch), S2 (pool), S3 (selection), S4 (audio), and S5 (pack).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from anki_tools.languages.mandarin.profile import (
    AUDIO_DIR,
    AUDIO_TSV_PATH,
    BACK_TEMPLATE,
    DEFAULT_CSV_PATH,
    DECK_DESCRIPTION,
    FIELDS as _PROFILE_FIELDS,
    FRONT_TEMPLATE,
    MANDARIN,
    POOL_REPORT_PATH,
    SENTENCES_PATH,
    SENTENCE_REVIEW_PATH,
    SORT_FIELD_INDEX,
    Candidate,
    WordForm,
    known_words,
    build_attribution,
    check_word_table,
    load_user_dictionary,
    load_word_table,
    make_candidate,
    write_attribution,
    write_sentence_csv,
    write_sentence_review_csv,
)
from anki_tools.languages.mandarin.characters import get_chinese_chars
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack import audio as shared_audio
from anki_tools.sentence_pack import guards as shared_guards
from anki_tools.sentence_pack import pack as shared_pack
from anki_tools.sentence_pack import pool as shared_pool
from anki_tools.sentence_pack import select as shared_select
from anki_tools.sentence_pack import tatoeba
from anki_tools.sentence_pack.audio import AUDIO_INDEX_NAME, probe_duration
from anki_tools.sentence_pack.audit import (
    AuditInputs,
    GateRow,
    gate_apkg,
    gate_coverage,
    gate_manifest_counts,
    gate_media_count,
    gate_no_false_friend,
    gate_no_sense_mismatch,
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
    run_pack_audit,
    sample,
)
from anki_tools.sentence_pack.pack import (
    AudioIndex,
    audio_author_map as _audio_author_map,
    audio_license_source as _audio_license_source,
    load_audio_index,
    media_paths,
    skipped_license_counts,
    slug,
    tts_engine_name,
)

# S0: Tatoeba fetch
TATOEBA_BASE = tatoeba.TATOEBA_BASE
TATOEBA_FILES: dict[str, str] = tatoeba.export_files("cmn")
CHECKSUMS_NAME = tatoeba.CHECKSUMS_NAME
CHUNK_SIZE = tatoeba.CHUNK_SIZE
DOWNLOAD_ATTEMPTS = tatoeba.DOWNLOAD_ATTEMPTS
DOWNLOAD_TIMEOUT = tatoeba.DOWNLOAD_TIMEOUT

_download = tatoeba.download
_sha256_file = tatoeba.sha256_file
_decompress = tatoeba._decompress
_load_checksums = tatoeba._load_checksums
_write_checksums = tatoeba._write_checksums
_is_cached = tatoeba._is_cached

# S4: audio
DEFAULT_VOICE = MANDARIN.voice
GTTS_VOICE = MANDARIN.gtts_voice
GTTS_LANG = MANDARIN.tts_lang
RATE_NORMAL = shared_audio.RATE_NORMAL
RATE_SLOW = shared_audio.RATE_SLOW
SLOW_RATE = shared_audio.RATE_SLOW
MIN_DURATION_S = shared_audio.MIN_DURATION_S
MAX_DURATION_S = shared_audio.MAX_DURATION_S
MAX_ATTEMPTS = shared_audio.MAX_ATTEMPTS
RETRY_DELAYS = shared_audio.RETRY_DELAYS
PUBLIC_PACK = shared_audio.PUBLIC_PACK
PRIVATE_PACK = shared_audio.PRIVATE_PACK
NATIVE_LICENSE_BLOCK = shared_audio.NATIVE_LICENSE_BLOCK
INDEX_FIELDS = shared_audio.INDEX_FIELDS
time = shared_audio.time

# S5: APKG pack
OUTPUT_PATH = default_output(MANDARIN.name, "pack")
MANIFEST_PATH = DeckIdentity(OUTPUT_PATH).side("manifest.json")
ATTRIBUTION_PATH = shared_pack.ATTRIBUTION_PATH
CSS_PATH = shared_pack.CSS_PATH
CHECKSUMS_PATH = shared_pack.CHECKSUMS_PATH
WORD_PREFIX = shared_pack.WORD_PREFIX
SENTENCE_PREFIXES = shared_pack.SENTENCE_PREFIXES
LICENSE_KEYS = shared_pack.LICENSE_KEYS

FIELDS = list(_PROFILE_FIELDS)
DEFAULT_AUDIO_LICENSE_SOURCE = {key: 0 for key in LICENSE_KEYS}

# Audit
WORD_CSV_PATH = MANDARIN.word_csv
REVIEW_PATH = MANDARIN.word_review_csv
OPEN_STATUS = "open"
FAILURE_SAMPLE = 3
DETAIL_WIDTH = 110

GateCheck = Callable[[AuditInputs, Any], tuple[bool, str]]


def mandarin_candidate_pool(
    word_table: list[WordForm],
    data_dir: Path = Path("data/tatoeba"),
    min_han: int = 4,
    max_han: int = 7,
    max_unknown: int = 2,
    fallback: bool = False,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Build the S2 candidate pool from the Tatoeba exports.

    :param word_table: Word table from ``load_word_table``.
    :param data_dir: Directory with the Tatoeba TSV files.
    :param min_han: Minimum Han character count for a core candidate.
    :param max_han: Maximum Han character count.
    :param max_unknown: Unknown token budget per sentence.
    :param fallback: Add scaffold candidates for forms with no core candidate.
    :returns: The candidate list and the pool report.
    """
    load_user_dictionary(word_table)
    known = known_words(word_table)
    return shared_pool.build_candidate_pool(
        word_table,
        MANDARIN,
        data_dir=data_dir,
        min_units=min_han,
        max_units=max_han,
        max_unknown=max_unknown,
        fallback=fallback,
        known_words=known,
        make_candidate=make_candidate,
    )


def mandarin_shape_failure(
    sentence: str, min_han: int = 4, max_han: int = 7, *, final_mark: bool = True
) -> str | None:
    """Apply filters F2 to F8 to one Mandarin sentence.

    :param sentence: Simplified Chinese sentence.
    :param min_han: Minimum Han character count.
    :param max_han: Maximum Han character count.
    :param final_mark: Apply filters F5 and F6 (the final-mark rules).
    :returns: The first failed filter key, or None when all filters pass.
    """
    return shared_guards.check_sentence_shape(
        sentence, MANDARIN, min_han, max_han, final_mark=final_mark
    )


def mandarin_rank_key_parts(candidate: Candidate) -> shared_select.RankKey:
    """Build the sort key of a candidate, first item first.

    :param candidate: Candidate sentence.
    :returns: Unknown count, audio penalty, negated Zipf, length penalty, id.
    """
    return shared_select.rank_key_parts(candidate, MANDARIN)


def mandarin_rank_order(candidate: Candidate) -> shared_select.RankKey:
    """Sort candidates with the ranking key.

    :param candidate: Candidate sentence.
    :returns: Sort key, lowest value first.
    """
    return shared_select.rank_order(candidate, MANDARIN)


def select_mandarin_sentences(
    pool: list[Candidate],
    word_table: list[WordForm],
    per_word: int = 2,
    max_cards: int = 450,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select cards with the greedy set cover of the Mandarin spec.

    :param pool: Candidate list from :func:`mandarin_candidate_pool`.
    :param word_table: Word table from ``load_word_table``.
    :param per_word: Card limit per form.
    :param max_cards: Hard card cap.
    :returns: The cards in Swadesh order and the selection report.
    """
    return shared_select.select_sentences(
        pool,
        word_table,
        MANDARIN,
        per_word=per_word,
        max_cards=max_cards,
        finalize=MANDARIN.finalize_candidate,
    )


def audio_license_source(tsv_path: Path = AUDIO_TSV_PATH) -> dict[str, int]:
    """Count the license values of the Tatoeba Mandarin audio export.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Empty-license and CC BY-NC 4.0 counts of the whole export.
    """
    return _audio_license_source(tsv_path)


def audio_author_map(tsv_path: Path = AUDIO_TSV_PATH) -> dict[int, str]:
    """Read the author of each Tatoeba audio id.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Mapping from audio id to author.
    """
    return _audio_author_map(tsv_path)


def mandarin_card(
    row: dict[str, str],
    word: Any,
    sentence_audio: str,
    word_audio: str,
) -> dict[str, Any]:
    """Build the note fields and tags for one Mandarin card.

    :param row: One row of the sentence CSV.
    :param word: Reviewed word form of the card target.
    :param sentence_audio: File name of the sentence MP3.
    :param word_audio: File name of the word MP3.
    :returns: Mapping from field name to value, plus a ``tags`` key.
    """
    return shared_pack.build_card(row, word, sentence_audio, word_audio, MANDARIN)


def build_mandarin_pack(
    csv_path: Path = SENTENCES_PATH,
    word_csv: Path = DEFAULT_CSV_PATH,
    audio_dir: Path = AUDIO_DIR,
    output: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    attribution_path: Path = ATTRIBUTION_PATH,
    pack_license: str = shared_pack.PUBLIC_PACK,
) -> dict[str, Any]:
    """Build the APKG file, the manifest, and the attribution file.

    :param csv_path: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param output: Path of the APKG file to write.
    :param manifest_path: Path of the manifest JSON file to write.
    :param attribution_path: Path of the attribution Markdown file to write.
    :param pack_license: ``public`` or ``private``.
    :raises FileNotFoundError: The audio index or a media file is missing.
    :returns: The written paths, the media count, the skipped card count, and
        the manifest mapping.
    """
    result = shared_pack.build_pack(
        MANDARIN,
        load_word_table,
        csv_path=csv_path,
        word_csv=word_csv,
        audio_dir=audio_dir,
        output=output,
        manifest_path=manifest_path,
        attribution_path=attribution_path,
        pack_license=pack_license,
    )
    result["attribution_path"] = str(attribution_path)
    return result


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
    native = [row for row in inputs.rows() if row.get("audio_source") == "native"]
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
    ("G17", gate_no_false_friend),
    ("G18", gate_no_sense_mismatch),
)


def run_audit(**paths: Path | None) -> list[GateRow]:
    """Run the quality gates G1 to G18.

    :param paths: Input paths that replace the defaults of the profile.
    :returns: One ``(gate, passed, message)`` row per gate, G1 first.
    """
    return run_pack_audit(
        MANDARIN,
        GATES,
        load_table=load_word_table,
        probe=probe_duration,
        **paths,
    )


__all__ = [
    "ATTRIBUTION_PATH",
    "AUDIO_DIR",
    "AUDIO_INDEX_NAME",
    "AudioIndex",
    "BACK_TEMPLATE",
    "CSS_PATH",
    "Candidate",
    "DECK_DESCRIPTION",
    "DEFAULT_VOICE",
    "DETAIL_WIDTH",
    "FAILURE_SAMPLE",
    "FIELDS",
    "FRONT_TEMPLATE",
    "GATES",
    "GTTS_LANG",
    "GTTS_VOICE",
    "MANDARIN",
    "MANIFEST_PATH",
    "OPEN_STATUS",
    "OUTPUT_PATH",
    "POOL_REPORT_PATH",
    "SENTENCES_PATH",
    "SENTENCE_REVIEW_PATH",
    "SORT_FIELD_INDEX",
    "audio_author_map",
    "audio_license_source",
    "build_attribution",
    "build_mandarin_pack",
    "load_audio_index",
    "load_word_table",
    "mandarin_card",
    "mandarin_candidate_pool",
    "mandarin_rank_key_parts",
    "mandarin_rank_order",
    "mandarin_shape_failure",
    "media_paths",
    "print_audit",
    "run_audit",
    "select_mandarin_sentences",
    "skipped_license_counts",
    "slug",
    "tts_engine_name",
    "write_attribution",
    "write_sentence_csv",
    "write_sentence_review_csv",
]

MediaFiles = Iterable[str]
Composer = Callable[[], dict[str, Any]]
Cards = Sequence[dict[str, Any]]
