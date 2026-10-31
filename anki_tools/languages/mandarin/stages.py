"""Mandarin callers of the shared sentence-pack stages.

The shared stage modules hold the logic: :mod:`anki_tools.sentence_pack` for the
pool, the selection, the audio, the APKG, the audit, and the Tatoeba cache. This
module binds the Mandarin profile to each stage and keeps the Mandarin gate
functions and the module-level patch points that the tests use.

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
    EXTRA_CSS,
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
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.apkg.templates import load_css as load_css_file
from anki_tools.languages.mandarin.characters import get_chinese_chars
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
tts_engine = shared_audio.tts_engine
synthesize_tts_shared = shared_audio.synthesize_tts

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


def fetch_tatoeba(
    dest: Path = Path("data/tatoeba"), refresh: bool = False
) -> dict[str, dict[str, Any]]:
    """Fetch the Tatoeba exports for the Mandarin Swadesh sentence pack.

    A file is skipped when its TSV file exists and its sha256 matches the value
    in ``checksums.json``. The checksum file is written after each file, so an
    interrupted run keeps its progress.

    :param dest: Cache directory for the TSV files and ``checksums.json``.
    :param refresh: Ignore the cache and download every file again.
    :returns: Per-file record with ``rows`` and ``source`` (``cache`` or
        ``network``).
    """
    return tatoeba.fetch_tatoeba(
        TATOEBA_FILES, Path(dest), refresh=refresh, download_file=_download
    )


def load_sentence_rows(path: Path = SENTENCES_PATH) -> list[dict[str, str]]:
    """Read the Mandarin sentence CSV.

    :param path: Sentence CSV file.
    :returns: One dict per card row.
    """
    return shared_audio.load_sentence_rows(path)


def validate_audio(path: Path) -> tuple[bool, float]:
    """Validate one audio file with the patchable ffprobe reader.

    :param path: Media file.
    :returns: ``(True, duration)`` when the file is valid, else ``(False, 0.0)``.
    """
    return shared_audio.validate_audio(path, probe_duration)


def effective_voice(voice: str) -> str:
    """Select the voice name that goes into the hash and the index.

    :param voice: Requested edge-tts voice.
    :returns: The requested voice, or ``gtts:zh-CN`` for the gTTS engine.
    """
    return shared_audio.effective_voice(voice, MANDARIN, tts_engine())


def choose_source(
    row: dict[str, str], native_audio: str, pack_license: str
) -> str:
    """Choose the audio tier for one card row.

    :param row: One sentence CSV row.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param pack_license: ``public`` or ``private``.
    :returns: ``native`` or ``tts``.
    """
    return shared_audio.choose_source(row, native_audio, pack_license, MANDARIN)


def synthesize_tts(
    text: str,
    dest: Path,
    voice: str,
    rate: str,
    slow: bool,
    profile: Any = None,
) -> None:
    """Write one Mandarin TTS file.

    :param text: Text that goes to the engine.
    :param dest: Destination MP3 file.
    :param voice: Requested edge-tts voice.
    :param rate: Engine rate string.
    :param slow: Slow flag for the gTTS engine.
    :param profile: Language profile from the shared caller, MANDARIN by default.
    """
    synthesize_tts_shared(text, dest, voice, rate, slow, profile or MANDARIN)


def generate_audio(
    rows: Sequence[dict[str, str]],
    audio_dir: Path = AUDIO_DIR,
    voice: str = DEFAULT_VOICE,
    native_audio: str = "none",
    slow_audio: bool = False,
    pack_license: str = PUBLIC_PACK,
    rate_limit: float = 1.0,
    refresh: bool = False,
    csv_path: Path | None = SENTENCES_PATH,
) -> dict[str, Any]:
    """Generate and validate the Mandarin sentence and word audio files.

    :param rows: Sentence CSV rows from :func:`load_sentence_rows`.
    :param audio_dir: Output directory for the MP3 files and the index.
    :param voice: Requested edge-tts voice.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param slow_audio: Generate the slow variant.
    :param pack_license: ``public`` or ``private``.
    :param rate_limit: Minimum seconds between two requests.
    :param refresh: Ignore the existing files and generate them again.
    :param csv_path: Sentence CSV to rewrite, or None to skip the rewrite.
    :returns: Counts, failures, total duration, and the written paths.
    """
    return shared_audio.generate_audio(
        rows,
        MANDARIN,
        audio_dir=audio_dir,
        voice=voice,
        native_audio=native_audio,
        slow_audio=slow_audio,
        pack_license=pack_license,
        rate_limit=rate_limit,
        refresh=refresh,
        csv_path=csv_path,
        engine=tts_engine(),
        synthesize=synthesize_tts,
        probe=probe_duration,
    )


def check_sentence_shape(
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


def build_candidate_pool(
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


def rank_key_parts(candidate: Candidate) -> shared_select.RankKey:
    """Build the sort key of a candidate, first item first.

    :param candidate: Candidate sentence.
    :returns: Unknown count, audio penalty, negated Zipf, length penalty, id.
    """
    return shared_select.rank_key_parts(candidate, MANDARIN)


def rank_order(candidate: Candidate) -> shared_select.RankKey:
    """Sort candidates with the ranking key.

    :param candidate: Candidate sentence.
    :returns: Sort key, lowest value first.
    """
    return shared_select.rank_order(candidate, MANDARIN)


def serialize_rank_key(parts: shared_select.RankKey) -> str:
    """Serialize a ranking key tuple into the CSV string.

    :param parts: Tuple from :func:`rank_key_parts`.
    :returns: Pipe-separated key, for example ``0|1|5.120|0|123``.
    """
    return shared_select.serialize_rank_key(parts)


def select_sentences(
    pool: list[Candidate],
    word_table: list[WordForm],
    per_word: int = 2,
    max_cards: int = 450,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select cards with the greedy set cover of the Mandarin spec.

    :param pool: Candidate list from :func:`build_candidate_pool`.
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


def build_pack_report(
    pool_report: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    """Merge the pool report and the selection report.

    :param pool_report: Report from :func:`build_candidate_pool`.
    :param selection: Report from :func:`select_sentences`.
    :returns: One mapping with the counts, the cards, and the coverage.
    """
    return shared_pool.build_pack_report(pool_report, selection)


def write_pool_report(
    report: dict[str, Any], path: Path = POOL_REPORT_PATH
) -> Path:
    """Write the pool report as JSON.

    :param report: Mapping from :func:`build_pack_report`.
    :param path: Output JSON file.
    :returns: The written path.
    """
    return shared_pool.write_pool_report(report, path)


def print_sentence_report(report: dict[str, Any]) -> None:
    """Print the filter counts, the card count, and the coverage.

    :param report: Mapping from :func:`build_pack_report`.
    """
    shared_pool.print_sentence_report(report)


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


def build_card(
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


def card_sort_key(word: Any, row: dict[str, str]) -> tuple:
    """Build the note order key of one card.

    :param word: Reviewed word form of the card target.
    :param row: One row of the sentence CSV.
    :returns: Swadesh index, form variant rank, then Tatoeba id.
    """
    return shared_pack.card_sort_key(word, row, MANDARIN)


def load_css(css_path: Path = CSS_PATH) -> str:
    """Build the card CSS from the shared sheet and the Mandarin classes.

    :param css_path: Base CSS file.
    :returns: Base CSS followed by the pack classes.
    """
    return load_css_file(css_path, EXTRA_CSS)


def build_pack(
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
    "build_card",
    "build_candidate_pool",
    "build_attribution",
    "build_pack",
    "build_pack_report",
    "card_sort_key",
    "check_sentence_shape",
    "choose_source",
    "effective_voice",
    "fetch_tatoeba",
    "generate_audio",
    "load_audio_index",
    "load_css",
    "load_sentence_rows",
    "load_word_table",
    "media_paths",
    "print_audit",
    "print_sentence_report",
    "rank_key_parts",
    "rank_order",
    "run_audit",
    "select_sentences",
    "serialize_rank_key",
    "skipped_license_counts",
    "slug",
    "synthesize_tts",
    "tts_engine_name",
    "validate_audio",
    "write_attribution",
    "write_pool_report",
    "write_sentence_csv",
    "write_sentence_review_csv",
]

MediaFiles = Iterable[str]
Composer = Callable[[], dict[str, Any]]
Cards = Sequence[dict[str, Any]]
