"""Mandarin caller of the shared word table, pool, and selection stages.

The Mandarin word table, the jieba and pinyin helpers, and the profile live in
``anki_tools.languages.mandarin.profile``. This module keeps the names and
the signatures that the CLI, the audit, and the tests use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anki_tools.sentence_pack import guards as shared_guards
from anki_tools.sentence_pack import pool as shared_pool
from anki_tools.sentence_pack.pool import FILTER_KEYS, FILTER_NAMES  # noqa: F401
from anki_tools.sentence_pack import select as shared_select
from anki_tools.sentence_pack import tatoeba
from anki_tools.languages.mandarin.profile import (  # noqa: F401
    AUDIO_DIR,
    AUDIO_TSV_PATH,
    DEFAULT_CSV_PATH,
    DEFAULT_OVERRIDES_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SWADESH_PATH,
    FALLBACK_MIN_HAN,
    FINAL_MARKS,
    MAX_FORMS,
    MIN_FORMS,
    MIN_ZIPF,
    OPEN_STATUS,
    OVERRIDE_FIELDS,
    POOL_REPORT_PATH,
    PREDICATE_TAG_PREFIXES,
    PREFERRED_MAX_HAN,
    PREFERRED_MIN_HAN,
    QUOTE_MARKS,
    REVIEW_FIELDS,
    SENTENCE_FIELDS,
    SENTENCE_REVIEW_FIELDS,
    SENTENCES_PATH,
    SENTENCE_REVIEW_PATH,
    SIMILARITY_LIMIT,
    SWADESH_ROWS,
    Candidate,
    MANDARIN,
    WordForm,
    align_pinyin,
    apply_tone_sandhi,
    build_known_set,
    check_word_table,
    count_form_occurrences,
    count_form_tokens,
    han_tokens,
    has_orphan_conjunction,
    has_predicate,
    load_user_dictionary,
    load_word_table,
    make_candidate,
    read_csv_rows,
    read_pinyin_overrides,
    read_swadesh_labels,
    sentence_hits,
    sentence_pinyin,
    split_forms,
    word_table_report,
    write_sentence_csv,
    write_sentence_review_csv,
)

_zipf = shared_pool.zipf_value


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


def read_english_links(links_path: Path) -> dict[int, list[int]]:
    """Read the Mandarin to English translation links.

    :param links_path: TSV file with columns ``cmn_id,eng_id`` and no header.
    :returns: Mapping from Mandarin sentence id to its English sentence ids.
    """
    return tatoeba.read_links(links_path)


def read_cmn_sentences(sentences_path: Path) -> dict[int, str]:
    """Read the Mandarin sentence texts.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :returns: Mapping from sentence id to the raw sentence text.
    """
    return tatoeba.read_sentences(sentences_path)


def read_cmn_authors(detailed_path: Path) -> dict[int, str]:
    """Read the author of each Mandarin sentence.

    :param detailed_path: TSV file with columns ``id,lang,text,author,...``.
    :returns: Mapping from sentence id to the author name.
    """
    return tatoeba.read_authors(detailed_path)


def read_cmn_audio(audio_path: Path) -> dict[int, tuple[int, str]]:
    """Read the native audio id and license of each Mandarin sentence.

    :param audio_path: TSV file with columns
        ``sentence_id,audio_id,author,license,attribution``.
    :returns: Mapping from sentence id to ``(audio_id, license)``.
    """
    return tatoeba.read_audio(audio_path)


def read_english_sentences(
    sentences_path: Path, wanted_ids: set[int]
) -> dict[int, str]:
    """Stream the English dump and keep the wanted rows.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :param wanted_ids: English sentence ids to keep.
    :returns: Mapping from English sentence id to its text.
    """
    return tatoeba.read_english_sentences(sentences_path, wanted_ids)


def build_candidate_pool(
    word_table: list[WordForm],
    data_dir: Path = Path("data/tatoeba"),
    min_han: int = 4,
    max_han: int = 7,
    max_unknown: int = 2,
    fallback: bool = False,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Build the S2 candidate pool from the Tatoeba exports.

    :param word_table: Word table from :func:`load_word_table`.
    :param data_dir: Directory with the Tatoeba TSV files.
    :param min_han: Minimum Han character count for a core candidate.
    :param max_han: Maximum Han character count.
    :param max_unknown: Unknown token budget per sentence.
    :param fallback: Add scaffold candidates for forms with no core candidate.
    :returns: The candidate list and the pool report.
    """
    load_user_dictionary(word_table)
    known = build_known_set(word_table)
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
    :param word_table: Word table from :func:`load_word_table`.
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


__all__ = [
    "Candidate",
    "MANDARIN",
    "SENTENCE_FIELDS",
    "SENTENCE_REVIEW_FIELDS",
    "SWADESH_ROWS",
    "WordForm",
    "build_candidate_pool",
    "build_known_set",
    "build_pack_report",
    "check_sentence_shape",
    "check_word_table",
    "count_form_occurrences",
    "han_tokens",
    "has_orphan_conjunction",
    "has_predicate",
    "load_word_table",
    "print_sentence_report",
    "rank_key_parts",
    "rank_order",
    "select_sentences",
    "sentence_hits",
    "sentence_pinyin",
    "serialize_rank_key",
    "word_table_report",
    "write_pool_report",
    "write_sentence_csv",
    "write_sentence_review_csv",
]
