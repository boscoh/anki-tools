"""Shared sentence-card row builders and CSV writers.

The language profiles build the same sentence and review CSV files. The only
differences are the unit column (``han_len`` or ``word_len``), the reading
column (``pinyin`` or ``jyutping``) when one exists, and the language-specific
hooks. This module holds that common plumbing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

from anki_tools.sentence_pack.profile import LanguageProfile
from anki_tools.sentence_pack.select import review_candidates, review_status
from anki_tools.sentence_pack.word_table import write_csv


def build_sentence_row(
    card: Any,
    *,
    unit_field: str,
    extra_fields: Sequence[tuple[str, Callable[[Any], Any]]] = (),
) -> dict[str, Any]:
    """Build one sentence CSV row from a card.

    :param card: Selected card.
    :param unit_field: Name of the unit-count column, ``han_len`` or ``word_len``.
    :param extra_fields: Extra column names and value functions, in CSV order.
    :returns: Mapping from CSV column to value.
    """
    row: dict[str, Any] = {
        "target": card.target,
        "sentence": card.sentence,
        "text_original": card.text_original,
        "english": card.english,
        "tatoeba_id": card.tatoeba_id,
        "tatoeba_author": card.tatoeba_author,
        "english_id": card.english_id,
        "audio_id": card.audio_id,
        "audio_source": card.audio_source,
        "audio_license": card.audio_license,
        "unknown_words": card.unknown_words,
        unit_field: getattr(card, unit_field),
        "zipf_mean": f"{card.zipf_mean:.2f}",
        "rank_key": card.rank_key,
        "tier": card.tier,
        "audio_missing": "true" if card.audio_missing else "false",
    }
    for name, get_value in extra_fields:
        row[name] = get_value(card)
    return row


def write_sentence_rows(
    cards: Sequence[Any],
    path: Path,
    fields: Sequence[str],
    row_builder: Callable[[Any], dict[str, Any]],
) -> Path:
    """Write card rows to a sentence CSV file.

    :param cards: Cards in output order.
    :param path: Output CSV file.
    :param fields: Column names of the file.
    :param row_builder: Function that turns one card into a row mapping.
    :returns: The written path.
    """
    return write_csv(path, fields, [row_builder(card) for card in cards])


def write_review_rows(
    profile: LanguageProfile,
    pool: Sequence[Any],
    word_table: Sequence[Any],
    path: Path,
    fields: Sequence[str],
    form_key: Callable[[Any], str],
    unit_field: str,
    *,
    zipf_min: float,
    per_word: int = 2,
    top_n: int = 3,
    transliteration_field: str | None = None,
    transliteration: Callable[[Any], str] | None = None,
) -> Path:
    """Write the review rows for the weak words of one pack.

    :param profile: Language profile.
    :param pool: Candidate list from the pool stage.
    :param word_table: Reviewed word table.
    :param path: Output CSV file.
    :param fields: Column names of the file.
    :param form_key: Function that gives the target form of one word.
    :param unit_field: Name of the unit-count column.
    :param zipf_min: Zipf value below which a form is rare.
    :param per_word: Card limit per form, used for the thin test.
    :param top_n: Candidate rows per word.
    :param transliteration_field: Name of the reading column, when present.
    :param transliteration: Fallback reader of a sentence, when present.
    :returns: The written path.
    """
    pool_counts, core_counts, by_form = review_candidates(pool, profile)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for form in word_table:
            key = form_key(form)
            status = review_status(
                form, pool_counts[key], core_counts[key], per_word, zipf_min
            )
            if status is None:
                continue
            base = {
                "target": key,
                "swadesh_index": form.swadesh_index,
                "english": form.english,
                "variant": form.variant,
                "status": status,
                "zipf": f"{form.zipf:.2f}",
                "candidate_count": pool_counts[key],
                "scaffold_candidates": pool_counts[key] - core_counts[key],
            }
            candidates = by_form.get(key, [])[:top_n]
            if not candidates:
                writer.writerow({**base, "rank": 0})
                continue
            for rank, candidate in enumerate(candidates, start=1):
                row: dict[str, Any] = {
                    **base,
                    "rank": rank,
                    "sentence": candidate.sentence,
                    "text_original": candidate.text_original,
                }
                if transliteration_field is not None:
                    value = getattr(candidate, transliteration_field)
                    if not value and transliteration is not None:
                        value = transliteration(candidate.sentence)
                    row[transliteration_field] = value
                row.update(
                    {
                        "english_text": candidate.english,
                        "tatoeba_id": candidate.tatoeba_id,
                        "unknown_words": candidate.unknown_words,
                        unit_field: getattr(candidate, unit_field),
                        "rank_key": candidate.rank_key,
                        "tier": candidate.tier,
                    }
                )
                writer.writerow(row)
    return path
