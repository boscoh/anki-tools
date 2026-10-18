"""Word table, candidate pool, and sentence selection for the Mandarin pack.

Stage S1 reads the reviewed ``vocab/english-mandarin.csv`` file, expands each
slash-separated row into one form, and checks the result against
``vocab/swadesh.txt``. Stage S2 turns the Tatoeba exports into a candidate
sentence pool. Stage S3 selects the cards with greedy set cover and writes
``vocab/mandarin_sentences.csv`` and ``vocab/mandarin_sentences_review.csv``.
The module reads the review files and never writes them.

Pinyin limit: :func:`sentence_pinyin` applies the two common 不 and 一 tone
sandhi rules only. It does not apply third-tone sandhi, and it does not know
when 一 is an ordinal number.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import jieba
import jieba.posseg as posseg
import opencc
import pypinyin
from pypinyin.contrib.tone_convert import to_tone, tone_to_tone3
from wordfreq import top_n_list, zipf_frequency

from anki_tools.rank import (
    char_similarity_zh,
    get_chinese_chars,
    is_chinese_char,
    is_repeated_zh_sentence,
)

DEFAULT_CSV_PATH = Path("vocab/english-mandarin.csv")
DEFAULT_SWADESH_PATH = Path("vocab/swadesh.txt")
DEFAULT_OVERRIDES_PATH = Path("vocab/mandarin_pinyin_overrides.csv")
DEFAULT_REVIEW_PATH = Path("vocab/swadesh_review.csv")

SWADESH_ROWS = 207
MIN_FORMS = 205
MAX_FORMS = 220
MIN_ZIPF = 3.0
OPEN_STATUS = "open"

DEFAULT_DATA_DIR = Path("data/tatoeba")
CMN_SENTENCES_FILE = "cmn_sentences.tsv"
CMN_LINKS_FILE = "cmn-eng_links.tsv"
CMN_DETAILED_FILE = "cmn_sentences_detailed.tsv"
CMN_AUDIO_FILE = "cmn_sentences_with_audio.tsv"
ENG_SENTENCES_FILE = "eng_sentences.tsv"
SENTENCES_PATH = Path("vocab/mandarin_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/mandarin_sentences_review.csv")
POOL_REPORT_PATH = Path("data/mandarin_pool_report.json")

FILTER_KEYS = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "unknown",
]
FILTER_NAMES = {
    "F1": "english link",
    "F2": "han length window",
    "F3": "no latin letters or digits",
    "F4": "no repeated pattern",
    "F5": "final mark",
    "F6": "one final mark",
    "F7": "no quotation mark",
    "F8": "at most one internal mark",
    "F9": "predicate tag",
    "F10": "no orphan conjunction",
    "F11": "english word count",
    "F12": "unique simplified text",
    "unknown": "unknown token budget",
}

FINAL_MARKS = "。？！.!?"
QUOTE_MARKS = '"“”「」『』'
INTERNAL_MARKS = "，：；,:;"
ORPHAN_CONJUNCTIONS = (
    "因为",
    "所以",
    "但是",
    "可是",
    "而且",
    "然后",
    "不过",
    "虽然",
)
PREDICATE_TAG_PREFIXES = ("v", "a")
MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12
PREFERRED_MIN_HAN = 5
PREFERRED_MAX_HAN = 9
FALLBACK_MIN_HAN = 3
KNOWN_WORDS_TOP_N = 1000
SIMILARITY_LIMIT = 0.67

SENTENCE_FIELDS = [
    "target",
    "sentence",
    "text_original",
    "pinyin",
    "english",
    "tatoeba_id",
    "tatoeba_author",
    "english_id",
    "audio_id",
    "audio_source",
    "audio_license",
    "unknown_words",
    "han_len",
    "zipf_mean",
    "rank_key",
    "tier",
    "audio_missing",
]
SENTENCE_REVIEW_FIELDS = [
    "target",
    "swadesh_index",
    "english",
    "variant",
    "status",
    "zipf",
    "candidate_count",
    "scaffold_candidates",
    "rank",
    "sentence",
    "text_original",
    "pinyin",
    "english_text",
    "tatoeba_id",
    "unknown_words",
    "han_len",
    "rank_key",
    "tier",
]

REVIEW_FIELDS = [
    "lang",
    "index",
    "english",
    "current",
    "canonical",
    "issue",
    "fix",
    "status",
]
OVERRIDE_FIELDS = ["mandarin", "pinyin", "reason"]

_converter: opencc.OpenCC | None = None
_t2s_converter: opencc.OpenCC | None = None


@dataclass
class WordForm:
    """One reviewed Mandarin form of a Swadesh concept.

    :param swadesh_index: CSV row order, 1-based.
    :param english: Swadesh English label of the row.
    :param mandarin: Simplified form, the unique key of the row.
    :param pinyin: Tone-marked pinyin of the form.
    :param traditional: Traditional form from OpenCC ``s2t``.
    :param variant: ``primary`` for the first form of a row, else ``alternate``.
    :param zipf: ``wordfreq`` Zipf value of the form for ``zh``.
    :param candidate_count: Sentence count after the S2 filters.
    :param tier: ``core`` or ``scaffold``.
    """

    swadesh_index: int
    english: str
    mandarin: str
    pinyin: str
    traditional: str
    variant: str
    zipf: float
    candidate_count: int = 0
    tier: str = "core"


def _traditional_converter() -> opencc.OpenCC:
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC("s2t")
    return _converter


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of stripped row mappings.

    :param csv_path: CSV file to read.
    :returns: One mapping per data row, with stripped values.
    """
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_swadesh_labels(swadesh_path: Path = DEFAULT_SWADESH_PATH) -> list[str]:
    """Read the Swadesh English labels in index order.

    :param swadesh_path: File with one English label per line.
    :returns: Label list, index 1 at position 0.
    """
    text = swadesh_path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def read_pinyin_overrides(
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> dict[str, str]:
    """Read the manual pinyin overrides.

    :param overrides_path: CSV file with columns ``mandarin,pinyin,reason``.
    :returns: Mapping from form to its reviewed pinyin. Missing file gives an
        empty mapping.
    """
    if not overrides_path.exists():
        return {}
    overrides: dict[str, str] = {}
    for row in _read_csv_rows(overrides_path):
        form = row.get("mandarin", "")
        pinyin = row.get("pinyin", "")
        if form and pinyin:
            overrides[form] = pinyin
    return overrides


def split_forms(mandarin: str, index: int) -> list[str]:
    """Split a slash-separated ``mandarin`` cell into single forms.

    :param mandarin: Raw cell value, for example ``老/旧``.
    :param index: Swadesh index, used in the error message.
    :raises ValueError: A form is empty, or the cell is empty.
    :returns: One form per slash-separated part.
    """
    forms = [part.strip() for part in mandarin.split("/")]
    if not forms or any(not form for form in forms):
        raise ValueError(f"row {index}: bad form cell {mandarin!r}")
    return forms


def align_pinyin(forms: list[str], transliteration: str, index: int) -> list[str]:
    """Assign the syllables of a transliteration cell to the forms.

    Each form takes as many syllables as it has Han characters, in order.

    :param forms: Forms of the row, in file order.
    :param transliteration: Space-separated pinyin of the whole row.
    :param index: Swadesh index, used in the error message.
    :raises ValueError: The syllable count does not equal the Han count.
    :returns: One pinyin string per form.
    """
    syllables = transliteration.split()
    han_count = sum(len(form) for form in forms)
    if han_count != len(syllables):
        raise ValueError(
            f"row {index}: {han_count} Han characters but {len(syllables)} "
            f"syllables in {transliteration!r}"
        )
    aligned: list[str] = []
    start = 0
    for form in forms:
        stop = start + len(form)
        aligned.append(" ".join(syllables[start:stop]))
        start = stop
    return aligned


def _row_forms(
    rows: list[dict[str, str]], overrides: dict[str, str]
) -> list[tuple[int, str, str, str, str]]:
    """Expand CSV rows into ``(index, english, form, pinyin, variant)`` tuples.

    :param rows: CSV rows in file order.
    :param overrides: Mapping from form to a manual pinyin value.
    :raises ValueError: A row does not align.
    :returns: One tuple per form.
    """
    expanded: list[tuple[int, str, str, str, str]] = []
    for index, row in enumerate(rows, start=1):
        english = row.get("english", "")
        forms = split_forms(row.get("mandarin", ""), index)
        pinyins = align_pinyin(forms, row.get("transliteration", ""), index)
        for position, (form, pinyin) in enumerate(zip(forms, pinyins)):
            variant = "primary" if position == 0 else "alternate"
            expanded.append(
                (index, english, form, overrides.get(form, pinyin), variant)
            )
    return expanded


def load_word_table(
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> list[WordForm]:
    """Load the Swadesh word table from the reviewed CSV.

    :param csv_path: Reviewed word CSV with columns
        ``english,mandarin,transliteration``.
    :param swadesh_path: File with one English label per line.
    :param overrides_path: CSV file with manual pinyin overrides.
    :raises ValueError: The CSV row count differs from the label count, or a
        row does not align.
    :returns: One :class:`WordForm` per expanded form.
    """
    rows = _read_csv_rows(csv_path)
    labels = read_swadesh_labels(swadesh_path)
    if len(rows) != len(labels):
        raise ValueError(
            f"{csv_path}: {len(rows)} rows but {len(labels)} Swadesh labels"
        )
    converter = _traditional_converter()
    table: list[WordForm] = []
    for index, english, form, pinyin, variant in _row_forms(
        rows, read_pinyin_overrides(overrides_path)
    ):
        table.append(
            WordForm(
                swadesh_index=index,
                english=english,
                mandarin=form,
                pinyin=pinyin,
                traditional=converter.convert(form),
                variant=variant,
                zipf=zipf_frequency(form, "zh"),
            )
        )
    return table


def _check_size(table: list[WordForm]) -> list[str]:
    failures: list[str] = []
    indices = sorted({form.swadesh_index for form in table})
    if len(indices) != SWADESH_ROWS:
        failures.append(f"row count: expected {SWADESH_ROWS}, found {len(indices)}")
    if not MIN_FORMS <= len(table) <= MAX_FORMS:
        failures.append(
            f"form count: expected {MIN_FORMS} to {MAX_FORMS}, found {len(table)}"
        )
    return failures


def _check_indices(table: list[WordForm]) -> list[str]:
    indices = {form.swadesh_index for form in table}
    expected = set(range(1, SWADESH_ROWS + 1))
    missing = sorted(expected - indices)
    extra = sorted(indices - expected)
    if not missing and not extra:
        return []
    return [f"swadesh index: missing {missing}, extra {extra}"]


def _check_unique(table: list[WordForm]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for form in table:
        if form.mandarin in seen:
            duplicates.add(form.mandarin)
        seen.add(form.mandarin)
    if not duplicates:
        return []
    return [f"duplicate form: {sorted(duplicates)}"]


def _check_labels(table: list[WordForm], swadesh_path: Path) -> list[str]:
    labels = read_swadesh_labels(swadesh_path)
    failures: list[str] = []
    if len(labels) != SWADESH_ROWS:
        failures.append(f"swadesh labels: expected {SWADESH_ROWS}, found {len(labels)}")
    for form in table:
        if not 1 <= form.swadesh_index <= len(labels):
            continue
        label = labels[form.swadesh_index - 1]
        if form.english != label:
            failures.append(
                f"label {form.swadesh_index}: expected {label!r}, "
                f"found {form.english!r}"
            )
    return failures


def _check_pinyin(
    table: list[WordForm], csv_path: Path, overrides_path: Path
) -> list[str]:
    try:
        rows = _read_csv_rows(csv_path)
        expected = {form: pinyin for _, _, form, pinyin, _ in _row_forms(rows, {})}
    except (OSError, ValueError) as error:
        return [f"pinyin source: {error}"]
    overrides = read_pinyin_overrides(overrides_path)
    failures: list[str] = []
    for form in table:
        source = expected.get(form.mandarin)
        if source is None:
            failures.append(f"pinyin {form.mandarin}: form is not in {csv_path}")
            continue
        if _normalize(form.pinyin) == source:
            continue
        if overrides.get(form.mandarin) == _normalize(form.pinyin):
            continue
        failures.append(
            f"pinyin {form.mandarin}: expected {source!r}, found {form.pinyin!r}"
        )
    return failures


def _check_review_file(review_path: Path) -> list[str]:
    if not review_path.exists():
        return [f"review file missing: {review_path}"]
    open_rows = [
        row
        for row in _read_csv_rows(review_path)
        if row.get("status", "").lower() == OPEN_STATUS
    ]
    if not open_rows:
        return []
    indices = [row.get("index", "") for row in open_rows]
    return [f"review file has {len(open_rows)} open row(s): {indices}"]


def check_word_table(
    table: list[WordForm],
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
) -> list[str]:
    """Check the word table against the reviewed files.

    :param table: Word table from :func:`load_word_table`.
    :param csv_path: Reviewed word CSV, the pinyin source.
    :param swadesh_path: File with one English label per line.
    :param overrides_path: CSV file with manual pinyin overrides.
    :param review_path: Review file that must exist and have no open row.
    :returns: Failure strings. An empty list means every check passes.
    """
    failures = _check_size(table)
    failures.extend(_check_indices(table))
    failures.extend(_check_unique(table))
    failures.extend(_check_labels(table, swadesh_path))
    failures.extend(_check_pinyin(table, csv_path, overrides_path))
    failures.extend(_check_review_file(review_path))
    return failures


def word_table_report(table: list[WordForm]) -> dict[str, Any]:
    """Summarize the word table.

    :param table: Word table from :func:`load_word_table`.
    :returns: Counts of words and forms, the low Zipf forms, and the forms
        with no candidate sentence. ``uncovered`` stays empty until S2.
    """
    forms = [form.mandarin for form in table]
    low = sorted(
        (
            (form.mandarin, round(form.zipf, 2))
            for form in table
            if form.zipf < MIN_ZIPF
        ),
        key=lambda item: item[1],
    )
    return {
        "words": len({form.swadesh_index for form in table}),
        "forms": len(table),
        "unique_forms": len(set(forms)),
        "primary_forms": sum(1 for form in table if form.variant == "primary"),
        "alternate_forms": sum(1 for form in table if form.variant == "alternate"),
        "core_forms": sum(1 for form in table if form.tier == "core"),
        "zipf_below_3": len(low),
        "low_zipf_forms": low,
        "uncovered": [],
    }


def print_word_table_report(report: dict[str, Any]) -> None:
    """Print a word table report.

    :param report: Mapping from :func:`word_table_report`.
    """
    print(
        f"words={report['words']} forms={report['forms']} "
        f"unique={report['unique_forms']} "
        f"primary={report['primary_forms']} "
        f"alternate={report['alternate_forms']}"
    )
    print(
        f"zipf<{MIN_ZIPF}: {report['zipf_below_3']} "
        f"uncovered={len(report['uncovered'])}"
    )
    low_forms = report.get("low_zipf_forms", [])
    if low_forms:
        print(
            "low zipf forms: "
            + ", ".join(f"{form} {score}" for form, score in low_forms)
        )


@dataclass(frozen=True)
class Candidate:
    """One candidate sentence for the Mandarin sentence pack.

    :param target: Assigned Swadesh form, the lowest covered form by index.
    :param sentence: Simplified Chinese sentence.
    :param text_original: Sentence text before the OpenCC conversion.
    :param pinyin: Tone-marked pinyin, empty until the card is selected.
    :param english: Paired English translation.
    :param tatoeba_id: Tatoeba sentence id.
    :param tatoeba_author: Tatoeba author name, empty when unknown.
    :param english_id: Tatoeba English sentence id.
    :param audio_id: Native Tatoeba audio id, 0 when absent.
    :param audio_license: License of the native recording, empty when absent.
    :param audio_source: ``native`` with a recording, else ``tts``.
    :param unknown_words: Token count outside the known set.
    :param han_len: Han character count.
    :param zipf_mean: Mean Zipf value of the sentence tokens.
    :param rank_key: Serialized ranking key.
    :param tier: ``core`` or ``scaffold``.
    :param audio_missing: True when the sentence has no native recording.
    :param covered: Forms that appear exactly one time as a token.
    """

    target: str
    sentence: str
    text_original: str
    pinyin: str
    english: str
    tatoeba_id: int
    tatoeba_author: str
    english_id: int
    audio_id: int
    audio_license: str
    audio_source: str
    unknown_words: int
    han_len: int
    zipf_mean: float
    rank_key: str
    tier: str
    audio_missing: bool
    covered: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _SentenceFacts:
    """Measured properties of one accepted Tatoeba sentence."""

    sentence: str
    text_original: str
    tatoeba_id: int
    tatoeba_author: str
    english: str
    english_id: int
    audio_id: int
    audio_license: str
    unknown_words: int
    han_len: int
    zipf_mean: float
    hits: tuple[str, ...]


def _simplified_converter() -> opencc.OpenCC:
    global _t2s_converter
    if _t2s_converter is None:
        _t2s_converter = opencc.OpenCC("t2s")
    return _t2s_converter


def convert_to_simplified(text: str) -> str:
    """Convert a sentence to Simplified Chinese with OpenCC.

    :param text: Sentence text, simplified or traditional.
    :returns: Simplified sentence text.
    """
    return _simplified_converter().convert(text)


def load_user_dictionary(word_table: list[WordForm]) -> None:
    """Add the word-table forms to the jieba user dictionary.

    :param word_table: Word table from :func:`load_word_table`.
    """
    for form in word_table:
        jieba.add_word(form.mandarin)


def build_known_set(word_table: list[WordForm]) -> set[str]:
    """Build the known word set from the word table and ``wordfreq``.

    :param word_table: Word table from :func:`load_word_table`.
    :returns: Word forms plus the top 1000 ``wordfreq`` zh words.
    """
    known = {form.mandarin for form in word_table}
    known.update(top_n_list("zh", KNOWN_WORDS_TOP_N))
    return known


def _tokens(sentence: str) -> list[tuple[str, str]]:
    return [(pair.word, pair.flag) for pair in posseg.cut(sentence)]


def han_tokens(sentence: str) -> list[str]:
    """Tokenize a sentence and keep Han-only tokens.

    Punctuation tokens stay out of the word counts.

    :param sentence: Simplified Chinese sentence.
    :returns: One token per word, in sentence order.
    """
    return [
        word
        for word, _ in _tokens(sentence)
        if word and all(is_chinese_char(char) for char in word)
    ]


def count_form_tokens(sentence: str, form: str) -> int:
    """Count the tokens of a sentence that equal one form.

    :param sentence: Simplified Chinese sentence.
    :param form: Word-table form.
    :returns: Token count of the form.
    """
    return sum(1 for word in han_tokens(sentence) if word == form)


def count_form_occurrences(sentence: str, form: str) -> int:
    """Count the exact form occurrences in the Han character stream.

    The count does not use the token list, because jieba merges some forms into
    a longer word. For example, 水 stays inside the token 喝水. Section 14 of
    the spec names this exact substring check as the mitigation.

    :param sentence: Simplified Chinese sentence.
    :param form: Word-table form.
    :returns: Non-overlapping occurrence count of the form.
    """
    return "".join(get_chinese_chars(sentence)).count(form)


def sentence_hits(sentence: str, forms: Iterable[str]) -> dict[str, int]:
    """Count the word-table forms that the sentence contains.

    :param sentence: Simplified Chinese sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to its occurrence count, for forms that occur.
    """
    chars = "".join(get_chinese_chars(sentence))
    present = set(chars)
    hits: dict[str, int] = {}
    for form in set(forms):
        if form and form[0] in present:
            count = chars.count(form)
            if count:
                hits[form] = count
    return hits


def has_predicate(sentence: str) -> bool:
    """Check filter F9: jieba tags at least one token as ``v*`` or ``a*``.

    :param sentence: Simplified Chinese sentence.
    :returns: True when the sentence has a predicate tag.
    """
    return any(tag.startswith(PREDICATE_TAG_PREFIXES) for _, tag in _tokens(sentence))


def has_orphan_conjunction(sentence: str) -> bool:
    """Check filter F10: a sentence-initial conjunction with no comma.

    :param sentence: Simplified Chinese sentence.
    :returns: True when the leading conjunction has no main clause.
    """
    if any(mark in sentence for mark in "，,"):
        return False
    return sentence.startswith(ORPHAN_CONJUNCTIONS)


def _count_marks(sentence: str, marks: str) -> int:
    return sum(1 for char in sentence if char in marks)


def check_sentence_shape(
    sentence: str, min_han: int = 4, max_han: int = 10
) -> str | None:
    """Apply filters F2 to F8 to one sentence.

    :param sentence: Simplified Chinese sentence.
    :param min_han: Minimum Han character count.
    :param max_han: Maximum Han character count.
    :returns: The first failed filter key, or None when all filters pass.
    """
    han_len = len(get_chinese_chars(sentence))
    if not min_han <= han_len <= max_han:
        return "F2"
    if re.search(r"[A-Za-z0-9０-９]", sentence):
        return "F3"
    if is_repeated_zh_sentence(sentence):
        return "F4"
    if not sentence or sentence[-1] not in FINAL_MARKS:
        return "F5"
    if _count_marks(sentence, FINAL_MARKS) != 1:
        return "F6"
    if any(char in QUOTE_MARKS for char in sentence):
        return "F7"
    if _count_marks(sentence, INTERNAL_MARKS) > 1:
        return "F8"
    return None


def read_english_links(links_path: Path) -> dict[int, list[int]]:
    """Read the Mandarin to English translation links.

    :param links_path: TSV file with columns ``cmn_id,eng_id`` and no header.
    :returns: Mapping from Mandarin sentence id to its English sentence ids.
    """
    links: dict[int, list[int]] = defaultdict(list)
    for parts in _iter_tsv(links_path):
        if len(parts) < 2:
            continue
        links[_to_int(parts[0])].append(_to_int(parts[1]))
    return dict(links)


def read_cmn_sentences(sentences_path: Path) -> dict[int, str]:
    """Read the Mandarin sentence texts.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :returns: Mapping from sentence id to the raw sentence text.
    """
    sentences: dict[int, str] = {}
    for parts in _iter_tsv(sentences_path):
        if len(parts) < 3:
            continue
        sentences[_to_int(parts[0])] = parts[2]
    return sentences


def read_cmn_authors(detailed_path: Path) -> dict[int, str]:
    """Read the author of each Mandarin sentence.

    :param detailed_path: TSV file with columns ``id,lang,text,author,...``.
    :returns: Mapping from sentence id to the author name.
    """
    authors: dict[int, str] = {}
    for parts in _iter_tsv(detailed_path):
        if len(parts) < 4:
            continue
        author = "" if parts[3] == "\\N" else parts[3]
        authors[_to_int(parts[0])] = author
    return authors


def read_cmn_audio(audio_path: Path) -> dict[int, tuple[int, str]]:
    """Read the native audio id and license of each Mandarin sentence.

    :param audio_path: TSV file with columns
        ``sentence_id,audio_id,author,license,attribution``.
    :returns: Mapping from sentence id to ``(audio_id, license)``.
    """
    audio: dict[int, tuple[int, str]] = {}
    for parts in _iter_tsv(audio_path):
        if len(parts) < 2:
            continue
        license_name = parts[3] if len(parts) > 3 else ""
        audio[_to_int(parts[0])] = (_to_int(parts[1]), license_name)
    return audio


def read_english_sentences(
    sentences_path: Path, wanted_ids: set[int]
) -> dict[int, str]:
    """Stream the English dump and keep the wanted rows.

    The file has 2 million rows, so this reads it one line at a time.

    :param sentences_path: TSV file with columns ``id,lang,text`` and no header.
    :param wanted_ids: English sentence ids to keep.
    :returns: Mapping from English sentence id to its text.
    """
    if not wanted_ids:
        return {}
    found: dict[int, str] = {}
    with sentences_path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            sentence_id = _to_int(parts[0])
            if sentence_id not in wanted_ids:
                continue
            found[sentence_id] = parts[2]
            if len(found) == len(wanted_ids):
                break
    return found


def _iter_tsv(path: Path) -> Iterable[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            yield line.rstrip("\n").split("\t")


def _to_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


@lru_cache(maxsize=100_000)
def _zipf(word: str) -> float:
    return zipf_frequency(word, "zh")


def _mean_zipf(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(_zipf(word) for word in words) / len(words)


def _pair_english(
    english_ids: list[int], english: dict[int, str]
) -> tuple[int, str] | None:
    """Pick the English translation with the fewest words, then the lowest id.

    :param english_ids: Linked English sentence ids.
    :param english: Mapping from English sentence id to its text.
    :returns: ``(english_id, text)`` or None when no link has a text.
    """
    best: tuple[int, int, str] | None = None
    for english_id in english_ids:
        text = english.get(english_id)
        if not text:
            continue
        words = len(text.split())
        if best is None or (words, english_id) < (best[0], best[1]):
            best = (words, english_id, text)
    if best is None:
        return None
    return best[1], best[2]


def rank_key_parts(candidate: Candidate) -> tuple[int, int, float, int, int]:
    """Build the sort key of section 8.3, first item first.

    :param candidate: Candidate sentence.
    :returns: Unknown count, audio penalty, negated Zipf, length penalty, id.
    """
    audio_penalty = 0 if candidate.audio_id else 1
    length_penalty = (
        0 if PREFERRED_MIN_HAN <= candidate.han_len <= PREFERRED_MAX_HAN else 1
    )
    return (
        candidate.unknown_words,
        audio_penalty,
        -candidate.zipf_mean,
        length_penalty,
        candidate.tatoeba_id,
    )


def serialize_rank_key(parts: tuple[int, int, float, int, int]) -> str:
    """Serialize a ranking key tuple into the CSV string.

    :param parts: Tuple from :func:`rank_key_parts`.
    :returns: Pipe-separated key, for example ``0|1|5.120|0|123``.
    """
    unknown_words, audio_penalty, negated_zipf, length_penalty, tatoeba_id = parts
    return (
        f"{unknown_words}|{audio_penalty}|{negated_zipf:.3f}|"
        f"{length_penalty}|{tatoeba_id}"
    )


def rank_order(candidate: Candidate) -> tuple[int, int, float, int, int]:
    """Sort candidates with the section 8.3 ranking key.

    :param candidate: Candidate sentence.
    :returns: Sort key, lowest value first.
    """
    return rank_key_parts(candidate)


def _make_candidate(facts: _SentenceFacts, target: str, tier: str) -> Candidate:
    audio_source = "native" if facts.audio_id else "tts"
    candidate = Candidate(
        target=target,
        sentence=facts.sentence,
        text_original=facts.text_original,
        pinyin="",
        english=facts.english,
        tatoeba_id=facts.tatoeba_id,
        tatoeba_author=facts.tatoeba_author,
        english_id=facts.english_id,
        audio_id=facts.audio_id,
        audio_license=facts.audio_license,
        audio_source=audio_source,
        unknown_words=facts.unknown_words,
        han_len=facts.han_len,
        zipf_mean=facts.zipf_mean,
        rank_key="",
        tier=tier,
        audio_missing=not facts.audio_id,
        covered=frozenset(facts.hits),
    )
    return replace(candidate, rank_key=serialize_rank_key(rank_key_parts(candidate)))


def _form_positions(word_table: list[WordForm]) -> dict[str, tuple[int, int]]:
    """Give each form its Swadesh index and its primary-form rank.

    :param word_table: Word table from :func:`load_word_table`.
    :returns: Mapping from form to ``(swadesh_index, variant_rank)``.
    """
    return {
        form.mandarin: (
            form.swadesh_index,
            0 if form.variant == "primary" else 1,
        )
        for form in word_table
    }


def _stage_number(key: str) -> int:
    if key == "unknown":
        return len(FILTER_KEYS)
    return int(key[1:])


def _count_hits(chars: str, forms: set[str]) -> tuple[str, ...]:
    present = set(chars)
    return tuple(
        sorted(form for form in forms if form[0] in present and chars.count(form) == 1)
    )


def build_candidate_pool(
    word_table: list[WordForm],
    data_dir: Path = DEFAULT_DATA_DIR,
    min_han: int = 4,
    max_han: int = 10,
    max_unknown: int = 2,
    fallback: bool = False,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Build the S2 candidate pool from the Tatoeba exports.

    Filters run in the section 7 order. With ``fallback``, a sentence that
    passes the 3 to 10 window and drops F6 and F8 becomes a scaffold
    candidate for a form that has no core candidate. The function also sets
    ``candidate_count`` and ``tier`` on the word table.

    :param word_table: Word table from :func:`load_word_table`.
    :param data_dir: Directory with the Tatoeba TSV files.
    :param min_han: Minimum Han character count for a core candidate.
    :param max_han: Maximum Han character count.
    :param max_unknown: Unknown token budget per sentence.
    :param fallback: Add scaffold candidates for forms with no core candidate.
    :returns: The candidate list and the pool report.
    """
    forms = [form.mandarin for form in word_table]
    form_set = set(forms)
    positions = _form_positions(word_table)
    load_user_dictionary(word_table)
    known = build_known_set(word_table)
    links = read_english_links(data_dir / CMN_LINKS_FILE)
    sentences = read_cmn_sentences(data_dir / CMN_SENTENCES_FILE)
    wanted_english = {
        english_id for english_ids in links.values() for english_id in english_ids
    }
    english = read_english_sentences(data_dir / ENG_SENTENCES_FILE, wanted_english)
    authors = read_cmn_authors(data_dir / CMN_DETAILED_FILE)
    audio = read_cmn_audio(data_dir / CMN_AUDIO_FILE)

    counts: dict[str, int] = dict.fromkeys(FILTER_KEYS, 0)
    core_facts: list[_SentenceFacts] = []
    relax_facts: list[_SentenceFacts] = []
    seen: set[str] = set()
    relax_seen: set[str] = set()
    total = 0
    for cmn_id in sorted(links):
        raw = sentences.get(cmn_id)
        if raw is None:
            continue
        total += 1
        pair = _pair_english(links[cmn_id], english)
        if pair is None:
            continue
        english_id, english_text = pair
        sentence = convert_to_simplified(raw).strip()
        han_chars = "".join(get_chinese_chars(sentence))
        han_len = len(han_chars)
        shape = check_sentence_shape(sentence, min_han, max_han)
        relax_shape = (
            check_sentence_shape(sentence, FALLBACK_MIN_HAN, max_han)
            if fallback
            else "F2"
        )
        relax_ok = relax_shape is None or relax_shape in ("F6", "F8")
        if shape is not None and not (fallback and relax_ok):
            for key in FILTER_KEYS[: _stage_number(shape) - 1]:
                counts[key] += 1
            continue

        tokens = _tokens(sentence)
        words = [
            word
            for word, _ in tokens
            if word and all(is_chinese_char(char) for char in word)
        ]
        predicate = any(tag.startswith(PREDICATE_TAG_PREFIXES) for _, tag in tokens)
        orphan = has_orphan_conjunction(sentence)
        unknown_words = sum(1 for word in words if word not in known)
        english_words = len(english_text.split())
        english_ok = MIN_ENGLISH_WORDS <= english_words <= MAX_ENGLISH_WORDS

        fail: int | None = _stage_number(shape) if shape is not None else None
        if fail is None:
            if not predicate:
                fail = 9
            elif orphan:
                fail = 10
            elif not english_ok:
                fail = 11
            elif sentence in seen:
                fail = 12
            elif unknown_words > max_unknown:
                fail = 13
        passed = (fail - 1) if fail is not None else len(FILTER_KEYS)
        for key in FILTER_KEYS[:passed]:
            counts[key] += 1

        hits = _count_hits(han_chars, form_set)
        if fail is None and hits:
            seen.add(sentence)
            audio_id, audio_license = audio.get(cmn_id, (0, ""))
            core_facts.append(
                _SentenceFacts(
                    sentence=sentence,
                    text_original=raw,
                    tatoeba_id=cmn_id,
                    tatoeba_author=authors.get(cmn_id, ""),
                    english=english_text,
                    english_id=english_id,
                    audio_id=audio_id,
                    audio_license=audio_license,
                    unknown_words=unknown_words,
                    han_len=han_len,
                    zipf_mean=_mean_zipf(words),
                    hits=hits,
                )
            )
        elif (
            fallback
            and relax_ok
            and predicate
            and not orphan
            and english_ok
            and unknown_words <= max_unknown
            and hits
            and sentence not in seen
            and sentence not in relax_seen
        ):
            relax_seen.add(sentence)
            audio_id, audio_license = audio.get(cmn_id, (0, ""))
            relax_facts.append(
                _SentenceFacts(
                    sentence=sentence,
                    text_original=raw,
                    tatoeba_id=cmn_id,
                    tatoeba_author=authors.get(cmn_id, ""),
                    english=english_text,
                    english_id=english_id,
                    audio_id=audio_id,
                    audio_license=audio_license,
                    unknown_words=unknown_words,
                    han_len=han_len,
                    zipf_mean=_mean_zipf(words),
                    hits=hits,
                )
            )

    core_covered: Counter[str] = Counter()
    pool: list[Candidate] = []
    for facts in core_facts:
        target = min(facts.hits, key=lambda form: positions[form])
        pool.append(_make_candidate(facts, target, "core"))
        core_covered.update(facts.hits)

    uncovered_forms = [form for form in forms if form not in core_covered]
    scaffold_texts: set[str] = set()
    scaffold_count = 0
    if fallback and uncovered_forms:
        uncovered_set = set(uncovered_forms)
        core_texts = {facts.sentence for facts in core_facts}
        for facts in relax_facts:
            if facts.sentence in core_texts:
                continue
            targets = [form for form in facts.hits if form in uncovered_set]
            if not targets:
                continue
            scaffold_texts.add(facts.sentence)
            for form in targets:
                scaffold_count += 1
                pool.append(
                    replace(
                        _make_candidate(facts, form, "scaffold"),
                        covered=frozenset((form,)),
                    )
                )

    candidates_per_form: Counter[str] = Counter()
    for candidate in pool:
        candidates_per_form.update(candidate.covered)
    for form in word_table:
        form.candidate_count = candidates_per_form[form.mandarin]
        if form.mandarin not in core_covered and form.candidate_count:
            form.tier = "scaffold"

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "min_han": min_han,
        "max_han": max_han,
        "max_unknown": max_unknown,
        "fallback": fallback,
        "english_ids": len(wanted_english),
        "total_linked": total,
        "counts": counts,
        "filter_names": dict(FILTER_NAMES),
        "pool_candidates": len(pool),
        "core_sentences": len(core_facts),
        "scaffold_sentences": len(scaffold_texts),
        "scaffold_candidates": scaffold_count,
        "forms_with_core_candidate": len(core_covered),
        "forms_without_core_candidate": len(uncovered_forms),
    }
    return pool, report


class SimilarityIndex:
    """Reject a sentence that is too close to a kept card.

    The guard uses the character-set Jaccard value of section 8.2. Only kept
    cards that share one character with the sentence are compared.
    """

    def __init__(self, limit: float = SIMILARITY_LIMIT) -> None:
        """Start an empty index.

        :param limit: Jaccard value at which a sentence counts as a duplicate.
        """
        self.limit = limit
        self._texts: list[str] = []
        self._by_char: dict[str, list[int]] = defaultdict(list)

    def too_similar(self, sentence: str) -> bool:
        """Check a sentence against the kept cards.

        :param sentence: Simplified Chinese sentence.
        :returns: True when a kept card reaches the Jaccard limit.
        """
        shared: set[int] = set()
        for char in set(get_chinese_chars(sentence)):
            shared.update(self._by_char.get(char, ()))
        return any(
            char_similarity_zh(sentence, self._texts[index]) >= self.limit
            for index in shared
        )

    def add(self, sentence: str) -> None:
        """Add a kept card sentence to the index.

        :param sentence: Simplified Chinese sentence.
        """
        position = len(self._texts)
        self._texts.append(sentence)
        for char in set(get_chinese_chars(sentence)):
            self._by_char[char].append(position)


def _assign_target(
    covered: frozenset[str],
    positions: dict[str, tuple[int, int]],
    counts: dict[str, int],
    per_word: int,
) -> str | None:
    """Pick the lowest covered form that still needs a card.

    :param covered: Forms that the sentence contains one time.
    :param positions: Mapping from form to ``(swadesh_index, variant_rank)``.
    :param counts: Current card count per form.
    :param per_word: Card limit per form.
    :returns: The target form, or None when every covered form is full.
    """
    for form in sorted(covered, key=lambda name: positions[name]):
        if counts.get(form, per_word) < per_word:
            return form
    return None


def select_sentences(
    pool: list[Candidate],
    word_table: list[WordForm],
    per_word: int = 2,
    max_cards: int = 450,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select cards with the greedy set cover of section 8.4.

    Forms are processed in Swadesh index order, primary form first. A sentence
    goes to the lowest covered form that still needs a card, so a sentence
    never becomes two cards. Only the assigned form gains a card.

    :param pool: Candidate list from :func:`build_candidate_pool`.
    :param word_table: Word table from :func:`load_word_table`.
    :param per_word: Card limit per form.
    :param max_cards: Hard card cap.
    :returns: The cards in Swadesh order and the selection report.
    """
    positions = _form_positions(word_table)
    ordered_forms = sorted(positions, key=lambda form: positions[form])
    counts: dict[str, int] = dict.fromkeys(ordered_forms, 0)
    by_form: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in pool:
        for form in candidate.covered:
            if form in counts:
                by_form[form].append(candidate)
    for candidates in by_form.values():
        candidates.sort(key=rank_order)

    used: set[int] = set()
    kept: list[Candidate] = []
    index = SimilarityIndex()
    for form in ordered_forms:
        for candidate in by_form.get(form, ()):
            if len(kept) >= max_cards or counts[form] >= per_word:
                break
            if candidate.tatoeba_id in used:
                continue
            target = _assign_target(candidate.covered, positions, counts, per_word)
            if target is None or index.too_similar(candidate.sentence):
                continue
            counts[target] += 1
            used.add(candidate.tatoeba_id)
            index.add(candidate.sentence)
            kept.append(
                replace(
                    candidate,
                    target=target,
                    pinyin=sentence_pinyin(candidate.sentence),
                )
            )

    kept.sort(key=lambda card: (positions[card.target], rank_order(card)))
    concepts = {form.swadesh_index: form.english for form in word_table}
    covered_concepts = {
        positions[form][0] for form in ordered_forms if counts[form] > 0
    }
    uncovered = [
        {
            "target": form,
            "swadesh_index": positions[form][0],
            "english": concepts.get(positions[form][0], ""),
            "pool_candidates": len(by_form.get(form, ())),
        }
        for form in ordered_forms
        if counts[form] == 0
    ]
    forms_with_card = sum(1 for form in ordered_forms if counts[form] > 0)
    report: dict[str, Any] = {
        "per_word": per_word,
        "max_cards": max_cards,
        "cards": len(kept),
        "tier_counts": dict(Counter(card.tier for card in kept)),
        "forms": {
            "total": len(ordered_forms),
            "with_card": forms_with_card,
            "ratio": round(forms_with_card / len(ordered_forms), 4)
            if ordered_forms
            else 0.0,
        },
        "core_coverage": {
            "concepts": len(concepts),
            "with_card": len(covered_concepts),
            "ratio": round(len(covered_concepts) / len(concepts), 4)
            if concepts
            else 0.0,
        },
        "uncovered": uncovered,
        "zero_candidates": [
            item["target"] for item in uncovered if not item["pool_candidates"]
        ],
        "cut_by_cap": [item["target"] for item in uncovered if item["pool_candidates"]],
    }
    return kept, report


def _tone_parts(syllable: str) -> tuple[str, int]:
    tone3 = tone_to_tone3(syllable)
    if tone3 and tone3[-1].isdigit():
        return tone3[:-1], int(tone3[-1])
    return tone3, 5


def apply_tone_sandhi(chars: list[str], syllables: list[str]) -> list[str]:
    """Apply the 不 and 一 tone sandhi rules to one sentence.

    不 becomes second tone before a fourth tone. 一 becomes second tone before
    a fourth tone and fourth tone before a first, second, or third tone. The
    function changes nothing when the tones do not match those patterns.

    :param chars: Han characters of the sentence, in order.
    :param syllables: Tone-marked pinyin, one entry per character.
    :returns: The adjusted pinyin list.
    """
    readings = [_tone_parts(syllable) for syllable in syllables]
    tuned = list(syllables)
    for position, char in enumerate(chars):
        base, tone = readings[position]
        next_tone = readings[position + 1][1] if position + 1 < len(chars) else 5
        if char == "不" and tone == 4 and next_tone == 4:
            tuned[position] = to_tone(f"{base}2")
        elif char == "一" and tone == 1 and position + 1 < len(chars):
            if next_tone == 4:
                tuned[position] = to_tone(f"{base}2")
            elif next_tone in (1, 2, 3):
                tuned[position] = to_tone(f"{base}4")
    return tuned


def sentence_pinyin(sentence: str) -> str:
    """Read a sentence with pypinyin and the 不 and 一 sandhi rules.

    :param sentence: Simplified Chinese sentence.
    :raises ValueError: The syllable count differs from the Han count.
    :returns: Space-separated tone-marked pinyin.
    """
    chars = [char for char in sentence if is_chinese_char(char)]
    if not chars:
        return ""
    groups = pypinyin.pinyin(sentence, style=pypinyin.Style.TONE, heteronym=False)
    if len(groups) == len(sentence):
        syllables = [
            group[0] for char, group in zip(sentence, groups) if is_chinese_char(char)
        ]
    else:
        syllables = [
            group[0]
            for group in pypinyin.pinyin(
                chars, style=pypinyin.Style.TONE, heteronym=False
            )
        ]
    if len(syllables) != len(chars):
        raise ValueError(
            f"{sentence!r}: {len(syllables)} syllables for {len(chars)} Han characters"
        )
    return " ".join(apply_tone_sandhi(chars, syllables))


def _sentence_row(card: Candidate) -> dict[str, Any]:
    return {
        "target": card.target,
        "sentence": card.sentence,
        "text_original": card.text_original,
        "pinyin": card.pinyin,
        "english": card.english,
        "tatoeba_id": card.tatoeba_id,
        "tatoeba_author": card.tatoeba_author,
        "english_id": card.english_id,
        "audio_id": card.audio_id,
        "audio_source": card.audio_source,
        "audio_license": card.audio_license,
        "unknown_words": card.unknown_words,
        "han_len": card.han_len,
        "zipf_mean": f"{card.zipf_mean:.2f}",
        "rank_key": card.rank_key,
        "tier": card.tier,
        "audio_missing": "true" if card.audio_missing else "false",
    }


def write_sentence_csv(cards: list[Candidate], path: Path = SENTENCES_PATH) -> Path:
    """Write the card rows of section 8.5.

    :param cards: Cards from :func:`select_sentences`, in output order.
    :param path: Output CSV file.
    :returns: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_FIELDS)
        writer.writeheader()
        for card in cards:
            writer.writerow(_sentence_row(card))
    return path


def _review_status(
    form: WordForm, pool_count: int, core_count: int, per_word: int
) -> str | None:
    if not pool_count:
        return "uncovered"
    if not core_count:
        return "scaffold_only"
    if pool_count < per_word:
        return "thin"
    if form.zipf < MIN_ZIPF:
        return "low_zipf"
    return None


def write_sentence_review_csv(
    pool: list[Candidate],
    word_table: list[WordForm],
    path: Path = SENTENCE_REVIEW_PATH,
    per_word: int = 2,
    top_n: int = 3,
) -> Path:
    """Write the review rows for the weak words.

    One row per candidate for the top ``top_n`` candidates of each word that
    needs review. The ``status`` column is ``uncovered`` (no candidate),
    ``scaffold_only`` (fallback window only), ``thin`` (fewer than
    ``per_word`` candidates), or ``low_zipf`` (Zipf value below 3.0). A word
    with no candidate gets one row with rank 0 and empty candidate columns.

    :param pool: Candidate list from :func:`build_candidate_pool`.
    :param word_table: Word table from :func:`load_word_table`.
    :param path: Output CSV file.
    :param per_word: Card limit per form, used for the thin test.
    :param top_n: Candidate rows per word.
    :returns: The written path.
    """
    pool_counts: Counter[str] = Counter()
    core_counts: Counter[str] = Counter()
    by_form: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in pool:
        for form in candidate.covered:
            pool_counts[form] += 1
            by_form[form].append(candidate)
            if candidate.tier == "core":
                core_counts[form] += 1
    for candidates in by_form.values():
        candidates.sort(key=rank_order)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_REVIEW_FIELDS)
        writer.writeheader()
        for form in word_table:
            status = _review_status(
                form, pool_counts[form.mandarin], core_counts[form.mandarin], per_word
            )
            if status is None:
                continue
            base = {
                "target": form.mandarin,
                "swadesh_index": form.swadesh_index,
                "english": form.english,
                "variant": form.variant,
                "status": status,
                "zipf": f"{form.zipf:.2f}",
                "candidate_count": pool_counts[form.mandarin],
                "scaffold_candidates": pool_counts[form.mandarin]
                - core_counts[form.mandarin],
            }
            candidates = by_form.get(form.mandarin, [])[:top_n]
            if not candidates:
                writer.writerow({**base, "rank": 0})
                continue
            for rank, candidate in enumerate(candidates, start=1):
                writer.writerow(
                    {
                        **base,
                        "rank": rank,
                        "sentence": candidate.sentence,
                        "text_original": candidate.text_original,
                        "pinyin": candidate.pinyin
                        or sentence_pinyin(candidate.sentence),
                        "english_text": candidate.english,
                        "tatoeba_id": candidate.tatoeba_id,
                        "unknown_words": candidate.unknown_words,
                        "han_len": candidate.han_len,
                        "rank_key": candidate.rank_key,
                        "tier": candidate.tier,
                    }
                )
    return path


def build_pack_report(
    pool_report: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    """Merge the pool report and the selection report.

    :param pool_report: Report from :func:`build_candidate_pool`.
    :param selection: Report from :func:`select_sentences`.
    :returns: One mapping with the counts, the cards, and the coverage.
    """
    report = dict(pool_report)
    report["cards"] = selection["cards"]
    report["per_word"] = selection["per_word"]
    report["max_cards"] = selection["max_cards"]
    report["tier_counts"] = selection["tier_counts"]
    report["form_coverage"] = selection["forms"]
    report["coverage"] = selection["core_coverage"]
    report["uncovered"] = selection["uncovered"]
    report["zero_candidates"] = selection["zero_candidates"]
    report["cut_by_cap"] = selection["cut_by_cap"]
    return report


def write_pool_report(report: dict[str, Any], path: Path = POOL_REPORT_PATH) -> Path:
    """Write the pool report as JSON.

    :param report: Mapping from :func:`build_pack_report`.
    :param path: Output JSON file.
    :returns: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def print_sentence_report(report: dict[str, Any]) -> None:
    """Print the filter counts, the card count, and the coverage.

    :param report: Mapping from :func:`build_pack_report`.
    """
    print(f"linked sentences: {report['total_linked']}")
    for key in FILTER_KEYS:
        print(f"  {key} {FILTER_NAMES[key]}: {report['counts'][key]}")
    print(
        f"pool candidates: {report['pool_candidates']} "
        f"core={report['core_sentences']} "
        f"scaffold={report['scaffold_candidates']}"
    )
    print(
        f"cards: {report['cards']} "
        f"core={report['tier_counts'].get('core', 0)} "
        f"scaffold={report['tier_counts'].get('scaffold', 0)}"
    )
    forms = report["form_coverage"]
    coverage = report["coverage"]
    print(
        f"forms with card: {forms['with_card']}/{forms['total']} "
        f"({forms['ratio'] * 100:.1f}%)"
    )
    print(
        f"core coverage: {coverage['with_card']}/{coverage['concepts']} "
        f"({coverage['ratio'] * 100:.1f}%)"
    )
    uncovered = report["uncovered"]
    print(f"uncovered words: {len(uncovered)}")
    if uncovered:
        preview = ", ".join(item["target"] for item in uncovered)
        print(f"  {preview}")
