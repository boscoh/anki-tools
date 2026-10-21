"""Mandarin language pack of the shared sentence pipeline.

The module holds the Mandarin word table, the jieba and pinyin helpers, and the
Mandarin language profile. The shared stages read the profile, so no shared
module contains a test on the language name.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import jieba
import jieba.posseg as posseg
import opencc
import pypinyin
from pypinyin.contrib.tone_convert import to_tone, tone_to_tone3
from wordfreq import top_n_list, zipf_frequency

from anki_tools.languages.mandarin.characters import (
    char_similarity_zh,
    get_chinese_chars,
    is_chinese_char,
    is_repeated_zh_sentence,
)
from anki_tools.sentence_pack.guards import compound_spans, inside_spans, occurrences
from anki_tools.sentence_pack.pack import (
    ATTRIBUTION_PATH,
    audio_license_source,
    tts_engine_name,
    write_attribution_section,
)
from anki_tools.sentence_pack.profile import LanguageProfile, make_sense_filter, register
from anki_tools.sentence_pack.select import rank_order, review_status

DEFAULT_CSV_PATH = Path("vocab/english-mandarin.csv")
DEFAULT_SWADESH_PATH = Path("vocab/swadesh.txt")
DEFAULT_OVERRIDES_PATH = Path("vocab/mandarin_pinyin_overrides.csv")
DEFAULT_REVIEW_PATH = Path("vocab/swadesh_review.csv")
SENTENCES_PATH = Path("vocab/mandarin_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/mandarin_sentences_review.csv")
POOL_REPORT_PATH = Path("data/mandarin_pool_report.json")
AUDIO_DIR = Path("audio/mandarin_sentences")
AUDIO_TSV_PATH = Path("data/tatoeba/cmn_sentences_with_audio.tsv")

SWADESH_ROWS = 207
MIN_FORMS = 205
MAX_FORMS = 220
MIN_ZIPF = 3.0
OPEN_STATUS = "open"
KNOWN_WORDS_TOP_N = 1000
SIMILARITY_LIMIT = 0.67
PREFERRED_MIN_HAN = 5
PREFERRED_MAX_HAN = 7
FALLBACK_MIN_HAN = 3
MIN_ENGLISH_WORDS = 2
MAX_ENGLISH_WORDS = 12

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
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9０-９]")

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

MODEL_NAME = "Mandarin Swadesh Sentence"
DECK_NAME = "Swadesh Mandarin Sentences"
FIELDS = (
    "Audio",
    "Sentence",
    "Pinyin",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
)
SORT_FIELD_INDEX = 4
DECK_DESCRIPTION = "Word data CC BY-SA 4.0. Sentence text CC BY 2.0 FR. Audio: gTTS."
SWADESH_URL = "https://en.wiktionary.org/wiki/Module:Swadesh/data/cmn"

FRONT_TEMPLATE = (
    '<div class="front-audio">{{Audio}}</div>\n'
    '<div class="hint">Tap to show the sentence</div>'
)
BACK_TEMPLATE = (
    "{{Audio}}\n"
    "<hr id=answer>\n"
    '<div class="sentence char-links">{{Sentence}}</div>\n'
    '<div class="pinyin">{{Pinyin}}</div>\n'
    '<div class="english">{{English}}</div>\n'
    '<div class="target">{{Target}} {{Gloss}} {{WordAudio}}</div>\n'
    '<div class="source">{{Source}}</div>'
)
EXTRA_CSS = """
.front-audio {
    text-align: center;
}
.sentence {
    font-size: 2.5rem;
    text-align: center;
    margin: 0.5rem 0;
}
.pinyin {
    font-size: 1.2rem;
    text-align: center;
    color: #444;
    margin-bottom: 0.5rem;
}
.english {
    font-size: 1.1rem;
    text-align: center;
    color: #333;
    margin-bottom: 0.75rem;
}
.target {
    font-size: 1.6rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.source {
    font-size: 0.8rem;
    text-align: center;
    color: gray;
}
.hint {
    font-size: 0.9rem;
    text-align: center;
    color: gray;
}
"""

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


def _traditional_converter() -> opencc.OpenCC:
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC("s2t")
    return _converter


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


def _normalize(text: str) -> str:
    return " ".join(text.split())


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of stripped row mappings.

    :param csv_path: CSV file to read.
    :returns: One mapping per data row, with stripped values.
    """
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_swadesh_labels(swadesh_path: Path = DEFAULT_SWADESH_PATH) -> list[str]:
    """Read the Swadesh English labels in index order.

    :param swadesh_path: File with one English label per line.
    :returns: Label list, index 1 at position 0.
    """
    text = Path(swadesh_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def read_pinyin_overrides(
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> dict[str, str]:
    """Read the manual pinyin overrides.

    :param overrides_path: CSV file with columns ``mandarin,pinyin,reason``.
    :returns: Mapping from form to its reviewed pinyin. Missing file gives an
        empty mapping.
    """
    if not Path(overrides_path).exists():
        return {}
    overrides: dict[str, str] = {}
    for row in read_csv_rows(overrides_path):
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
    rows = read_csv_rows(csv_path)
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
        rows = read_csv_rows(csv_path)
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
    if not Path(review_path).exists():
        return [f"review file missing: {review_path}"]
    open_rows = [
        row
        for row in read_csv_rows(review_path)
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


FALSE_FRIENDS: dict[str, tuple[str, ...]] = {
    "切": ("一切",),
    "闻": ("新闻", "传闻"),
    "数": ("数学", "数大", "数字", "岁数", "人数"),
    "烧": ("发烧",),
    "抓": ("抓住", "抓到", "抓紧"),
    "挖": ("挖苦",),
    "挤": ("挤满",),
}


SENSE_CHECKS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "抓": (
        frozenset({"scratch", "scratches", "scratched", "scratching", "itch"}),
        frozenset({
            "grab", "grabs", "grabbed", "grabbing", "catch", "catches",
            "caught", "catching", "seize", "seizes", "seized", "arrest",
            "arrests", "arrested",
        }),
    ),
    "数": (
        frozenset({"count", "counts", "counted", "counting"}),
        frozenset({"math", "maths", "mathematics", "arithmetic", "number", "numbers"}),
    ),
    "挤": (
        frozenset({"squeeze", "squeezes", "squeezed", "squeezing"}),
        frozenset({"crowded", "crowd", "crowds", "cramped", "packed"}),
    ),
}


def false_friend_spans(sentence: str, form: str) -> list[tuple[int, int]]:
    """Give the spans where the form takes a different sense.

    :param sentence: Simplified Chinese sentence.
    :param form: Word-table form.
    :returns: ``(start, end)`` offsets in the Han character stream.
    """
    chars = "".join(get_chinese_chars(sentence))
    return compound_spans(chars, FALSE_FRIENDS.get(form, ()))


sense_filter = make_sense_filter(SENSE_CHECKS)


def form_occurrences(sentence: str, form: str) -> list[int]:
    """Give the offsets of the form occurrences that keep the Swadesh sense.

    :param sentence: Simplified Chinese sentence.
    :param form: Word-table form.
    :returns: Start offsets in the Han character stream.
    """
    if not form:
        return []
    chars = "".join(get_chinese_chars(sentence))
    blocked = false_friend_spans(sentence, form)
    return [
        position
        for position in occurrences(chars, form)
        if not inside_spans(position, blocked)
    ]


def count_form_occurrences(sentence: str, form: str) -> int:
    """Count the form occurrences that keep the Swadesh sense.

    The count does not use the token list, because jieba merges some forms into
    a longer word. For example, 水 stays inside the token 喝水. An occurrence
    inside a false-friend compound never counts, so 闻 does not hit inside 新闻.

    :param sentence: Simplified Chinese sentence.
    :param form: Word-table form.
    :returns: Occurrence count outside a different-sense compound.
    """
    return len(form_occurrences(sentence, form))


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
            count = len(form_occurrences(sentence, form))
            if count:
                hits[form] = count
    return hits


def single_hits(sentence: str, forms: Iterable[str]) -> dict[str, int]:
    """Count the forms that the sentence contains exactly one time.

    :param sentence: Simplified Chinese sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to 1, for forms with one sense-keeping hit.
    """
    chars = "".join(get_chinese_chars(sentence))
    present = set(chars)
    return {
        form: 1
        for form in set(forms)
        if form and form[0] in present and len(form_occurrences(sentence, form)) == 1
    }


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


def _form_positions(word_table: Sequence[WordForm]) -> dict[str, tuple[int, int]]:
    return {
        form.mandarin: (
            form.swadesh_index,
            0 if form.variant == "primary" else 1,
        )
        for form in word_table
    }


def make_candidate(
    facts: Any, target: str, tier: str, covered: frozenset[str]
) -> Candidate:
    """Build one Mandarin candidate card from the measured facts.

    :param facts: Sentence facts of the shared pool stage.
    :param target: Assigned word-table form.
    :param tier: ``core`` or ``scaffold``.
    :param covered: Forms that the sentence covers.
    :returns: The candidate card.
    """
    from anki_tools.sentence_pack.select import rank_key_parts, serialize_rank_key

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
        han_len=facts.unit_len,
        zipf_mean=facts.zipf_mean,
        rank_key="",
        tier=tier,
        audio_missing=not facts.audio_id,
        covered=covered,
    )
    return replace(
        candidate, rank_key=serialize_rank_key(rank_key_parts(candidate, MANDARIN))
    )


def _gloss(word: WordForm) -> str:
    return f"{word.english} · {word.pinyin}"


def _card_fields(row: dict[str, str], word: WordForm) -> dict[str, str]:
    return {"Pinyin": row.get("pinyin", "")}


def _card_order(word: WordForm, row: dict[str, str]) -> tuple:
    variant_rank = 0 if word.variant == "primary" else 1
    return (word.swadesh_index, variant_rank, int(row["tatoeba_id"]))


def _excluded_words(
    table: Sequence[WordForm], targets: Iterable[str]
) -> list[str]:
    wanted = set(targets)
    return [form.mandarin for form in table if form.mandarin not in wanted]


def _extra_manifest(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    return {"audio_license_source": audio_license_source(AUDIO_TSV_PATH)}


def build_attribution(
    rows: Sequence[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
) -> list[str]:
    """Build the body lines of the Mandarin pack section.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author.
    :returns: Markdown lines of the ``## Mandarin pack`` section body.
    """
    from collections import defaultdict

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        author = row.get("tatoeba_author", "").strip() or "unknown"
        grouped[author].append(int(row["tatoeba_id"]))

    lines = [
        "Sentences come from Tatoeba (<https://tatoeba.org>) and use the "
        "CC BY 2.0 FR license.",
        "",
        "### Word data",
        "",
        f"The Swadesh Mandarin list comes from <{SWADESH_URL}>.",
        "It uses the CC BY-SA 4.0 license.",
        "The reviewed forms are in `vocab/english-mandarin.csv`.",
        "",
        "### Sentence text",
        "",
        "The Tatoeba Mandarin sentences below use the CC BY 2.0 FR",
        "license. Each number is a Tatoeba sentence id.",
    ]
    for author in sorted(grouped):
        ids = ", ".join(str(value) for value in sorted(grouped[author]))
        lines.extend(["", f"#### {author}", "", ids])

    lines.extend(
        [
            "",
            "### Audio",
            "",
            f"Engine: {tts_engine_name(voice)}. Voice: {voice}.",
            "Each file is synthesized, so the engine and the voice give",
            "the license.",
        ]
    )

    native = [row for row in rows if row.get("audio_source") == "native"]
    if native:
        authors = audio_authors or {}
        lines.extend(
            [
                "",
                "### Native recordings",
                "",
                "Tatoeba recordings keep their own license.",
                "",
                "| Audio id | Author | License |",
                "| -------- | ------ | ------- |",
            ]
        )
        for row in sorted(native, key=lambda item: int(item["audio_id"])):
            audio_id = int(row["audio_id"])
            author = authors.get(audio_id, "") or row.get("tatoeba_author", "")
            license_text = row.get("audio_license", "") or "per recording"
            lines.append(f"| {audio_id} | {author} | {license_text} |")

    return lines


def write_attribution(
    rows: Sequence[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
    path: Path = ATTRIBUTION_PATH,
) -> None:
    """Write ``ATTRIBUTION.md`` for the Mandarin pack.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author.
    :param path: Output Markdown file.
    """
    write_attribution_section(
        Path(path),
        "Mandarin",
        build_attribution(rows, voice, audio_authors),
        header=["# Attribution", ""],
    )


MANDARIN = register(
    LanguageProfile(
        name="mandarin",
        lang_tag="",
        tatoeba_lang="cmn",
        english_lang="eng",
        word_csv=DEFAULT_CSV_PATH,
        sentence_csv=SENTENCES_PATH,
        sentence_review_csv=SENTENCE_REVIEW_PATH,
        pool_report=POOL_REPORT_PATH,
        audio_dir=AUDIO_DIR,
        word_review_csv=DEFAULT_REVIEW_PATH,
        deck_name=DECK_NAME,
        model_name=MODEL_NAME,
        output_apkg=Path("apkg/swadesh_mandarin_sentences.apkg"),
        manifest_path=Path("apkg/swadesh_mandarin_sentences.manifest.json"),
        fields=FIELDS,
        sort_field_index=SORT_FIELD_INDEX,
        front_template=FRONT_TEMPLATE,
        back_template=BACK_TEMPLATE,
        char_links=True,
        deck_description=DECK_DESCRIPTION,
        licenses=(
            ("Swadesh word data", "CC BY-SA 4.0"),
            ("Tatoeba sentences", "CC BY 2.0 FR"),
            ("Tatoeba audio", "per recording"),
        ),
        extra_css=EXTRA_CSS,
        min_units=4,
        max_units=7,
        unit_name="han",
        fallback_min_units=FALLBACK_MIN_HAN,
        preferred_min_units=PREFERRED_MIN_HAN,
        preferred_max_units=PREFERRED_MAX_HAN,
        final_marks=FINAL_MARKS,
        internal_marks=INTERNAL_MARKS,
        quote_marks=QUOTE_MARKS,
        comma_marks="，,",
        orphan_conjunctions=ORPHAN_CONJUNCTIONS,
        wordfreq_lang="zh",
        tokenizer="",
        native_audio=True,
        prefer_native_audio=True,
        voice="zh-CN-XiaoxiaoNeural",
        tts_lang="zh-CN",
        gtts_voice="gtts:zh-CN",
        swadesh_url=SWADESH_URL,
        swadesh_license="CC BY-SA 4.0",
        tatoeba_license="CC BY 2.0 FR",
        core_concepts=SWADESH_ROWS,
        zipf_min=MIN_ZIPF,
        similarity_limit=SIMILARITY_LIMIT,
        coverage_min=0.80,
        prepare_text=convert_to_simplified,
        unit_count=lambda sentence: len(get_chinese_chars(sentence)),
        script_ok=lambda sentence: not LATIN_OR_DIGIT.search(sentence),
        is_repeated=is_repeated_zh_sentence,
        has_predicate=has_predicate,
        has_orphan=has_orphan_conjunction,
        hits=single_hits,
        match_count=count_form_occurrences,
        candidate_length=lambda card: card.han_len,
        count_unknown=lambda sentence, known: sum(
            1 for word in han_tokens(sentence) if word not in known
        ),
        zipf_words=han_tokens,
        similarity=char_similarity_zh,
        dedupe_key=lambda sentence: sentence,
        similarity_keys=lambda sentence: set(get_chinese_chars(sentence)),
        form_text=lambda word: word.mandarin,
        form_positions=_form_positions,
        gloss=_gloss,
        card_fields=_card_fields,
        card_order=_card_order,
        excluded_words=_excluded_words,
        extra_manifest=_extra_manifest,
        attribution=write_attribution,
        sense_filter=sense_filter,
        false_friend_spans=false_friend_spans,
        finalize_candidate=lambda card, target: replace(
            card, pinyin=sentence_pinyin(card.sentence)
        ),
    )
)


def _review_row_status(
    form: WordForm, pool_count: int, core_count: int, per_word: int
) -> str | None:
    return review_status(form, pool_count, core_count, per_word, MIN_ZIPF)


def sentence_row(card: Candidate) -> dict[str, Any]:
    """Build one sentence CSV row of a card.

    :param card: Selected card.
    :returns: Mapping from CSV column to value.
    """
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
    """Write the card rows of the sentence CSV.

    :param cards: Cards from the selection stage, in output order.
    :param path: Output CSV file.
    :returns: The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_FIELDS)
        writer.writeheader()
        for card in cards:
            writer.writerow(sentence_row(card))
    return path


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
    ``per_word`` candidates), or ``low_zipf`` (Zipf value below 3.0).

    :param pool: Candidate list from the pool stage.
    :param word_table: Word table from :func:`load_word_table`.
    :param path: Output CSV file.
    :param per_word: Card limit per form, used for the thin test.
    :param top_n: Candidate rows per word.
    :returns: The written path.
    """
    from anki_tools.sentence_pack.select import review_candidates

    pool_counts, core_counts, by_form = review_candidates(pool, MANDARIN)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_REVIEW_FIELDS)
        writer.writeheader()
        for form in word_table:
            status = _review_row_status(
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


__all__ = [
    "Counter",
    "MANDARIN",
    "Candidate",
    "WordForm",
    "rank_order",
]
