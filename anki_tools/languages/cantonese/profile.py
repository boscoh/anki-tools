"""Cantonese language pack of the shared sentence pipeline.

The module holds the Cantonese word table, the jyutping helper, the jieba
tokenizer that uses the Cifu dictionary, and the Cantonese language profile.
The shared stages read the profile, so no shared module contains a test on the
language name.

The sentence text uses Traditional characters. :func:`prepare_text` converts a
simplified Tatoeba row to Traditional, and the word table uses Traditional
forms too. A Cantonese sentence has no part-of-speech tagger, so filters F9 and
F10 always pass. ``specs/cantonese-swadesh-sentence-pack.md`` records that
decision.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

import jieba
import opencc
from wordfreq import top_n_list, zipf_frequency

from anki_tools.apkg.templates import TEMPLATES_DIR, load_text
from anki_tools.languages.cantonese.jyutping import get_jyutping
from anki_tools.languages.mandarin.characters import (
    char_similarity_zh,
    get_chinese_chars,
    is_chinese_char,
    is_repeated_zh_sentence,
)
from anki_tools.sentence_pack.pack import (
    ATTRIBUTION_PATH,
    audio_license_source,
    tts_engine_name,
    write_attribution_section,
)
from anki_tools.sentence_pack.guards import (
    compound_spans,
    inside_spans,
    occurrences,
)
from anki_tools.sentence_pack.profile import LanguageProfile, make_sense_filter, register
from anki_tools.sentence_pack.select import rank_order
from anki_tools.sentence_pack.word_table import (
    check_form_count,
    check_form_labels,
    check_missing_indices,
    check_review_file,
    duplicate_forms,
    form_positions as shared_form_positions,
    read_csv_rows,
    read_swadesh_labels as shared_read_swadesh_labels,
    read_word_review as shared_read_word_review,
    summarize_word_table,
)
from anki_tools.sentence_pack.cards import (
    build_sentence_row,
    write_review_rows,
    write_sentence_rows,
)
from anki_tools.sentence_pack.wiktionary import (
    fetch_module_terms as shared_fetch_module_terms,
    parse_module_terms_lines,
    read_module_terms,
)
from anki_tools.languages.cantonese.frequency import (
    build_yue_tokenizer,
    cifu_path,
    download_cifu,
    read_cifu,
    read_overrides,
)

DEFAULT_CSV_PATH = Path("vocab/english-cantonese.csv")
DEFAULT_SWADESH_PATH = Path("vocab/swadesh.txt")
DEFAULT_REVIEW_PATH = Path("vocab/cantonese_word_review.csv")
DEFAULT_VARIANT_ROWS_PATH = Path("vocab/cantonese_variant_rows.csv")
SENTENCES_PATH = Path("vocab/cantonese_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/cantonese_sentences_review.csv")
POOL_REPORT_PATH = Path("data/cantonese_pool_report.json")
AUDIO_DIR = Path("audio/cantonese_sentences")
AUDIO_TSV_PATH = Path("data/tatoeba/yue_sentences_with_audio.tsv")
MODULE_CACHE_PATH = Path("data/swadesh/cantonese_module.txt")
FREQUENCY_PATH = Path("data/cantonese_frequency.tsv")

SWADESH_ROWS = 207
MIN_FORMS = 205
MAX_FORMS = 220
MIN_ZIPF = 3.0
OPEN_STATUS = "open"
KNOWN_WORDS_TOP_N = 1000
KNOWN_YUE_ROWS = 2000
SIMILARITY_LIMIT = 0.67
PREFERRED_MIN_HAN = 4
PREFERRED_MAX_HAN = 8
FALLBACK_MIN_HAN = 3

FINAL_MARKS = "。？！.!?"
QUOTE_MARKS = '"“”「」『』'
INTERNAL_MARKS = "，：；,:;"
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9０-９]")

SENTENCE_FIELDS = [
    "target",
    "sentence",
    "text_original",
    "jyutping",
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
    "jyutping",
    "english_text",
    "tatoeba_id",
    "unknown_words",
    "han_len",
    "rank_key",
    "tier",
]
WORD_REVIEW_FIELDS = [
    "index",
    "english",
    "current",
    "module_terms",
    "chosen",
    "issue",
    "fix",
    "status",
]

MODEL_NAME = "Swadesh Cantonese Sentences"
DECK_NAME = "Swadesh Cantonese Sentences"
FIELDS = (
    "Audio",
    "Sentence",
    "Jyutping",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
)
SORT_FIELD_INDEX = 4
DECK_DESCRIPTION = (
    "Word data CC BY-SA 4.0. Sentence text CC BY 2.0 FR. "
    "Audio: native recordings and gTTS Cantonese."
)
SWADESH_URL = "https://en.wiktionary.org/wiki/Module:Swadesh/data/yue"
SWADESH_PAGE = "Module:Swadesh/data/yue"
TEMPLATE_DIR = TEMPLATES_DIR / "sentence_pack" / "cantonese"

FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
EXTRA_CSS = load_text(TEMPLATE_DIR / "card.css")

_s2t: opencc.OpenCC | None = None
_t2s: opencc.OpenCC | None = None


@dataclass
class WordForm:
    """One reviewed Cantonese form of a Swadesh concept.

    :param swadesh_index: CSV row order, 1-based.
    :param english: Swadesh English label of the row.
    :param cantonese: Traditional Cantonese form, the unique key of the row.
    :param jyutping: Reviewed jyutping reading of the form.
    :param variant: ``primary`` or ``alternate``.
    :param form_kind: ``single`` or ``phrase``.
    :param zipf: ``wordfreq`` Zipf value of the simplified form for ``zh``.
    :param candidate_count: Sentence count after the S2 filters.
    :param tier: ``core`` or ``scaffold``.
    """

    swadesh_index: int
    english: str
    cantonese: str
    jyutping: str
    variant: str
    form_kind: str
    zipf: float
    candidate_count: int = 0
    tier: str = "core"


@dataclass(frozen=True)
class Candidate:
    """One candidate sentence for the Cantonese sentence pack.

    :param target: Assigned Swadesh form, the lowest covered form by index.
    :param sentence: Prepared Traditional Cantonese sentence.
    :param text_original: Sentence text before the form conversion.
    :param jyutping: Jyutping reading of the sentence, filled by S3.
    :param english: Paired English translation.
    :param tatoeba_id: Tatoeba sentence id.
    :param tatoeba_author: Tatoeba author name, empty when unknown.
    :param english_id: Tatoeba English sentence id.
    :param audio_id: Native Tatoeba audio id, 0 when the pool holds none.
    :param audio_source: ``tts`` or ``native``.
    :param audio_license: Recording license of a native file, else empty.
    :param unknown_words: Token count outside the known set.
    :param han_len: Han character count of the sentence.
    :param zipf_mean: Mean Zipf value of the sentence tokens.
    :param rank_key: Serialized ranking key.
    :param tier: ``core`` or ``scaffold``.
    :param audio_missing: True when the sentence has no audio file yet.
    :param covered: Forms that appear exactly one time.
    """

    target: str
    sentence: str
    text_original: str
    jyutping: str
    english: str
    tatoeba_id: int
    tatoeba_author: str
    english_id: int
    audio_id: int
    audio_source: str
    audio_license: str
    unknown_words: int
    han_len: int
    zipf_mean: float
    rank_key: str
    tier: str
    audio_missing: bool
    covered: frozenset[str] = frozenset()


def _traditional_converter() -> opencc.OpenCC:
    global _s2t
    if _s2t is None:
        _s2t = opencc.OpenCC("s2t")
    return _s2t


def _simplified_converter() -> opencc.OpenCC:
    global _t2s
    if _t2s is None:
        _t2s = opencc.OpenCC("t2s")
    return _t2s


def convert_to_traditional(text: str) -> str:
    """Convert simplified Chinese to Traditional Chinese.

    :param text: Sentence text.
    :returns: Text with Traditional characters.
    """
    return _traditional_converter().convert(text)


def convert_to_simplified(text: str) -> str:
    """Convert Traditional Chinese to simplified Chinese.

    :param text: Sentence text.
    :returns: Text with simplified characters.
    """
    return _simplified_converter().convert(text)


def prepare_text(text: str) -> str:
    """Normalize Cantonese text and convert it to Traditional characters.

    NFC is applied, runs of spaces collapse, and the simplified characters of a
    Tatoeba row become Traditional. The case and the punctuation stay.

    :param text: Raw Cantonese text.
    :returns: NFC Traditional text with normal spaces.
    """
    text = unicodedata.normalize("NFC", text)
    return " ".join(convert_to_traditional(text).split())


def read_swadesh_labels(swadesh_path: Path = DEFAULT_SWADESH_PATH) -> list[str]:
    """Read the Swadesh English labels in index order."""
    return shared_read_swadesh_labels(swadesh_path)


def form_kind(form: str) -> str:
    """Give the shape class of one form.

    :param form: Cantonese form.
    :returns: ``single`` for one written word, else ``phrase``.
    """
    return "single" if " " not in form.strip() else "phrase"


def read_variant_rows(path: Path = DEFAULT_VARIANT_ROWS_PATH) -> list[dict[str, str]]:
    """Read the alternate form rows of the Cantonese word table.

    :param path: CSV file of variant rows. A missing file gives no rows.
    :returns: One mapping per row.
    """
    if not Path(path).exists():
        return []
    return read_csv_rows(path)


def load_word_table(
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    variant_path: Path = DEFAULT_VARIANT_ROWS_PATH,
) -> list[WordForm]:
    """Load the reviewed Cantonese word table.

    :param csv_path: Reviewed word CSV with columns
        ``english,cantonese,transliteration``.
    :param swadesh_path: File with one English label per line.
    :param variant_path: Optional CSV of alternate forms. A variant row
        carries ``swadesh_index``, ``cantonese``, and ``transliteration``. It
        joins the table as ``variant=alternate``, so it gets its own cards
        and the primary form keeps the first place.
    :raises ValueError: The row count or a form is wrong.
    :returns: One :class:`WordForm` per concept.
    """
    rows = read_csv_rows(csv_path)
    labels = read_swadesh_labels(swadesh_path)
    if len(rows) != len(labels):
        raise ValueError(f"{csv_path}: {len(rows)} rows but {len(labels)} labels")
    table: list[WordForm] = []
    for position, row in enumerate(rows, start=1):
        form = prepare_text(row.get("cantonese", ""))
        if not form:
            raise ValueError(f"row {position}: empty Cantonese form")
        table.append(
            WordForm(
                swadesh_index=int(row.get("swadesh_index") or position),
                english=row.get("english") or labels[position - 1],
                cantonese=form,
                jyutping=row.get("transliteration", "").strip(),
                variant=row.get("variant") or "primary",
                form_kind=row.get("form_kind") or form_kind(form),
                zipf=zipf_frequency(convert_to_simplified(form), "zh"),
                candidate_count=int(row.get("candidate_count") or 0),
                tier=row.get("tier") or "core",
            )
        )
    for row in read_variant_rows(variant_path):
        index = int(row["swadesh_index"])
        form = prepare_text(row.get("cantonese", ""))
        if not form:
            raise ValueError(f"variant row {index}: empty Cantonese form")
        table.append(
            WordForm(
                swadesh_index=index,
                english=labels[index - 1],
                cantonese=form,
                jyutping=row.get("transliteration", "").strip(),
                variant=row.get("variant") or "alternate",
                form_kind=form_kind(form),
                zipf=zipf_frequency(convert_to_simplified(form), "zh"),
            )
        )
    return table


def fetch_module_terms(
    cache_path: Path = MODULE_CACHE_PATH, *, refresh: bool = False
) -> Path:
    """Download the Cantonese Swadesh module wikitext into the cache."""
    return shared_fetch_module_terms(SWADESH_PAGE, cache_path, refresh=refresh)


def parse_module_terms(text: str) -> dict[int, tuple[str, ...]]:
    """Parse the Swadesh module wikitext into terms per index."""
    return parse_module_terms_lines(text)


@lru_cache(maxsize=1)
def module_terms(cache_path: Path = MODULE_CACHE_PATH) -> dict[int, tuple[str, ...]]:
    """Read the cached module terms, with a download on the first run."""
    return read_module_terms(SWADESH_PAGE, cache_path, parse_module_terms_lines)

def read_word_review(review_path: Path = DEFAULT_REVIEW_PATH) -> list[dict[str, str]]:
    """Read the Cantonese word review CSV."""
    return shared_read_word_review(review_path)


def _module_forms(terms: dict[int, tuple[str, ...]], index: int) -> set[str]:
    return {prepare_text(term) for term in terms.get(index, ())}


def check_word_table(
    table: list[WordForm],
    csv_path: Path = DEFAULT_CSV_PATH,
    swadesh_path: Path = DEFAULT_SWADESH_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
) -> list[str]:
    """Check the Cantonese word table against the reviewed files.

    :param table: Word table from :func:`load_word_table`.
    :param csv_path: Reviewed word CSV.
    :param swadesh_path: File with one English label per line.
    :param review_path: Review CSV that must exist and have no open row.
    :returns: Failure strings. An empty list means every check passes.
    """
    failures: list[str] = []
    labels = read_swadesh_labels(swadesh_path)
    failures.extend(
        check_form_count(
            table, min_forms=MIN_FORMS, max_forms=MAX_FORMS, label="row count"
        )
    )
    failures.extend(check_missing_indices(table, swadesh_rows=SWADESH_ROWS))
    duplicates = duplicate_forms(table, key=lambda form: form.cantonese)
    if duplicates:
        failures.append(f"duplicate forms: {duplicates}")
    failures.extend(check_form_labels(table, labels))
    for form in table:
        if not 1 <= form.swadesh_index <= len(labels):
            continue
        if not form.jyutping:
            failures.append(f"form {form.cantonese}: empty jyutping")
        if any(char.isascii() and char.isalnum() for char in form.cantonese):
            failures.append(f"form {form.cantonese}: Latin letter or digit")
    failures.extend(check_review_file(review_path))
    review = read_word_review(review_path)
    reviewed = {row.get("index", "") for row in review if row.get("chosen", "")}
    try:
        terms = module_terms()
    except RuntimeError as error:
        failures.append(f"module terms unavailable: {error}")
        return failures
    for form in table:
        if form.cantonese in _module_forms(terms, form.swadesh_index):
            continue
        if str(form.swadesh_index) in reviewed:
            continue
        failures.append(
            f"form {form.cantonese}: not in the module row set of index "
            f"{form.swadesh_index} and no review row"
        )
    return failures


def word_table_report(table: list[WordForm]) -> dict[str, Any]:
    """Summarize the Cantonese word table.

    :param table: Word table from :func:`load_word_table`.
    :returns: Counts and the low Zipf forms.
    """
    return summarize_word_table(
        table,
        form_key=lambda form: form.cantonese,
        zipf_min=MIN_ZIPF,
        extra=_report_extras,
    )


def _report_extras(table: list[WordForm]) -> dict[str, Any]:
    return {
        "phrase_forms": sum(1 for form in table if form.form_kind == "phrase"),
    }


def _tokenizer_from_cifu() -> jieba.Tokenizer:
    path = cifu_path()
    if not path.exists():
        path = download_cifu(path)
    cifu = read_cifu(path)
    extra: dict[str, str] = dict(read_overrides())
    for form in load_word_table():
        extra.setdefault(form.cantonese, form.cantonese)
    return build_yue_tokenizer(cifu, extra_words=extra)


@lru_cache(maxsize=1)
def tokenizer() -> jieba.Tokenizer:
    """Build the Cantonese tokenizer once.

    The tokenizer uses the Cifu dictionary and the frequency overrides. It does
    not change the global jieba dictionary.

    :raises RuntimeError: Cifu is absent and cannot be downloaded.
    :returns: The jieba tokenizer of the pack.
    """
    try:
        return _tokenizer_from_cifu()
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "the Cantonese tokenizer needs the Cifu lexicon. Run "
            "`uv run anki freq yue` to download it."
        ) from error


def han_tokens(sentence: str) -> list[str]:
    """Tokenize a sentence and keep Han-only tokens.

    :param sentence: Traditional Cantonese sentence.
    :returns: One token per written word, in sentence order.
    """
    return [word for word, _, _ in han_spans(sentence)]


@lru_cache(maxsize=50_000)
def han_spans(sentence: str) -> tuple[tuple[str, int, int], ...]:
    """Tokenize a sentence and keep the offsets of its Han tokens.

    :param sentence: Traditional Cantonese sentence.
    :returns: One ``(token, start, end)`` tuple per Han token.
    """
    return tuple(
        (word, start, end)
        for word, start, end in tokenizer().tokenize(sentence)
        if word and all(is_chinese_char(char) for char in word)
    )


def han_stream(sentence: str) -> str:
    """Join the Han characters of a sentence.

    :param sentence: Traditional Cantonese sentence.
    :returns: The Han characters in order, with no punctuation.
    """
    return "".join(get_chinese_chars(sentence))


@lru_cache(maxsize=1)
def table_forms() -> frozenset[str]:
    """Give the forms of the reviewed word table.

    :returns: One entry per form of the pack.
    """
    return frozenset(form.cantonese for form in load_word_table())


def _token_at(spans: Sequence[tuple[str, int, int]], position: int) -> str:
    for word, start, end in spans:
        if start <= position < end:
            return word
    return ""


FALSE_FRIENDS: dict[str, tuple[str, ...]] = {
    "切": ("趕得切", "一切"),
    "聽": ("聽日",),
    "數": ("着數", "數學", "彩數", "多數"),
    "燒": ("發燒",),
    "揸": ("揸車",),
}


SENSE_CHECKS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "揸": (
        frozenset({
            "hold", "holds", "held", "holding", "grip", "grips",
            "grasp", "grasps", "clutch", "clutches",
        }),
        frozenset({
            "drive", "drives", "drove", "driving", "driver", "fly",
            "flies", "flew", "flying", "pilot", "plane", "planes",
        }),
    ),
    "燒": (
        frozenset({"burn", "burns", "burned", "burnt", "burning", "fire"}),
        frozenset({"fever", "fevers", "temperature", "temperatures"}),
    ),
    "數": (
        frozenset({"count", "counts", "counted", "counting"}),
        frozenset({
            "math", "maths", "mathematics", "arithmetic", "number",
            "numbers",
        }),
    ),
}


def false_friend_spans(sentence: str, form: str) -> list[tuple[int, int]]:
    """Give the spans where the form sits inside a different-sense compound.

    :param sentence: Traditional Cantonese sentence.
    :param form: Word-table form.
    :returns: ``(start, end)`` offsets of each matching compound.
    """
    return compound_spans(sentence, FALSE_FRIENDS.get(form, ()))


sense_filter = make_sense_filter(SENSE_CHECKS)


def form_offsets(sentence: str, form: str) -> list[int]:
    """Find the form occurrences that count as a target hit.

    Cantonese 我 and 我哋 are different Swadesh concepts. A form matches when
    it holds whole tokens. An occurrence inside a longer word counts only when
    that word is not itself a target form, so 食 hits inside 食飯 but 我 does
    not hit inside 我哋. An occurrence inside a false-friend compound never
    counts, so 聽 does not hit inside 聽日.

    :param sentence: Traditional Cantonese sentence.
    :param form: Word-table form.
    :returns: Start offsets of the matching occurrences.
    """
    if not form:
        return []
    spans = han_spans(sentence)
    starts = {start for _, start, _ in spans}
    ends = {end for _, _, end in spans}
    targets = table_forms()
    blocked = false_friend_spans(sentence, form)
    found: list[int] = []
    for position in occurrences(sentence, form):
        if inside_spans(position, blocked):
            continue
        if position in starts and position + len(form) in ends:
            found.append(position)
        elif _token_at(spans, position) not in targets:
            found.append(position)
    return found


def single_hits(sentence: str, forms: Iterable[str]) -> dict[str, int]:
    """Count the forms that the sentence contains exactly one time.

    :param sentence: Traditional Cantonese sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to 1, for forms with one whole-token occurrence.
    """
    return {form: 1 for form in set(forms) if len(form_offsets(sentence, form)) == 1}


def sentence_hits(sentence: str, forms: Iterable[str]) -> dict[str, int]:
    """Count the whole-token forms that the sentence contains.

    :param sentence: Traditional Cantonese sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to its occurrence count.
    """
    hits: dict[str, int] = {}
    for form in set(forms):
        count = len(form_offsets(sentence, form))
        if count:
            hits[form] = count
    return hits


def count_form_occurrences(sentence: str, form: str) -> int:
    """Count the whole-token occurrences of a form.

    :param sentence: Traditional Cantonese sentence.
    :param form: Word-table form.
    :returns: Occurrence count of the form.
    """
    return len(form_offsets(sentence, form))


def highlight(sentence: str, form: str) -> str:
    """Wrap the whole-token matches of a card target in a bold tag.

    :param sentence: Traditional Cantonese sentence.
    :param form: Card target.
    :returns: Sentence with the matched tokens in ``<b>`` tags.
    """
    offsets = form_offsets(sentence, form)
    for position in reversed(offsets):
        end = position + len(form)
        sentence = (
            f"{sentence[:position]}<b>{sentence[position:end]}</b>{sentence[end:]}"
        )
    return sentence


def has_predicate(_sentence: str) -> bool:
    """Filter F9 is off, because jieba has no Cantonese tagger.

    :param sentence: Traditional Cantonese sentence.
    :returns: Always True.
    """
    return True


def has_orphan_conjunction(_sentence: str) -> bool:
    """Filter F10 is off, because the yue selection spec did not test it.

    :param sentence: Traditional Cantonese sentence.
    :returns: Always False.
    """
    return False


def count_unknown(sentence: str, known: Collection[str]) -> int:
    """Count the tokens outside the known word set.

    :param sentence: Traditional Cantonese sentence.
    :param known: Known word set.
    :returns: Unknown token count.
    """
    return sum(1 for token in han_tokens(sentence) if token not in known)


def zipf_words(sentence: str) -> list[str]:
    """Give the tokens of a sentence for the ``wordfreq`` lookup.

    ``wordfreq`` has no Cantonese word list, so the words are simplified and
    the lookup uses ``zh``.

    :param sentence: Traditional Cantonese sentence.
    :returns: Simplified token list.
    """
    return [convert_to_simplified(token) for token in han_tokens(sentence)]


def script_ok(sentence: str) -> bool:
    """Check filter F3: no Latin letter and no digit.

    :param sentence: Traditional Cantonese sentence.
    :returns: True when the sentence has no Latin letter or digit.
    """
    return not LATIN_OR_DIGIT.search(sentence)


def unit_count(sentence: str) -> int:
    """Count the Han characters of a sentence.

    :param sentence: Traditional Cantonese sentence.
    :returns: Han character count.
    """
    return len(get_chinese_chars(sentence))


def sentence_jyutping(sentence: str) -> str:
    """Read a sentence with the ToJyutping tool.

    :param sentence: Traditional Cantonese sentence.
    :returns: Space-separated jyutping, or an empty string without the tool.
    """
    return get_jyutping(sentence)


def form_positions(table: Sequence[WordForm]) -> dict[str, tuple[int, int]]:
    """Give each form its Swadesh index and its variant rank.

    :param table: Reviewed word table.
    :returns: Mapping from form to ``(swadesh_index, variant_rank)``.
    """
    return shared_form_positions(table, key=lambda form: form.cantonese)


def known_words(word_table: Sequence[WordForm]) -> set[str]:
    """Build the known word set of filter F11.

    The set holds the Swadesh forms and the most frequent Cantonese words. The
    frequency list of ``anki freq yue`` applies when it exists. Otherwise the
    set uses the top ``wordfreq`` Chinese words in Traditional characters.

    :param word_table: Word table from :func:`load_word_table`.
    :returns: Known word set.
    """
    known = {form.cantonese for form in word_table}
    if Path(FREQUENCY_PATH).exists():
        rows = read_csv_rows_tsv(FREQUENCY_PATH, KNOWN_YUE_ROWS)
        known.update(row["word"] for row in rows)
    else:
        known.update(prepare_text(word) for word in top_n_list("zh", KNOWN_WORDS_TOP_N))
    return known


def read_csv_rows_tsv(path: Path, limit: int) -> list[dict[str, str]]:
    """Read the first rows of a TSV file.

    :param path: TSV file with a header row.
    :param limit: Maximum data rows to read.
    :returns: One mapping per row.
    """
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if len(rows) >= limit:
                break
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


def make_candidate(
    facts: Any, target: str, tier: str, covered: frozenset[str]
) -> Candidate:
    """Build one Cantonese candidate card from the measured facts.

    :param facts: Sentence facts of the shared pool stage.
    :param target: Assigned word-table form.
    :param tier: ``core`` or ``scaffold``.
    :param covered: Forms that the sentence covers.
    :returns: The candidate card.
    """
    from anki_tools.sentence_pack.select import rank_key_parts, serialize_rank_key

    candidate = Candidate(
        target=target,
        sentence=facts.sentence,
        text_original=facts.text_original,
        jyutping="",
        english=facts.english,
        tatoeba_id=facts.tatoeba_id,
        tatoeba_author=facts.tatoeba_author,
        english_id=facts.english_id,
        audio_id=facts.audio_id,
        audio_license=facts.audio_license,
        audio_source="native" if facts.audio_id else "tts",
        unknown_words=facts.unknown_words,
        han_len=facts.unit_len,
        zipf_mean=facts.zipf_mean,
        rank_key="",
        tier=tier,
        audio_missing=not facts.audio_id,
        covered=covered,
    )
    return replace(
        candidate, rank_key=serialize_rank_key(rank_key_parts(candidate, CANTONESE))
    )


def glossary(word: WordForm) -> str:
    """Build the card gloss of one Cantonese form.

    :param word: Reviewed word form.
    :returns: Gloss text with the jyutping reading.
    """
    return f"{word.english} · {word.jyutping}" if word.jyutping else word.english


def card_fields(row: dict[str, str], _word: WordForm) -> dict[str, str]:
    """Build the language-specific note fields of one card.

    :param row: Sentence CSV row.
    :param word: Reviewed word form.
    :returns: Mapping from the ``Jyutping`` field to its value.
    """
    return {"Jyutping": row.get("jyutping", "")}


def card_order(word: WordForm, row: dict[str, str]) -> tuple:
    """Build the note order key of one card.

    :param word: Reviewed word form.
    :param row: Sentence CSV row.
    :returns: Swadesh index, variant rank, then Tatoeba id.
    """
    variant_rank = 0 if word.variant == "primary" else 1
    return (word.swadesh_index, variant_rank, int(row["tatoeba_id"]))


def excluded_words(table: Sequence[WordForm], targets: Collection[str]) -> list[str]:
    """List the word-table forms that get no card.

    :param table: Reviewed word table.
    :param targets: Card target strings of the pack.
    :returns: Forms that the pack leaves out.
    """
    wanted = set(targets)
    return [form.cantonese for form in table if form.cantonese not in wanted]


def extra_manifest(_rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Add the Cantonese manifest keys.

    :param rows: Sentence CSV rows that entered the pack.
    :returns: The pack license source counts.
    """
    return {"audio_license_source": audio_license_source(AUDIO_TSV_PATH)}


def build_attribution(
    rows: Sequence[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
) -> list[str]:
    """Build the body lines of the Cantonese pack section.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author.
    :returns: Markdown lines of the ``## Cantonese pack`` section body.
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
        f"The Swadesh Cantonese list comes from <{SWADESH_URL}>.",
        "It uses the CC BY-SA 4.0 license.",
        "The reviewed forms are in `vocab/english-cantonese.csv`.",
        "",
        "### Sentence text",
        "",
        "The Tatoeba Cantonese sentences below use the CC BY 2.0 FR",
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
            "Each synthesized file gives its license through the engine and",
            "the voice. Native recordings keep their own license.",
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
    """Write ``ATTRIBUTION.md`` for the Cantonese pack.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author.
    :param path: Output Markdown file.
    """
    write_attribution_section(
        Path(path),
        "Cantonese",
        build_attribution(rows, voice, audio_authors),
        header=["# Attribution", ""],
    )


CANTONESE = register(
    LanguageProfile(
        name="cantonese",
        lang_tag="yue",
        tatoeba_lang="yue",
        english_lang="eng",
        word_csv=DEFAULT_CSV_PATH,
        sentence_csv=SENTENCES_PATH,
        sentence_review_csv=SENTENCE_REVIEW_PATH,
        pool_report=POOL_REPORT_PATH,
        audio_dir=AUDIO_DIR,
        word_review_csv=DEFAULT_REVIEW_PATH,
        deck_name=DECK_NAME,
        model_name=MODEL_NAME,
        output_apkg=Path("apkg/swadesh_cantonese_sentences.apkg"),
        manifest_path=Path("apkg/swadesh_cantonese_sentences.manifest.json"),
        fields=FIELDS,
        sort_field_index=SORT_FIELD_INDEX,
        front_template=FRONT_TEMPLATE,
        back_template=BACK_TEMPLATE,
        char_links=True,
        deck_description=DECK_DESCRIPTION,
        licenses=(
            ("Cantonese Swadesh word data", "CC BY-SA 4.0"),
            ("Tatoeba sentences", "CC BY 2.0 FR"),
            ("Tatoeba audio", "per recording"),
        ),
        extra_css=EXTRA_CSS,
        min_units=3,
        max_units=12,
        unit_name="han",
        fallback_min_units=FALLBACK_MIN_HAN,
        preferred_min_units=PREFERRED_MIN_HAN,
        preferred_max_units=PREFERRED_MAX_HAN,
        final_marks=FINAL_MARKS,
        internal_marks=INTERNAL_MARKS,
        quote_marks=QUOTE_MARKS,
        comma_marks="，,",
        orphan_conjunctions=(),
        wordfreq_lang="zh",
        tokenizer="",
        native_audio=True,
        prefer_native_audio=True,
        voice="zh-HK-HiuMaanNeural",
        tts_lang="yue",
        gtts_voice="gtts:yue",
        swadesh_url=SWADESH_URL,
        swadesh_license="CC BY-SA 4.0",
        tatoeba_license="CC BY 2.0 FR",
        core_concepts=SWADESH_ROWS,
        zipf_min=MIN_ZIPF,
        similarity_limit=SIMILARITY_LIMIT,
        coverage_min=0.70,
        prepare_text=prepare_text,
        unit_count=unit_count,
        script_ok=script_ok,
        is_repeated=is_repeated_zh_sentence,
        has_predicate=has_predicate,
        has_orphan=has_orphan_conjunction,
        hits=single_hits,
        match_count=count_form_occurrences,
        candidate_length=lambda card: card.han_len,
        count_unknown=count_unknown,
        zipf_words=zipf_words,
        similarity=char_similarity_zh,
        dedupe_key=lambda sentence: sentence,
        similarity_keys=lambda sentence: set(get_chinese_chars(sentence)),
        form_text=lambda word: word.cantonese,
        form_positions=form_positions,
        gloss=glossary,
        card_fields=card_fields,
        card_order=card_order,
        excluded_words=excluded_words,
        extra_manifest=extra_manifest,
        attribution=write_attribution,
        highlight=highlight,
        sense_filter=sense_filter,
        false_friend_spans=false_friend_spans,
        finalize_candidate=lambda card, _target: replace(
            card, jyutping=sentence_jyutping(card.sentence)
        ),
    )
)


def sentence_row(card: Candidate) -> dict[str, Any]:
    """Build one sentence CSV row of a card.

    :param card: Selected card.
    :returns: Mapping from CSV column to value.
    """
    return build_sentence_row(
        card,
        unit_field="han_len",
        extra_fields=(("jyutping", lambda card: card.jyutping),),
    )


def write_sentence_csv(cards: list[Candidate], path: Path = SENTENCES_PATH) -> Path:
    """Write the card rows of the sentence CSV.

    :param cards: Cards from the selection stage, in output order.
    :param path: Output CSV file.
    :returns: The written path.
    """
    return write_sentence_rows(cards, path, SENTENCE_FIELDS, sentence_row)


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
    return write_review_rows(
        CANTONESE,
        pool,
        word_table,
        path,
        SENTENCE_REVIEW_FIELDS,
        form_key=lambda form: form.cantonese,
        unit_field="han_len",
        zipf_min=MIN_ZIPF,
        per_word=per_word,
        top_n=top_n,
        transliteration_field="jyutping",
        transliteration=sentence_jyutping,
    )


def gate_word_table(
    inputs: Any,
    _profile: LanguageProfile,
) -> tuple[bool, str]:
    """Cantonese gate G1: the reviewed word table passes every S1 check.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    failures = check_word_table(
        table,
        csv_path=inputs.word_csv,
        review_path=inputs.review_csv,
    )
    concepts = len({form.swadesh_index for form in table})
    if failures:
        return False, " | ".join(failures[:3])
    return True, f"{concepts} concepts, {len(table)} forms"


def gate_attribution(
    inputs: Any,
    _profile: LanguageProfile,
) -> tuple[bool, str]:
    """Cantonese gate G12: the Cantonese section names every sentence author.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    path = inputs.attribution_path
    if not path.is_file():
        return False, f"attribution file missing: {path}"
    text = path.read_text(encoding="utf-8")
    authors = {
        row.get("tatoeba_author", "").strip() or "unknown" for row in inputs.rows()
    }
    problems: list[str] = []
    if "## Cantonese pack" not in text:
        problems.append("no '## Cantonese pack' section")
    missing = sorted(author for author in authors if author not in text)
    if missing:
        problems.append(f"{len(missing)} author(s) missing: {', '.join(missing[:3])}")
    if problems:
        return False, " | ".join(problems)
    return True, f"{len(authors)} author(s) named in the Cantonese section"


def gate_script(inputs: Any, _profile: LanguageProfile) -> tuple[bool, str]:
    """Cantonese gate G9: no card holds a Latin letter or a digit.

    :param inputs: Lazy file reader of the audit.
    :param _profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        row.get("sentence", "")
        for row in rows
        if LATIN_OR_DIGIT.search(row.get("sentence", ""))
        or LATIN_OR_DIGIT.search(row.get("target", ""))
    ]
    if bad:
        return False, f"{len(bad)} card(s) hold Latin text: {bad[0]}"
    return True, f"{len(rows)} cards hold Han text only"


def gate_jyutping(
    inputs: Any,
    _profile: LanguageProfile,
) -> tuple[bool, str]:
    """Cantonese gate G17: every card has a jyutping reading.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    missing = [row.get("sentence", "") for row in rows if not row.get("jyutping")]
    if missing:
        return False, f"{len(missing)} card(s) have no jyutping: {missing[0]}"
    return True, f"{len(rows)} cards have a jyutping reading"


def gate_native_license(
    inputs: Any,
    _profile: LanguageProfile,
) -> tuple[bool, str]:
    """Cantonese gate G18: a native recording has a license.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        row.get("sentence", "")
        for row in rows
        if row.get("audio_source") == "native" and not row.get("audio_license")
    ]
    if bad:
        return False, f"{len(bad)} native card(s) have no license: {bad[0]}"
    native = sum(1 for row in rows if row.get("audio_source") == "native")
    return True, f"{native} native card(s) keep a license"


def cantonese_gates() -> tuple[tuple[str, Any], ...]:
    """Give the ordered gate list G1 to G21 of the Cantonese pack.

    :returns: Pairs of gate name and gate function.
    """
    from anki_tools.sentence_pack.audit import (
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
        gate_target_in_table,
        gate_target_once,
        gate_targets_in_table,
        gate_unique_forms,
        gate_unique_tatoeba_ids,
        gate_word_audio,
    )

    return (
        ("G1", gate_word_table),
        ("G2", gate_unique_forms),
        ("G3", gate_targets_in_table),
        ("G4", gate_unique_tatoeba_ids),
        ("G5", gate_target_once),
        ("G6", gate_similarity),
        ("G7", gate_sentence_audio),
        ("G8", gate_media_count),
        ("G9", gate_script),
        ("G10", gate_coverage),
        ("G11", gate_manifest_counts),
        ("G12", gate_attribution),
        ("G13", gate_apkg),
        ("G14", gate_proper_sentences),
        ("G15", gate_review_file),
        ("G16", gate_word_audio),
        ("G17", gate_jyutping),
        ("G18", gate_native_license),
        ("G19", gate_target_in_table),
        ("G20", gate_no_false_friend),
        ("G21", gate_no_sense_mismatch),
    )


def run_cantonese_audit(
    *,
    apkg_path: Path | None = None,
    sentences_csv: Path = SENTENCES_PATH,
    word_csv: Path = DEFAULT_CSV_PATH,
    review_csv: Path = DEFAULT_REVIEW_PATH,
    audio_dir: Path = AUDIO_DIR,
    manifest_path: Path | None = None,
    attribution_path: Path = ATTRIBUTION_PATH,
) -> list[tuple[str, bool, str]]:
    """Run the Cantonese quality gates G1 to G21.

    :param apkg_path: APKG file of the pack.
    :param sentences_csv: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param review_csv: Word review CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param manifest_path: Manifest JSON file.
    :param attribution_path: Attribution Markdown file.
    :returns: One ``(gate, passed, message)`` row per gate, G1 first.
    """
    from anki_tools.sentence_pack.audit import AuditInputs, run_gates

    inputs = AuditInputs(
        CANTONESE,
        apkg_path=apkg_path or CANTONESE.output_apkg,
        sentences_csv=sentences_csv,
        word_csv=word_csv,
        review_csv=review_csv,
        audio_dir=audio_dir,
        manifest_path=manifest_path or CANTONESE.manifest_path,
        attribution_path=attribution_path,
        load_table=lambda csv_path: load_word_table(csv_path),
    )
    return run_gates(cantonese_gates(), inputs, CANTONESE)


__all__ = [
    "CANTONESE",
    "Candidate",
    "WordForm",
    "check_word_table",
    "han_tokens",
    "known_words",
    "load_word_table",
    "module_terms",
    "rank_order",
    "run_cantonese_audit",
    "sentence_jyutping",
    "tokenizer",
]
