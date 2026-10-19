"""Greek language pack of the shared sentence pipeline.

The module holds the Greek word table, the monotonic Greek checks, the lemma
matcher, the Tatoeba module reader, and the Greek language profile. Greek has no
native Tatoeba audio, so the profile sets ``native_audio=False`` and the pack is
TTS only.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Sequence

from wordfreq import zipf_frequency

from anki_tools.rank import greek_text_dedup_key, word_similarity_el
from anki_tools.sentence_pack.audit import AuditInputs, GateRow, sample
from anki_tools.sentence_pack.pack import (
    ATTRIBUTION_PATH,
    attribution_lines,
    write_attribution_section,
)
from anki_tools.sentence_pack.profile import LanguageProfile, register
from anki_tools.sentence_pack.tokens import Token, has_finite_verb, tokenize

WORD_CSV_PATH = Path("vocab/english-greek.csv")
SWADESH_PATH = Path("vocab/swadesh.txt")
WORD_REVIEW_PATH = Path("vocab/greek_word_review.csv")
SENTENCES_PATH = Path("vocab/greek_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/greek_sentences_review.csv")
POOL_REPORT_PATH = Path("data/greek_pool_report.json")
AUDIO_DIR = Path("audio/greek_sentences")
MODULE_CACHE_PATH = Path("data/swadesh/el_module.txt")

MODEL_NAME = "Greek Swadesh Sentence"
DECK_NAME = "Swadesh Greek Sentences"
FIELDS = (
    "Audio",
    "Sentence",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
)
SORT_FIELD_INDEX = 3
DECK_DESCRIPTION = (
    "Word data CC BY-SA 4.0. Sentence text CC BY 2.0 FR. "
    "Audio: synthesized speech (gTTS)."
)
SWADESH_URL = "https://en.wiktionary.org/wiki/Module:Swadesh/data/el"
TOKENIZER = "el_core_news_sm"

FRONT_TEMPLATE = (
    '<div class="front-audio">{{Audio}}</div>\n'
    '<div class="hint">Tap to show the sentence</div>'
)
BACK_TEMPLATE = (
    "{{Audio}}\n"
    "<hr id=answer>\n"
    '<div class="sentence">{{Sentence}}</div>\n'
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

SWADESH_ROWS = 207
UNIQUE_FORMS = 206
SHARED_INDEX = 81
SHARED_FORM = "πόδι"
OPEN_STATUS = "open"
MIN_ZIPF = 3.0
FINAL_MARKS = ".?!;"
INTERNAL_MARKS = ",·:"
QUOTE_MARKS = '«»"“”—–'
ORPHAN_CONJUNCTIONS = ("και", "αλλά", "ή", "γιατί", "επειδή", "λοιπόν", "όμως")
PREDICATE_MARKERS = frozenset({"είναι", "έχει", "έχουν", "δεν", "θα"})
VERB_FORMS: set[str] = set()
ARTICLES = (
    "ο",
    "η",
    "το",
    "οι",
    "τα",
    "τον",
    "την",
    "τη",
    "τους",
    "τις",
    "των",
    "ένας",
    "μία",
    "μια",
    "ένα",
)
NOTE_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
GENDER_BY_ARTICLE = {
    "ο": "m",
    "τον": "m",
    "η": "f",
    "την": "f",
    "τη": "f",
    "το": "n",
}
GREEK_LETTER = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
POLYTONIC = re.compile(r"[\u1f00-\u1fff]")
DIGIT_OR_LATIN = re.compile(r"[A-Za-z0-9]")
CARRIERS = ("{form}", "η {form}", "ο {form}", "το {form}", "τον {form}")

SENTENCE_FIELDS = [
    "target",
    "sentence",
    "text_original",
    "english",
    "tatoeba_id",
    "tatoeba_author",
    "english_id",
    "audio_id",
    "audio_source",
    "audio_license",
    "unknown_words",
    "word_len",
    "zipf_mean",
    "rank_key",
    "tier",
    "form_kind",
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
    "english_text",
    "tatoeba_id",
    "unknown_words",
    "word_len",
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


@dataclass
class WordForm:
    """One reviewed Greek form of a Swadesh concept.

    :param swadesh_index: CSV row order, 1-based.
    :param english: Swadesh English label of the row.
    :param greek: Monotonic Greek form, the unique key of the row.
    :param gender: ``m``, ``f``, ``n``, or empty.
    :param variant: ``primary``, ``alternate``, or ``shared``.
    :param form_kind: ``single`` or ``pair``.
    :param zipf: ``wordfreq`` Zipf value of the form for ``el``.
    :param candidate_count: Sentence count after the S2 filters.
    :param tier: ``core`` or ``scaffold``.
    """

    swadesh_index: int
    english: str
    greek: str
    gender: str
    variant: str
    form_kind: str
    zipf: float
    candidate_count: int = 0
    tier: str = "core"


@dataclass(frozen=True)
class Candidate:
    """One candidate sentence for the Greek sentence pack.

    :param target: Assigned Swadesh form, the lowest covered form by index.
    :param sentence: NFC Greek sentence.
    :param text_original: Sentence text before the NFC normalization.
    :param english: Paired English translation.
    :param tatoeba_id: Tatoeba sentence id.
    :param tatoeba_author: Tatoeba author name, empty when unknown.
    :param english_id: Tatoeba English sentence id.
    :param audio_id: Native Tatoeba audio id, always 0 for Greek.
    :param audio_source: ``tts``.
    :param audio_license: Always empty for a TTS file.
    :param unknown_words: Token count outside the known set.
    :param word_len: Word count of the sentence.
    :param zipf_mean: Mean Zipf value of the sentence tokens.
    :param rank_key: Serialized ranking key.
    :param tier: ``core`` or ``scaffold``.
    :param form_kind: ``single`` or ``pair``.
    :param audio_missing: True when the sentence has no audio file yet.
    :param covered: Forms that appear exactly one time.
    """

    target: str
    sentence: str
    text_original: str
    english: str
    tatoeba_id: int
    tatoeba_author: str
    english_id: int
    audio_id: int
    audio_source: str
    audio_license: str
    unknown_words: int
    word_len: int
    zipf_mean: float
    rank_key: str
    tier: str
    form_kind: str
    audio_missing: bool
    covered: frozenset[str] = frozenset()


def normalized(text: str) -> str:
    """Normalize Greek text to NFC with collapsed spaces.

    :param text: Sentence text.
    :returns: NFC text.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def surface_key(text: str) -> str:
    """Build the surface key of one Greek word.

    :param text: Word text.
    :returns: NFC, casefolded text with final sigma mapped to sigma.
    """
    return unicodedata.normalize("NFC", text).casefold().replace("ς", "σ")


def strip_note(text: str) -> str:
    """Remove a parenthetical note from a form.

    :param text: Raw form, for example ``εσύ (ενικός)``.
    :returns: The form without the note.
    """
    return NOTE_PATTERN.sub(" ", text).strip()


def strip_article(text: str) -> tuple[str, str]:
    """Remove a leading article from a form.

    A one-word form keeps its article, because a word such as ``ένας`` is the
    word for one, not an article plus a noun.

    :param text: Raw form, for example ``η γυναίκα``.
    :returns: The form and the gender that the article gives.
    """
    parts = normalized(text).split()
    if len(parts) < 2:
        return " ".join(parts), ""
    article = parts[0].casefold()
    if article in ARTICLES:
        return " ".join(parts[1:]), GENDER_BY_ARTICLE.get(article, "")
    return " ".join(parts), ""


def clean_form(text: str) -> tuple[str, str]:
    """Clean one CSV form cell.

    :param text: Raw cell, for example ``η γυναίκα`` or ``εσύ (ενικός)``.
    :returns: The cleaned form and the gender of its article.
    """
    return strip_article(strip_note(text))


def form_kind(form: str) -> str:
    """Give the kind of a word-table form.

    :param form: Reviewed form.
    :returns: ``single`` for one token, else ``pair``.
    """
    return "single" if len(form.split()) == 1 else "pair"


def script_problem(text: str) -> str | None:
    """Check the monotonic Greek script rules of one text.

    :param text: Text to check.
    :returns: Failure text, or None when the text passes.
    """
    if unicodedata.normalize("NFC", text) != text:
        return "not NFC"
    if POLYTONIC.search(text):
        return "polytonic mark"
    if DIGIT_OR_LATIN.search(text):
        return "Latin letter or digit"
    for char in text:
        if char.isalpha() and not GREEK_LETTER.match(char):
            return f"letter outside the Greek block: {char}"
    for word in re.findall(r"\w+", text):
        if "ς" in word[:-1]:
            return f"final sigma inside a word: {word}"
    return None


@lru_cache(maxsize=200_000)
def greek_tokens(sentence: str) -> tuple[Token, ...]:
    """Tokenize a Greek sentence once.

    :param sentence: Greek sentence.
    :returns: Tokens of the sentence.
    """
    return tuple(tokenize(TOKENIZER, sentence))


def token_words(sentence: str) -> list[str]:
    """Give the word tokens of a Greek sentence.

    :param sentence: Greek sentence.
    :returns: Token texts without punctuation.
    """
    return [token.text for token in greek_tokens(sentence) if not token.punctuation]


def unit_count(sentence: str) -> int:
    """Count the words of a Greek sentence.

    :param sentence: Greek sentence.
    :returns: Word count without punctuation.
    """
    return len(token_words(sentence))


def script_ok(sentence: str) -> bool:
    """Check filter F3: Greek letters only, no Latin letter and no digit.

    :param sentence: Greek sentence.
    :returns: True when the sentence passes.
    """
    return not DIGIT_OR_LATIN.search(sentence) and not POLYTONIC.search(sentence)


def has_orphan(sentence: str) -> bool:
    """Check filter F10: a sentence-initial conjunction with no comma.

    :param sentence: Greek sentence.
    :returns: True when the leading conjunction has no main clause.
    """
    if "," in sentence or "·" in sentence:
        return False
    return sentence.casefold().startswith(ORPHAN_CONJUNCTIONS)


@lru_cache(maxsize=4096)
def carrier_lemmas(form: str) -> frozenset[str]:
    """Read the citation lemma of a form from its carrier phrases.

    :param form: Reviewed form.
    :returns: Surface keys of the form and of the carrier head words.
    """
    keys = {surface_key(form)}
    for carrier in CARRIERS:
        phrase = carrier.format(form=form)
        tokens = [token for token in tokenize(TOKENIZER, phrase) if not token.punctuation]
        if not tokens:
            continue
        keys.add(surface_key(tokens[-1].text))
        if tokens[-1].lemma:
            keys.add(surface_key(tokens[-1].lemma))
    return frozenset(keys)


def form_keys(form: str) -> frozenset[str]:
    """Build the target key set of one form.

    :param form: Reviewed form.
    :returns: Surface keys that a token may match.
    """
    parts = form.split()
    if len(parts) == 1:
        return carrier_lemmas(form)
    keys: set[str] = set()
    for part in parts:
        keys.update(carrier_lemmas(part))
    return frozenset(keys)


@lru_cache(maxsize=8)
def key_index(forms: frozenset[str]) -> dict[str, frozenset[str]]:
    """Map each target key to the forms that it matches.

    :param forms: Word-table forms.
    :returns: Mapping from a surface key to the forms with that key.
    """
    index: dict[str, set[str]] = {}
    for form in forms:
        for key in form_keys(form):
            index.setdefault(key, set()).add(form)
    return {key: frozenset(value) for key, value in index.items()}


def _phrase_count(tokens: Sequence[Token], phrase: Sequence[str]) -> int:
    wanted = [surface_key(part) for part in phrase]
    surfaces = [surface_key(token.text) for token in tokens]
    count = 0
    size = len(wanted)
    for start in range(len(surfaces) - size + 1):
        if surfaces[start : start + size] == wanted:
            count += 1
    return count


def token_matches(sentence: str, form: str) -> int:
    """Count the matches of one form in a sentence.

    A single-token form matches by surface key or by lemma key. A pair form
    matches only when both parts are adjacent tokens.

    :param sentence: Greek sentence.
    :param form: Reviewed form.
    :returns: Match count.
    """
    tokens = greek_tokens(sentence)
    parts = form.split()
    if len(parts) > 1:
        return _phrase_count(tokens, parts)
    keys = form_keys(form)
    return sum(
        1
        for token in tokens
        if not token.punctuation
        and (
            surface_key(token.text) in keys
            or (token.lemma and surface_key(token.lemma) in keys)
        )
    )


def hits(sentence: str, forms: Collection[str]) -> dict[str, int]:
    """Count the forms that the sentence contains exactly one time.

    :param sentence: Greek sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to 1, for forms with one match.
    """
    tokens = greek_tokens(sentence)
    index = key_index(frozenset(forms))
    counts: dict[str, int] = {}
    for token in tokens:
        if token.punctuation:
            continue
        for key in {surface_key(token.text), surface_key(token.lemma)}:
            for form in index.get(key, ()):
                if len(form.split()) == 1:
                    counts[form] = counts.get(form, 0) + 1
    found = {form: count for form, count in counts.items() if count == 1}
    for form in forms:
        parts = form.split()
        if len(parts) > 1 and form not in found and _phrase_count(tokens, parts) == 1:
            found[form] = 1
    return found


def match_count(sentence: str, form: str) -> int:
    """Count the matches of one form, for the audit gates.

    :param sentence: Greek sentence.
    :param form: Card target.
    :returns: Match count.
    """
    return token_matches(sentence, form)


def count_unknown(sentence: str, known: Collection[str]) -> int:
    """Count the tokens outside the known word set.

    :param sentence: Greek sentence.
    :param known: Known word set, compared by surface and by lemma.
    :returns: Unknown token count.
    """
    keys = {surface_key(word) for word in known}
    unknown = 0
    for token in greek_tokens(sentence):
        if token.punctuation:
            continue
        if surface_key(token.text) in keys:
            continue
        if token.lemma and surface_key(token.lemma) in keys:
            continue
        unknown += 1
    return unknown


def has_predicate(sentence: str) -> bool:
    """Check filter F9: a finite verb, the marker fallback, or a word-list verb.

    :param sentence: Greek sentence.
    :returns: True when the sentence has a finite verb.
    """
    tokens = greek_tokens(sentence)
    if has_finite_verb(list(tokens)):
        return True
    if any(surface_key(token.text) in PREDICATE_MARKERS for token in tokens):
        return True
    if not VERB_FORMS:
        return False
    return bool(hits(sentence, frozenset(VERB_FORMS)))


def glossary(word: WordForm) -> str:
    """Build the card gloss of one Greek form.

    :param word: Reviewed word form.
    :returns: Gloss text, with the gender for a noun.
    """
    label = word.english
    if word.gender:
        return f"{label} ({word.gender})"
    return label


def form_positions(table: Sequence[WordForm]) -> dict[str, tuple[int, int]]:
    """Give each form its Swadesh index and its variant rank.

    :param table: Reviewed word table.
    :returns: Mapping from form to ``(swadesh_index, variant_rank)``.
    """
    positions: dict[str, tuple[int, int]] = {}
    ordered = sorted(
        table,
        key=lambda form: (
            form.swadesh_index,
            0 if form.variant == "primary" else 1,
        ),
    )
    for form in ordered:
        positions[form.greek] = (
            form.swadesh_index,
            0 if form.variant == "primary" else 1,
        )
    return positions


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file into stripped row mappings.

    :param csv_path: CSV file to read.
    :returns: One mapping per data row.
    """
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_swadesh_labels(swadesh_path: Path = SWADESH_PATH) -> list[str]:
    """Read the Swadesh English labels in index order.

    :param swadesh_path: File with one English label per line.
    :returns: Label list, index 1 at position 0.
    """
    text = Path(swadesh_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def load_word_table(
    csv_path: Path = WORD_CSV_PATH, swadesh_path: Path = SWADESH_PATH
) -> list[WordForm]:
    """Load the reviewed Greek word table.

    :param csv_path: Reviewed word CSV with columns ``english,greek,gender``.
    :param swadesh_path: File with one English label per line.
    :raises ValueError: The row count or a form is wrong.
    :returns: One :class:`WordForm` per concept.
    """
    rows = read_csv_rows(csv_path)
    labels = read_swadesh_labels(swadesh_path)
    if len(rows) != len(labels):
        raise ValueError(f"{csv_path}: {len(rows)} rows but {len(labels)} labels")
    table: list[WordForm] = []
    global VERB_FORMS
    for index, row in enumerate(rows, start=1):
        form = normalized(row.get("greek", ""))
        if not form:
            raise ValueError(f"row {index}: empty Greek form")
        variant = "shared" if index == SHARED_INDEX else "primary"
        table.append(
            WordForm(
                swadesh_index=index,
                english=row.get("english", ""),
                greek=form,
                gender=row.get("gender", ""),
                variant=variant,
                form_kind=form_kind(form),
                zipf=zipf_frequency(form, "el"),
            )
        )
    VERB_FORMS = {form.greek for form in table if form.english.startswith("to ")}
    return table


def module_terms_url() -> str:
    """Build the API URL of the Greek Swadesh module.

    :returns: The MediaWiki API URL.
    """
    query = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": "Module:Swadesh/data/el",
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"https://en.wiktionary.org/w/api.php?{query}"


def fetch_module_terms(
    cache_path: Path = MODULE_CACHE_PATH, *, refresh: bool = False
) -> Path:
    """Download the Greek Swadesh module wikitext into the cache.

    :param cache_path: Cache file of the wikitext.
    :param refresh: Ignore an existing cache file.
    :returns: The cache path.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return cache_path
    request = urllib.request.Request(
        module_terms_url(), headers={"User-Agent": "anki-tools"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        text = data["parse"]["wikitext"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"cannot read {module_terms_url()}: {error}") from error
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return cache_path


MODULE_ROW = re.compile(r"^m\[(\d+)\]\s*=\s*\{(.*)\}\s*$")
MODULE_TERM = re.compile(r'term\s*=\s*"([^"]+)"')


def parse_module_terms(text: str) -> dict[int, tuple[str, ...]]:
    """Parse the Swadesh module wikitext into terms per index.

    :param text: Module wikitext.
    :returns: Mapping from index to its terms.
    """
    terms: dict[int, tuple[str, ...]] = {}
    for line in text.splitlines():
        match = MODULE_ROW.match(line.strip())
        if not match:
            continue
        terms[int(match.group(1))] = tuple(MODULE_TERM.findall(match.group(2)))
    return terms


@lru_cache(maxsize=1)
def module_terms(cache_path: Path = MODULE_CACHE_PATH) -> dict[int, tuple[str, ...]]:
    """Read the cached module terms, with a download on the first run.

    :param cache_path: Cache file of the wikitext.
    :returns: Mapping from index to its terms.
    """
    path = cache_path if Path(cache_path).exists() else fetch_module_terms(cache_path)
    return parse_module_terms(Path(path).read_text(encoding="utf-8"))


def read_word_review(review_path: Path = WORD_REVIEW_PATH) -> list[dict[str, str]]:
    """Read the Greek word review CSV.

    :param review_path: Review CSV file.
    :returns: One mapping per row, empty without a file.
    """
    if not Path(review_path).exists():
        return []
    return read_csv_rows(review_path)


def _module_forms(terms: dict[int, tuple[str, ...]], index: int) -> set[str]:
    return {
        normalized(strip_note(term)) for term in terms.get(index, ())
    } | {clean_form(term)[0] for term in terms.get(index, ())}


def check_word_table(
    table: list[WordForm],
    csv_path: Path = WORD_CSV_PATH,
    swadesh_path: Path = SWADESH_PATH,
    review_path: Path = WORD_REVIEW_PATH,
) -> list[str]:
    """Check the Greek word table against the reviewed files.

    :param table: Word table from :func:`load_word_table`.
    :param csv_path: Reviewed word CSV.
    :param swadesh_path: File with one English label per line.
    :param review_path: Review CSV that must exist and have no open row.
    :returns: Failure strings. An empty list means every check passes.
    """
    failures: list[str] = []
    labels = read_swadesh_labels(swadesh_path)
    indices = sorted({form.swadesh_index for form in table})
    if len(indices) != SWADESH_ROWS:
        failures.append(f"row count: expected {SWADESH_ROWS}, found {len(indices)}")
    expected = set(range(1, SWADESH_ROWS + 1))
    missing = sorted(expected - set(indices))
    if missing:
        failures.append(f"swadesh index: missing {missing}")
    forms = [form.greek for form in table]
    if len(set(forms)) != UNIQUE_FORMS:
        failures.append(
            f"unique forms: expected {UNIQUE_FORMS}, found {len(set(forms))}"
        )
    for form in table:
        if not 1 <= form.swadesh_index <= len(labels):
            failures.append(f"index out of range: {form.swadesh_index}")
            continue
        label = labels[form.swadesh_index - 1]
        if form.english != label:
            failures.append(
                f"label {form.swadesh_index}: expected {label!r}, "
                f"found {form.english!r}"
            )
        problem = script_problem(form.greek)
        if problem:
            failures.append(f"form {form.greek}: {problem}")
        if form.greek != form.greek.strip():
            failures.append(f"form {form.greek}: surrounding space")
    shared = [form for form in table if form.variant == "shared"]
    if len(shared) != 1 or shared[0].greek != SHARED_FORM:
        failures.append("shared form: expected one πόδι row at index 81")
    review = read_word_review(review_path)
    if not Path(review_path).exists():
        failures.append(f"review file missing: {review_path}")
    open_rows = [
        row.get("index", "")
        for row in review
        if row.get("status", "").lower() == OPEN_STATUS
    ]
    if open_rows:
        failures.append(f"review file has {len(open_rows)} open row(s): {open_rows}")
    reviewed = {row.get("index", "") for row in review if row.get("chosen", "")}
    terms = module_terms()
    for form in table:
        if form.greek in _module_forms(terms, form.swadesh_index):
            continue
        if str(form.swadesh_index) in reviewed:
            continue
        failures.append(
            f"form {form.greek}: not in the module row set of index "
            f"{form.swadesh_index} and no review row"
        )
    return failures


def word_table_report(table: list[WordForm]) -> dict[str, Any]:
    """Summarize the Greek word table.

    :param table: Word table from :func:`load_word_table`.
    :returns: Counts, the low Zipf forms, and the uncovered forms.
    """
    forms = [form.greek for form in table]
    low = sorted(
        (
            (form.greek, round(form.zipf, 2))
            for form in table
            if form.zipf < MIN_ZIPF
        ),
        key=lambda item: item[1],
    )
    return {
        "words": len({form.swadesh_index for form in table}),
        "forms": len(table),
        "unique_forms": len(set(forms)),
        "gender_forms": sum(1 for form in table if form.gender),
        "shared_forms": sum(1 for form in table if form.variant == "shared"),
        "zipf_below_3": len(low),
        "low_zipf_forms": low,
        "uncovered": [],
    }


def make_candidate(
    facts: Any, target: str, tier: str, covered: frozenset[str]
) -> Candidate:
    """Build one Greek candidate card from the measured facts.

    :param facts: Sentence facts of the shared pool stage.
    :param target: Assigned word-table form.
    :param tier: ``core`` or ``scaffold``.
    :param covered: Forms that the sentence covers.
    :returns: The candidate card.
    """
    from dataclasses import replace

    from anki_tools.sentence_pack.select import rank_key_parts, serialize_rank_key

    candidate = Candidate(
        target=target,
        sentence=facts.sentence,
        text_original=facts.text_original,
        english=facts.english,
        tatoeba_id=facts.tatoeba_id,
        tatoeba_author=facts.tatoeba_author,
        english_id=facts.english_id,
        audio_id=facts.audio_id,
        audio_source="tts",
        audio_license="",
        unknown_words=facts.unknown_words,
        word_len=facts.unit_len,
        zipf_mean=facts.zipf_mean,
        rank_key="",
        tier=tier,
        form_kind=form_kind(target),
        audio_missing=True,
        covered=covered,
    )
    return replace(
        candidate, rank_key=serialize_rank_key(rank_key_parts(candidate, GREEK))
    )


def sentence_row(card: Candidate) -> dict[str, Any]:
    """Build one sentence CSV row of a card.

    :param card: Selected card.
    :returns: Mapping from CSV column to value.
    """
    return {
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
        "word_len": card.word_len,
        "zipf_mean": f"{card.zipf_mean:.2f}",
        "rank_key": card.rank_key,
        "tier": card.tier,
        "form_kind": card.form_kind,
        "audio_missing": "true" if card.audio_missing else "false",
    }


def write_sentence_csv(
    cards: list[Candidate], path: Path = SENTENCES_PATH
) -> Path:
    """Write the card rows of the Greek sentence CSV.

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
    """Write the review rows for the weak Greek words.

    :param pool: Candidate list from the pool stage.
    :param word_table: Word table from :func:`load_word_table`.
    :param path: Output CSV file.
    :param per_word: Card limit per form, used for the thin test.
    :param top_n: Candidate rows per word.
    :returns: The written path.
    """
    from anki_tools.sentence_pack.select import review_candidates, review_status

    pool_counts, core_counts, by_form = review_candidates(pool, GREEK)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_REVIEW_FIELDS)
        writer.writeheader()
        for form in word_table:
            status = review_status(
                form, pool_counts[form.greek], core_counts[form.greek], per_word, MIN_ZIPF
            )
            if status is None:
                continue
            base = {
                "target": form.greek,
                "swadesh_index": form.swadesh_index,
                "english": form.english,
                "variant": form.variant,
                "status": status,
                "zipf": f"{form.zipf:.2f}",
                "candidate_count": pool_counts[form.greek],
                "scaffold_candidates": pool_counts[form.greek]
                - core_counts[form.greek],
            }
            candidates = by_form.get(form.greek, [])[:top_n]
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
                        "english_text": candidate.english,
                        "tatoeba_id": candidate.tatoeba_id,
                        "unknown_words": candidate.unknown_words,
                        "word_len": candidate.word_len,
                        "rank_key": candidate.rank_key,
                        "tier": candidate.tier,
                    }
                )
    return path


def known_words(word_table: Sequence[WordForm]) -> set[str]:
    """Build the known word set of the unknown-token filter.

    :param word_table: Reviewed word table.
    :returns: Word forms plus the top 1000 ``wordfreq`` el words.
    """
    from wordfreq import top_n_list

    known = {form.greek for form in word_table}
    known.update(top_n_list("el", 1000))
    return known


def greek_attribution(
    rows: Sequence[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
    path: Path = ATTRIBUTION_PATH,
) -> None:
    """Add the Greek section to the shared ``ATTRIBUTION.md`` file.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author, unused for Greek.
    :param path: Output Markdown file.
    """
    lines = attribution_lines(GREEK, rows, voice)
    lines.extend(
        [
            "",
            f"Greek word data: <{SWADESH_URL}>, CC BY-SA 4.0.",
            f"Tagger and lemmatizer: spaCy {TOKENIZER}, MIT license.",
        ]
    )
    write_attribution_section(
        path,
        "Greek",
        lines,
        header=["# Attribution", ""],
    )


def greek_extra_manifest(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Add the Greek manifest keys.

    :param rows: Sentence CSV rows.
    :returns: Extra manifest keys.
    """
    return {"native_audio_available": False}


def greek_excluded_words(
    table: Sequence[WordForm], targets: Collection[str]
) -> list[str]:
    """List the Greek forms that get no card.

    :param table: Reviewed word table.
    :param targets: Card targets of the pack.
    :returns: Shared and uncovered forms.
    """
    wanted = set(targets)
    excluded: list[str] = []
    for form in table:
        if form.variant == "shared" or form.greek not in wanted:
            if form.greek not in excluded:
                excluded.append(form.greek)
    return excluded


GREEK = register(
    LanguageProfile(
        name="greek",
        lang_tag="el",
        tatoeba_lang="ell",
        english_lang="eng",
        word_csv=WORD_CSV_PATH,
        sentence_csv=SENTENCES_PATH,
        sentence_review_csv=SENTENCE_REVIEW_PATH,
        pool_report=POOL_REPORT_PATH,
        audio_dir=AUDIO_DIR,
        word_review_csv=WORD_REVIEW_PATH,
        deck_name=DECK_NAME,
        model_name=MODEL_NAME,
        output_apkg=Path("apkg/swadesh_greek_sentences.apkg"),
        manifest_path=Path("apkg/swadesh_greek_sentences.manifest.json"),
        fields=FIELDS,
        sort_field_index=SORT_FIELD_INDEX,
        front_template=FRONT_TEMPLATE,
        back_template=BACK_TEMPLATE,
        deck_description=DECK_DESCRIPTION,
        licenses=(
            ("Greek Swadesh word data", "CC BY-SA 4.0"),
            ("Tatoeba sentences", "CC BY 2.0 FR"),
            (f"spaCy {TOKENIZER}", "MIT"),
        ),
        extra_css=EXTRA_CSS,
        min_units=4,
        max_units=10,
        unit_name="word",
        fallback_min_units=3,
        preferred_min_units=5,
        preferred_max_units=8,
        final_marks=FINAL_MARKS,
        internal_marks=INTERNAL_MARKS,
        quote_marks=QUOTE_MARKS,
        comma_marks=",·",
        orphan_conjunctions=ORPHAN_CONJUNCTIONS,
        wordfreq_lang="el",
        tokenizer=TOKENIZER,
        native_audio=False,
        prefer_native_audio=False,
        voice="gtts:el",
        tts_lang="el",
        gtts_voice="gtts:el",
        swadesh_url=SWADESH_URL,
        swadesh_license="CC BY-SA 4.0",
        tatoeba_license="CC BY 2.0 FR",
        core_concepts=SWADESH_ROWS,
        zipf_min=MIN_ZIPF,
        similarity_limit=0.67,
        coverage_min=0.76,
        prepare_text=normalized,
        unit_count=unit_count,
        script_ok=script_ok,
        is_repeated=lambda sentence: False,
        has_predicate=has_predicate,
        has_orphan=has_orphan,
        hits=hits,
        match_count=match_count,
        candidate_length=lambda card: card.word_len,
        count_unknown=count_unknown,
        zipf_words=token_words,
        similarity=word_similarity_el,
        dedupe_key=greek_text_dedup_key,
        similarity_keys=lambda sentence: set(surface_key(w) for w in token_words(sentence)),
        form_text=lambda word: word.greek,
        form_positions=form_positions,
        gloss=glossary,
        card_fields=lambda row, word: {},
        card_order=lambda word, row: (
            word.swadesh_index,
            0 if word.variant == "primary" else 1,
            int(row["tatoeba_id"]),
        ),
        excluded_words=greek_excluded_words,
        extra_manifest=greek_extra_manifest,
        attribution=greek_attribution,
        finalize_candidate=lambda card, target: card,
    )
)


def gate_word_table(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G1: the reviewed word table passes every S1 check.

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
        return False, sample(failures)
    return True, f"{concepts} concepts, {len(table)} forms"


def gate_word_boundary(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G9: no card matches the target inside a longer word.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad: list[str] = []
    for row in rows:
        target = row["target"]
        if token_matches(row["sentence"], target) != 1:
            bad.append(f"{target} in {row['sentence']}")
    if bad:
        return False, f"{len(bad)} card(s) match inside a longer word: {sample(bad)}"
    return True, f"{len(rows)} cards match a whole token"


def gate_attribution(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G12: the Greek section names every sentence author.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    path = inputs.attribution_path
    if not path.is_file():
        return False, f"attribution file missing: {path}"
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if "## Greek pack" not in text:
        problems.append("no '## Greek pack' section")
    authors = {
        row.get("tatoeba_author", "").strip() or "unknown" for row in inputs.rows()
    }
    missing = sorted(author for author in authors if author not in text)
    if missing:
        problems.append(f"{len(missing)} author(s) missing: {sample(missing)}")
    if problems:
        return False, sample(problems)
    return True, f"{len(authors)} author(s) named in the Greek section"


def gate_script(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G17: every card field is NFC, monotonic, and keeps ς at the end.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad: list[str] = []
    for row in rows:
        for field in ("sentence", "text_original", "target"):
            value = row.get(field, "")
            if not value:
                continue
            problem = script_problem(value)
            if problem:
                bad.append(f"{field}={value}: {problem}")
    if bad:
        return False, f"{len(bad)} card field(s) fail the script rules: {sample(bad)}"
    return True, f"{len(rows)} cards hold monotonic NFC text"


def gate_no_native(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G18: no card uses the native audio source.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    native = [
        row["sentence"] for row in inputs.rows() if row.get("audio_source") == "native"
    ]
    if native:
        return False, f"{len(native)} card(s) use native audio: {sample(native)}"
    return True, "no card uses native audio"


def gate_targets(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Greek gate G19: every target is a table form and is not shared.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    forms = {form.greek for form in table}
    shared = {form.greek for form in table if form.variant == "shared"}
    rows = inputs.rows()
    problems: list[str] = []
    for target in sorted({row["target"] for row in rows}):
        if target not in forms:
            problems.append(f"{target} is not in the word table")
        elif target in shared:
            problems.append(f"{target} is a shared form")
    if problems:
        return False, f"{len(problems)} target(s) invalid: {sample(problems)}"
    return True, f"{len(rows)} card targets are table forms with a card"


def greek_gates() -> tuple[tuple[str, Any], ...]:
    """Give the ordered gate list G1 to G19 of the Greek pack.

    :returns: Pairs of gate name and gate function.
    """
    from anki_tools.sentence_pack.audit import (
        gate_apkg,
        gate_coverage,
        gate_manifest_counts,
        gate_media_count,
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
        ("G9", gate_word_boundary),
        ("G10", gate_coverage),
        ("G11", gate_manifest_counts),
        ("G12", gate_attribution),
        ("G13", gate_apkg),
        ("G14", gate_proper_sentences),
        ("G15", gate_review_file),
        ("G16", gate_word_audio),
        ("G17", gate_script),
        ("G18", gate_no_native),
        ("G19", gate_target_in_table),
    )


def run_greek_audit(
    *,
    apkg_path: Path | None = None,
    sentences_csv: Path = SENTENCES_PATH,
    word_csv: Path = WORD_CSV_PATH,
    review_csv: Path = WORD_REVIEW_PATH,
    audio_dir: Path = AUDIO_DIR,
    manifest_path: Path | None = None,
    attribution_path: Path = ATTRIBUTION_PATH,
) -> list[GateRow]:
    """Run the Greek quality gates G1 to G19.

    :param apkg_path: APKG file of the pack.
    :param sentences_csv: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param review_csv: Word review CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param manifest_path: Manifest JSON file.
    :param attribution_path: Attribution Markdown file.
    :returns: One ``(gate, passed, message)`` row per gate, G1 first.
    """
    from anki_tools.sentence_pack.audit import run_gates

    inputs = AuditInputs(
        GREEK,
        apkg_path=apkg_path or GREEK.output_apkg,
        sentences_csv=sentences_csv,
        word_csv=word_csv,
        review_csv=review_csv,
        audio_dir=audio_dir,
        manifest_path=manifest_path or GREEK.manifest_path,
        attribution_path=attribution_path,
        load_table=lambda csv_path: load_word_table(csv_path),
    )
    return run_gates(greek_gates(), inputs, GREEK)


__all__ = [
    "GREEK",
    "Candidate",
    "WordForm",
    "check_word_table",
    "clean_form",
    "hits",
    "load_word_table",
    "run_greek_audit",
    "script_problem",
    "strip_article",
    "strip_note",
]
