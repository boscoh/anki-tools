"""French language pack of the shared sentence pipeline.

The module holds the French word table reader, the Tatoeba module reader, the
elision-aware form-key matcher, the proper-sentence rules, and the French
language profile. French inflects, so a form matches by token surface or by
token lemma. The lemmatizer gives a wrong lemma for a bare noun (``pluie``
gives ``plui``), so the key set also holds the head-word lemma of a carrier
phrase.
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

from wordfreq import top_n_list, zipf_frequency

from anki_tools.languages.french.rank import word_similarity_fr
from anki_tools.ranking.latin import latin_text_dedup_key
from anki_tools.sentence_pack.audit import AuditInputs, GateRow, sample
from anki_tools.sentence_pack.pack import (
    ATTRIBUTION_PATH,
    attribution_lines,
    write_attribution_section,
)
from anki_tools.sentence_pack.profile import LanguageProfile, make_sense_filter, register
from anki_tools.sentence_pack.tokens import Token, has_finite_verb, tokenize

WORD_CSV_PATH = Path("vocab/english-french.csv")
SWADESH_PATH = Path("vocab/swadesh.txt")
WORD_REVIEW_PATH = Path("vocab/french_word_review.csv")
SENTENCES_PATH = Path("vocab/french_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/french_sentences_review.csv")
POOL_REPORT_PATH = Path("data/french_pool_report.json")
AUDIO_DIR = Path("audio/french_sentences")
MODULE_CACHE_PATH = Path("data/swadesh/fr_module.txt")

MODEL_NAME = "French Swadesh Sentence"
DECK_NAME = "Swadesh French Sentences"
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
    "Audio: synthesized speech (gTTS), voice gtts:fr."
)
SWADESH_URL = "https://en.wiktionary.org/wiki/Module:Swadesh/data/fr"
TOKENIZER = "fr_core_news_sm"

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
UNIQUE_FORMS = 207
MIN_ZIPF = 3.0
OPEN_STATUS = "open"
FINAL_MARKS = ".?!"
INTERNAL_MARKS = ",;:"
QUOTE_MARKS = '«»"“”—–'
ORPHAN_CONJUNCTIONS = (
    "et",
    "mais",
    "ou",
    "car",
    "donc",
    "puis",
    "parce que",
    "puisque",
    "bien que",
)
PREDICATE_MARKERS = frozenset({"est", "sont", "était", "a", "ont"})
DISCONTINUOUS_FORMS = frozenset({"ne pas"})
CARRIERS = ("{form}", "le {form}", "je {form}", "il {form}")
ELIDED_CLITICS = (
    "l'",
    "d'",
    "j'",
    "n'",
    "s'",
    "c'",
    "qu'",
    "m'",
    "t'",
    "jusqu'",
    "lorsqu'",
    "puisqu'",
)
FRENCH_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ")

NOTE_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
LINK_PATTERN = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
MODULE_ROW_START = re.compile(r"m\[(\d+)\]\s*=\s*")
MODULE_TERM = re.compile(r'term\s*=\s*"([^"]+)"')
SPACE_CHARS = ("\u202f", "\u00a0")
NOUN_INDICES = frozenset(
    {
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 177, 178, 179, 207,
    }
)  # fmt: skip

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
    """One reviewed French form of a Swadesh concept.

    :param swadesh_index: CSV row order, 1-based.
    :param english: Swadesh English label of the row.
    :param french: French form, the unique key of the row.
    :param gender: ``m``, ``f``, ``n``, or empty for a word that is not a noun.
    :param variant: ``primary`` or ``alternate``.
    :param form_kind: ``single``, ``pair``, or ``discontinuous``.
    :param zipf: ``wordfreq`` Zipf value of the form for ``fr``.
    :param candidate_count: Sentence count after the S2 filters.
    :param tier: ``core`` or ``scaffold``.
    """

    swadesh_index: int
    english: str
    french: str
    gender: str
    variant: str
    form_kind: str
    zipf: float
    candidate_count: int = 0
    tier: str = "core"


@dataclass(frozen=True)
class Candidate:
    """One candidate sentence for the French sentence pack.

    :param target: Assigned Swadesh form, the lowest covered form by index.
    :param sentence: Prepared French sentence with normal spaces.
    :param text_original: Sentence text before the space normalization.
    :param english: Paired English translation.
    :param tatoeba_id: Tatoeba sentence id.
    :param tatoeba_author: Tatoeba author name, empty when unknown.
    :param english_id: Tatoeba English sentence id.
    :param audio_id: Native Tatoeba audio id, 0 when the pool holds none.
    :param audio_source: ``tts`` or ``native``.
    :param audio_license: Recording license of a native file, else empty.
    :param unknown_words: Token count outside the known set.
    :param word_len: Word count of the sentence.
    :param zipf_mean: Mean Zipf value of the sentence tokens.
    :param rank_key: Serialized ranking key.
    :param tier: ``core`` or ``scaffold``.
    :param form_kind: Kind of the assigned target form.
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


def prepare_text(text: str) -> str:
    """Normalize the spaces of French text and keep its case.

    NFC is applied, the narrow no-break space and the no-break space become a
    normal space, and runs of spaces collapse. The case and the accents stay.

    :param text: Raw French text.
    :returns: NFC text with normal spaces.
    """
    text = unicodedata.normalize("NFC", text)
    for char in SPACE_CHARS:
        text = text.replace(char, " ")
    return " ".join(text.split())


def surface_key(text: str) -> str:
    """Build the comparison key of one French word.

    :param text: Word or sentence text.
    :returns: NFC, space-normalized, casefolded text.
    """
    return prepare_text(text).casefold()


@lru_cache(maxsize=50_000)
def french_tokens(sentence: str) -> tuple[Token, ...]:
    """Tokenize a French sentence once.

    :param sentence: Prepared French sentence.
    :returns: Tokens of the sentence.
    """
    return tuple(tokenize(TOKENIZER, sentence))


def word_tokens(sentence: str) -> tuple[Token, ...]:
    """Give the word tokens of a French sentence.

    :param sentence: Prepared French sentence.
    :returns: Tokens with the punctuation tokens removed.
    """
    return tuple(token for token in french_tokens(sentence) if not token.punctuation)


def token_words(sentence: str) -> list[str]:
    """Give the word tokens of a French sentence.

    :param sentence: Prepared French sentence.
    :returns: Token texts without punctuation.
    """
    return [token.text for token in word_tokens(sentence)]


def unit_count(sentence: str) -> int:
    """Count the words of a French sentence.

    An elided article splits on the apostrophe, so ``l'eau`` counts as two
    words.

    :param sentence: Prepared French sentence.
    :returns: Word count without punctuation.
    """
    return len(word_tokens(sentence))


def script_ok(sentence: str) -> bool:
    """Check filter F3: Latin letters and no digit.

    :param sentence: Prepared French sentence.
    :returns: True when every letter is a French Latin letter and no digit exists.
    """
    for char in sentence:
        if char.isdigit():
            return False
        if char.isalpha() and char.casefold() not in FRENCH_LETTERS:
            return False
    return True


def is_repeated_sentence(sentence: str) -> bool:
    """Check filter F4: the word sequence is a repetition of a shorter one.

    :param sentence: Prepared French sentence.
    :returns: True for text such as ``Oui oui oui.``.
    """
    tokens = [surface_key(token.text) for token in word_tokens(sentence)]
    count = len(tokens)
    if count < 2:
        return False
    for period in range(1, count // 2 + 1):
        if tokens[: count - period] == tokens[period:]:
            return True
    return False


def has_orphan(sentence: str) -> bool:
    """Check filter F10: a sentence-initial conjunction with no comma.

    :param sentence: Prepared French sentence.
    :returns: True when the leading conjunction has no main clause.
    """
    if "," in sentence or ";" in sentence:
        return False
    tokens = [surface_key(token.text) for token in word_tokens(sentence)]
    if not tokens:
        return False
    if tokens[0] in ORPHAN_CONJUNCTIONS:
        return True
    return " ".join(tokens[:2]) in ORPHAN_CONJUNCTIONS


@lru_cache(maxsize=8192)
def carrier_keys(form: str) -> frozenset[str]:
    """Read the target keys of one form from its carrier phrases.

    :param form: Word-table form or one part of it.
    :returns: Normalized surfaces of the form and of the carrier head words.
    """
    keys = {surface_key(form)}
    for carrier in CARRIERS:
        tokens = [
            token
            for token in tokenize(TOKENIZER, carrier.format(form=form))
            if not token.punctuation
        ]
        if not tokens:
            continue
        head = tokens[-1]
        keys.add(surface_key(head.text))
        if head.lemma:
            keys.add(surface_key(head.lemma))
    return frozenset(keys)


@lru_cache(maxsize=8192)
def form_parts(form: str) -> tuple[str, ...]:
    """Split a form into its tokens.

    The elided form ``s'étendre`` gives two tokens, ``s'`` and ``étendre``.

    :param form: Reviewed form.
    :returns: Token texts without punctuation.
    """
    return tuple(token.text for token in word_tokens(prepare_text(form)))


def form_kind(form: str) -> str:
    """Give the kind of a word-table form.

    :param form: Reviewed form.
    :returns: ``single``, ``pair``, or ``discontinuous``.
    """
    parts = form_parts(form)
    if len(parts) <= 1:
        return "single"
    if surface_key(form) in DISCONTINUOUS_FORMS:
        return "discontinuous"
    return "pair"


def form_keys(form: str) -> frozenset[str]:
    """Build the target key set of one form.

    :param form: Reviewed form.
    :returns: Surface keys that a token may match.
    """
    parts = form_parts(form)
    if len(parts) <= 1:
        return carrier_keys(form)
    keys: set[str] = set()
    for part in parts:
        keys.update(carrier_keys(part))
    return frozenset(keys)


def _token_keys(token: Token) -> set[str]:
    keys = {surface_key(token.text)}
    if token.lemma:
        keys.add(surface_key(token.lemma))
    return keys


def _token_matches(token: Token, keys: frozenset[str]) -> bool:
    return bool(_token_keys(token) & keys)


def _phrase_count(tokens: Sequence[Token], parts: Sequence[str]) -> int:
    wanted = [carrier_keys(part) for part in parts]
    count = 0
    size = len(wanted)
    for start in range(len(tokens) - size + 1):
        if all(
            _token_matches(tokens[start + offset], wanted[offset])
            for offset in range(size)
        ):
            count += 1
    return count


def _in_order(tokens: Sequence[Token], parts: Sequence[str]) -> bool:
    position = 0
    for part in parts:
        keys = carrier_keys(part)
        while position < len(tokens) and not _token_matches(tokens[position], keys):
            position += 1
        if position == len(tokens):
            return False
        position += 1
    return True


@lru_cache(maxsize=8)
def key_index(forms: frozenset[str]) -> dict[str, frozenset[str]]:
    """Map each target key to the single-token forms that it matches.

    :param forms: Word-table forms.
    :returns: Mapping from a surface key to the forms with that key.
    """
    index: dict[str, set[str]] = {}
    for form in forms:
        if len(form_parts(form)) > 1:
            continue
        for key in form_keys(form):
            index.setdefault(key, set()).add(form)
    return {key: frozenset(value) for key, value in index.items()}


def token_matches(sentence: str, form: str) -> int:
    """Count the matches of one form in a sentence.

    A single-token form matches by surface key or by lemma key. A pair form
    matches only when both parts are adjacent tokens. A discontinuous form
    matches when both parts appear in that order.

    :param sentence: Prepared French sentence.
    :param form: Reviewed form.
    :returns: Match count.
    """
    tokens = word_tokens(sentence)
    parts = form_parts(form)
    if len(parts) > 1:
        if form_kind(form) == "discontinuous":
            return 1 if _in_order(tokens, parts) else 0
        return _phrase_count(tokens, parts)
    keys = carrier_keys(form)
    return sum(1 for token in tokens if _token_matches(token, keys))


SENSE_CHECKS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "queue": (
        frozenset({"tail", "tails"}),
        frozenset({"queue", "queues", "queuing", "queued", "line", "lines", "wait", "waiting"}),
    ),
    "sentir": (
        frozenset({"smell", "smells", "smelled", "smelt", "smelling", "scent", "scents"}),
        frozenset({"feel", "feels", "felt", "feeling"}),
    ),
    "voler": (
        frozenset({"fly", "flies", "flew", "flying", "flown", "flight"}),
        frozenset({"steal", "steals", "stole", "stolen", "stealing", "rob", "robbed", "theft"}),
    ),
    "tirer": (
        frozenset({"pull", "pulls", "pulled", "pulling", "drag", "drags", "dragged"}),
        frozenset({"shoot", "shoots", "shot", "shooting"}),
    ),
    "langue": (
        frozenset({"tongue", "tongues"}),
        frozenset({"language", "languages"}),
    ),
    "plume": (
        frozenset({"feather", "feathers", "plumage"}),
        frozenset({"pen", "pens", "quill", "quills"}),
    ),
    "foie": (
        frozenset({"liver", "livers"}),
        frozenset({"chicken", "scared", "afraid", "fear", "frightened", "feet"}),
    ),
    "gonfler": (
        frozenset({"swell", "swells", "swelled", "swollen", "swelling", "inflate", "inflated"}),
        frozenset({"bold", "nerve", "cheeky", "cheek", "gall"}),
    ),
    "rond": (
        frozenset({"round", "rounded", "circle", "circular"}),
        frozenset({"dime", "dimes", "penny", "pennies", "money", "cash", "coin", "coins"}),
    ),
    "droit": (
        frozenset({"straight"}),
        frozenset({"right", "rights", "allowed", "entitled", "permit", "permission"}),
    ),
}


sense_filter = make_sense_filter(SENSE_CHECKS)


def hits(sentence: str, forms: Collection[str]) -> dict[str, int]:
    """Count the forms that the sentence contains exactly one time.

    :param sentence: Prepared French sentence.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to 1, for forms with one match.
    """
    tokens = word_tokens(sentence)
    index = key_index(frozenset(forms))
    counts: dict[str, int] = {}
    for token in tokens:
        for key in _token_keys(token):
            for form in index.get(key, ()):
                counts[form] = counts.get(form, 0) + 1
    found = {form: count for form, count in counts.items() if count == 1}
    for form in forms:
        if form in found or len(form_parts(form)) <= 1:
            continue
        if token_matches(sentence, form) == 1:
            found[form] = 1
    return found


def match_count(sentence: str, form: str) -> int:
    """Count the matches of one form, for the audit gates.

    :param sentence: Prepared French sentence.
    :param form: Card target.
    :returns: Match count.
    """
    return token_matches(sentence, form)


def match_spans(sentence: str, form: str) -> list[tuple[int, int]]:
    """Find the character spans of the matched target tokens.

    :param sentence: Prepared French sentence.
    :param form: Card target.
    :returns: ``(start, end)`` offsets of the first match, empty when no token matches.
    """
    tokens = list(word_tokens(sentence))
    parts = form_parts(form)
    if len(parts) > 1:
        wanted = [carrier_keys(part) for part in parts]
        if form_kind(form) == "discontinuous":
            spans: list[tuple[int, int]] = []
            position = 0
            for keys in wanted:
                while position < len(tokens) and not _token_matches(
                    tokens[position], keys
                ):
                    position += 1
                if position == len(tokens):
                    return []
                token = tokens[position]
                spans.append((token.idx, token.idx + len(token.text)))
                position += 1
            return spans
        size = len(wanted)
        for start in range(len(tokens) - size + 1):
            if all(
                _token_matches(tokens[start + offset], wanted[offset])
                for offset in range(size)
            ):
                first = tokens[start]
                last = tokens[start + size - 1]
                return [(first.idx, last.idx + len(last.text))]
        return []
    keys = carrier_keys(form)
    for token in tokens:
        if _token_matches(token, keys):
            return [(token.idx, token.idx + len(token.text))]
    return []


def highlight(sentence: str, form: str) -> str:
    """Wrap the matched target tokens of a card sentence in a bold tag.

    :param sentence: Card sentence with normal spaces.
    :param form: Card target.
    :returns: Sentence with the matched tokens in ``<b>`` tags.
    """
    spans = match_spans(sentence, form)
    if not spans:
        return sentence
    parts: list[str] = []
    position = 0
    for start, end in spans:
        parts.append(sentence[position:start])
        parts.append(f"<b>{sentence[start:end]}</b>")
        position = end
    parts.append(sentence[position:])
    return "".join(parts)


def count_unknown(sentence: str, known: Collection[str]) -> int:
    """Count the tokens outside the known word set.

    A token is known when its surface key or its lemma key is in the set.

    :param sentence: Prepared French sentence.
    :param known: Known word set.
    :returns: Unknown token count.
    """
    keys = {surface_key(word) for word in known}
    return sum(
        1
        for token in word_tokens(sentence)
        if not (_token_keys(token) & keys)
    )


def has_predicate(sentence: str) -> bool:
    """Check filter F9: a finite verb, a marker, or a word-list verb.

    :param sentence: Prepared French sentence.
    :returns: True when the sentence has a finite verb.
    """
    tokens = french_tokens(sentence)
    if has_finite_verb(list(tokens)):
        return True
    surfaces = [surface_key(token.text) for token in tokens if not token.punctuation]
    if any(surface in PREDICATE_MARKERS for surface in surfaces):
        return True
    if any(
        surfaces[index] == "y" and surfaces[index + 1] == "a"
        for index in range(len(surfaces) - 1)
    ):
        return True
    infinitives = verb_forms()
    return any(
        surface_key(token.lemma) in infinitives
        for token in tokens
        if not token.punctuation
    )


def glossary(word: WordForm) -> str:
    """Build the card gloss of one French form.

    :param word: Reviewed word form.
    :returns: Gloss text with the gender of a noun or the infinitive of a verb.
    """
    label = word.english
    if label.startswith("to "):
        return f"{label} ({word.french})"
    if word.gender:
        return f"{label} ({word.gender})"
    return label


def form_positions(table: Sequence[WordForm]) -> dict[str, tuple[int, int]]:
    """Give each form its Swadesh index and its variant rank.

    :param table: Reviewed word table.
    :returns: Mapping from form to ``(swadesh_index, variant_rank)``.
    """
    return {
        form.french: (form.swadesh_index, 0 if form.variant == "primary" else 1)
        for form in table
    }


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
    """Load the reviewed French word table.

    :param csv_path: Reviewed word CSV, the S1 output.
    :param swadesh_path: File with one English label per line.
    :raises ValueError: The row count or a form is wrong.
    :returns: One :class:`WordForm` per concept.
    """
    rows = read_csv_rows(csv_path)
    labels = read_swadesh_labels(swadesh_path)
    if len(rows) != len(labels):
        raise ValueError(f"{csv_path}: {len(rows)} rows but {len(labels)} labels")
    table: list[WordForm] = []
    for position, row in enumerate(rows, start=1):
        index = int(row.get("swadesh_index") or position)
        form = prepare_text(row.get("french", ""))
        if not form:
            raise ValueError(f"row {position}: empty French form")
        table.append(
            WordForm(
                swadesh_index=index,
                english=row.get("english") or labels[index - 1],
                french=form,
                gender=row.get("gender", ""),
                variant=row.get("variant") or "primary",
                form_kind=row.get("form_kind") or form_kind(form),
                zipf=zipf_frequency(form, "fr"),
                candidate_count=int(row.get("candidate_count") or 0),
                tier=row.get("tier") or "core",
            )
        )
    return table


@lru_cache(maxsize=1)
def verb_forms() -> frozenset[str]:
    """Give the infinitive forms of the word-list verbs.

    :returns: Surface keys of the verb rows of the reviewed word table.
    """
    return frozenset(
        surface_key(form.french)
        for form in load_word_table()
        if form.english.startswith("to ")
    )


def module_terms_url() -> str:
    """Build the API URL of the French Swadesh module.

    :returns: The MediaWiki API URL.
    """
    query = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": "Module:Swadesh/data/fr",
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"https://en.wiktionary.org/w/api.php?{query}"


def fetch_module_terms(
    cache_path: Path = MODULE_CACHE_PATH, *, refresh: bool = False
) -> Path:
    """Download the French Swadesh module wikitext into the cache.

    :param cache_path: Cache file of the wikitext.
    :param refresh: Ignore an existing cache file.
    :raises RuntimeError: The download or the JSON parse fails.
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


def parse_module_terms(text: str) -> dict[int, tuple[str, ...]]:
    """Parse the Swadesh module wikitext into terms per index.

    A module row can span several lines, so the parser reads each row up to its
    closing brace.

    :param text: Module wikitext.
    :returns: Mapping from index to its raw terms.
    """
    terms: dict[int, tuple[str, ...]] = {}
    for match in MODULE_ROW_START.finditer(text):
        index = int(match.group(1))
        start = match.end()
        depth = 0
        end = 0
        for position in range(start, len(text)):
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if not end:
            continue
        terms[index] = tuple(MODULE_TERM.findall(text[start:end]))
    return terms


@lru_cache(maxsize=1)
def module_terms(cache_path: Path = MODULE_CACHE_PATH) -> dict[int, tuple[str, ...]]:
    """Read the cached module terms, with a download on the first run.

    :param cache_path: Cache file of the wikitext.
    :returns: Mapping from index to its raw terms.
    """
    path = cache_path if Path(cache_path).exists() else fetch_module_terms(cache_path)
    return parse_module_terms(Path(path).read_text(encoding="utf-8"))


def clean_term(term: str) -> str:
    """Clean one module term.

    Link markup is removed, the text after a pipe stays, and a parenthetical
    note is removed.

    :param term: Raw module term, for example ``[[ne]] [[pas]]``.
    :returns: The cleaned form.
    """
    text = LINK_PATTERN.sub(lambda match: match.group(1), term)
    text = NOTE_PATTERN.sub(" ", text)
    return prepare_text(text)


def read_word_review(review_path: Path = WORD_REVIEW_PATH) -> list[dict[str, str]]:
    """Read the French word review CSV.

    :param review_path: Review CSV file.
    :returns: One mapping per row, empty without a file.
    """
    if not Path(review_path).exists():
        return []
    return read_csv_rows(review_path)


def _module_forms(index: int) -> set[str]:
    return {clean_term(term) for term in module_terms().get(index, ())}


def check_word_table(
    table: list[WordForm],
    swadesh_path: Path = SWADESH_PATH,
    review_path: Path = WORD_REVIEW_PATH,
) -> list[str]:
    """Check the French word table against the reviewed files.

    :param table: Word table from :func:`load_word_table`.
    :param swadesh_path: File with one English label per line.
    :param review_path: Review CSV that must exist and have no open row.
    :returns: Failure strings. An empty list means every check passes.
    """
    failures: list[str] = []
    labels = read_swadesh_labels(swadesh_path)
    indices = sorted({form.swadesh_index for form in table})
    if len(indices) != SWADESH_ROWS:
        failures.append(f"row count: expected {SWADESH_ROWS}, found {len(indices)}")
    missing = sorted(set(range(1, SWADESH_ROWS + 1)) - set(indices))
    if missing:
        failures.append(f"swadesh index: missing {missing}")
    forms = [form.french for form in table]
    if len(set(forms)) != UNIQUE_FORMS:
        failures.append(
            f"unique forms: expected {UNIQUE_FORMS}, found {len(set(forms))}"
        )
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
        problem = form_problem(form.french)
        if problem:
            failures.append(f"form {form.french}: {problem}")
        if form.form_kind != form_kind(form.french):
            failures.append(
                f"form {form.french}: form_kind {form.form_kind!r} "
                f"but the form is {form_kind(form.french)!r}"
            )
        if form.french in _module_forms(form.swadesh_index):
            continue
        if str(form.swadesh_index) in reviewed:
            continue
        terms = "|".join(module_terms().get(form.swadesh_index, ()))
        failures.append(
            f"form {form.french}: not in the module row set of index "
            f"{form.swadesh_index} (module terms {terms}) and no review row"
        )
    return failures


def form_problem(form: str) -> str | None:
    """Check the character set of one reviewed form.

    :param form: Reviewed form.
    :returns: Failure text, or None when the form passes.
    """
    if not form:
        return "empty form"
    if prepare_text(form) != form:
        return "not space-normalized"
    for char in form:
        if char in " '’-":
            continue
        if char.isalpha() and char.casefold() not in FRENCH_LETTERS:
            return f"letter outside the French set: {char}"
        if not char.isalpha():
            return f"character outside the French set: {char}"
    return None


def word_table_report(table: list[WordForm]) -> dict[str, Any]:
    """Summarize the French word table.

    :param table: Word table from :func:`load_word_table`.
    :returns: Counts, the low Zipf forms, and the noun gender coverage.
    """
    low = sorted(
        (
            (form.french, round(form.zipf, 2))
            for form in table
            if form.zipf < MIN_ZIPF
        ),
        key=lambda item: item[1],
    )
    nouns = [form for form in table if form.swadesh_index in NOUN_INDICES]
    return {
        "words": len({form.swadesh_index for form in table}),
        "forms": len(table),
        "unique_forms": len({form.french for form in table}),
        "nouns": len(nouns),
        "nouns_with_gender": sum(1 for form in nouns if form.gender),
        "zipf_below_3": len(low),
        "low_zipf_forms": low,
    }


def make_candidate(
    facts: Any, target: str, tier: str, covered: frozenset[str]
) -> Candidate:
    """Build one French candidate card from the measured facts.

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
        candidate, rank_key=serialize_rank_key(rank_key_parts(candidate, FRENCH))
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


def write_sentence_csv(cards: list[Candidate], path: Path = SENTENCES_PATH) -> Path:
    """Write the card rows of the French sentence CSV.

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
    """Write the review rows for the weak French words.

    :param pool: Candidate list from the pool stage.
    :param word_table: Word table from :func:`load_word_table`.
    :param path: Output CSV file.
    :param per_word: Card limit per form, used for the thin test.
    :param top_n: Candidate rows per word.
    :returns: The written path.
    """
    from anki_tools.sentence_pack.select import review_candidates, review_status

    pool_counts, core_counts, by_form = review_candidates(pool, FRENCH)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_REVIEW_FIELDS)
        writer.writeheader()
        for form in word_table:
            status = review_status(
                form,
                pool_counts[form.french],
                core_counts[form.french],
                per_word,
                MIN_ZIPF,
            )
            if status is None:
                continue
            base = {
                "target": form.french,
                "swadesh_index": form.swadesh_index,
                "english": form.english,
                "variant": form.variant,
                "status": status,
                "zipf": f"{form.zipf:.2f}",
                "candidate_count": pool_counts[form.french],
                "scaffold_candidates": pool_counts[form.french]
                - core_counts[form.french],
            }
            candidates = by_form.get(form.french, [])[:top_n]
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
    :returns: Word forms, elided clitics, and the top 1000 ``wordfreq`` fr words.
    """
    known = {form.french for form in word_table}
    known.update(ELIDED_CLITICS)
    known.update(top_n_list("fr", 1000))
    return known


def french_attribution(
    rows: Sequence[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
    path: Path = ATTRIBUTION_PATH,
) -> None:
    """Add the French section to the shared ``ATTRIBUTION.md`` file.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author, unused for TTS.
    :param path: Output Markdown file.
    """
    lines = attribution_lines(FRENCH, rows, voice)
    lines.extend(
        [
            "",
            f"French word data: <{SWADESH_URL}>, CC BY-SA 4.0.",
            f"Tagger and lemmatizer: spaCy {TOKENIZER}, MIT license.",
            "French typography: narrow no-break spaces and no-break spaces are "
            "normalized to a normal space in the card text.",
        ]
    )
    if audio_authors:
        lines.extend(["", "Native Tatoeba recordings:", ""])
        lines.extend(
            f"- audio id {audio_id}: {author}"
            for audio_id, author in sorted(audio_authors.items())
        )
    write_attribution_section(
        path,
        "French",
        lines,
        header=["# Attribution", ""],
    )


def french_extra_manifest(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Add the French manifest keys.

    :param rows: Sentence CSV rows.
    :returns: Extra manifest keys, with the noun gender count of gate G20.
    """
    table = load_word_table()
    nouns = [form for form in table if form.swadesh_index in NOUN_INDICES]
    return {
        "native_audio_available": True,
        "noun_rows": len(nouns),
        "noun_gender_missing": sum(1 for form in nouns if not form.gender),
    }


def french_excluded_words(
    table: Sequence[WordForm], targets: Collection[str]
) -> list[str]:
    """List the French forms that get no card.

    :param table: Reviewed word table.
    :param targets: Card targets of the pack.
    :returns: Forms without a card, in Swadesh order.
    """
    wanted = set(targets)
    excluded: list[str] = []
    for form in table:
        if form.french not in wanted and form.french not in excluded:
            excluded.append(form.french)
    return excluded


FRENCH = register(
    LanguageProfile(
        name="french",
        lang_tag="fr",
        tatoeba_lang="fra",
        english_lang="eng",
        word_csv=WORD_CSV_PATH,
        sentence_csv=SENTENCES_PATH,
        sentence_review_csv=SENTENCE_REVIEW_PATH,
        pool_report=POOL_REPORT_PATH,
        audio_dir=AUDIO_DIR,
        word_review_csv=WORD_REVIEW_PATH,
        deck_name=DECK_NAME,
        model_name=MODEL_NAME,
        output_apkg=Path("apkg/swadesh_french_sentences.apkg"),
        manifest_path=Path("apkg/swadesh_french_sentences.manifest.json"),
        fields=FIELDS,
        sort_field_index=SORT_FIELD_INDEX,
        front_template=FRONT_TEMPLATE,
        back_template=BACK_TEMPLATE,
        deck_description=DECK_DESCRIPTION,
        licenses=(
            ("French Swadesh word data", "CC BY-SA 4.0"),
            ("Tatoeba sentences", "CC BY 2.0 FR"),
            ("Tatoeba audio", "per recording"),
            (f"spaCy {TOKENIZER}", "MIT"),
            ("gTTS", "MIT"),
        ),
        extra_css=EXTRA_CSS,
        min_units=4,
        max_units=10,
        unit_name="word",
        fallback_min_units=3,
        preferred_min_units=5,
        preferred_max_units=9,
        final_marks=FINAL_MARKS,
        internal_marks=INTERNAL_MARKS,
        quote_marks=QUOTE_MARKS,
        comma_marks=",",
        orphan_conjunctions=ORPHAN_CONJUNCTIONS,
        wordfreq_lang="fr",
        tokenizer=TOKENIZER,
        native_audio=True,
        prefer_native_audio=True,
        voice="gtts:fr",
        tts_lang="fr",
        gtts_voice="gtts:fr",
        swadesh_url=SWADESH_URL,
        swadesh_license="CC BY-SA 4.0",
        tatoeba_license="CC BY 2.0 FR",
        core_concepts=SWADESH_ROWS,
        zipf_min=MIN_ZIPF,
        similarity_limit=0.67,
        coverage_min=0.95,
        prepare_text=prepare_text,
        unit_count=unit_count,
        script_ok=script_ok,
        is_repeated=is_repeated_sentence,
        has_predicate=has_predicate,
        has_orphan=has_orphan,
        hits=hits,
        match_count=match_count,
        candidate_length=lambda card: card.word_len,
        count_unknown=count_unknown,
        zipf_words=token_words,
        similarity=word_similarity_fr,
        dedupe_key=latin_text_dedup_key,
        similarity_keys=lambda sentence: {
            surface_key(word) for word in token_words(sentence)
        },
        form_text=lambda word: word.french,
        form_positions=form_positions,
        gloss=glossary,
        card_fields=lambda row, word: {},
        card_order=lambda word, row: (
            word.swadesh_index,
            0 if word.variant == "primary" else 1,
            int(row["tatoeba_id"]),
        ),
        excluded_words=french_excluded_words,
        extra_manifest=french_extra_manifest,
        attribution=french_attribution,
        finalize_candidate=lambda card, target: card,
        highlight=highlight,
        sense_filter=sense_filter,
    )
)


def gate_word_table(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G1: the reviewed word table passes every S1 check.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    failures = check_word_table(
        table,
        review_path=inputs.review_csv,
    )
    concepts = len({form.swadesh_index for form in table})
    if failures:
        return False, sample(failures)
    return True, f"{concepts} concepts, {len(table)} forms"


def gate_word_boundary(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G9: no card matches the target inside a longer word.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        f"{row['target']} in {row['sentence']}"
        for row in rows
        if token_matches(row["sentence"], row["target"]) != 1
    ]
    if bad:
        return False, f"{len(bad)} card(s) match inside a longer word: {sample(bad)}"
    return True, f"{len(rows)} cards match a whole token"


def gate_attribution(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G12: the French section names every sentence author.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    path = inputs.attribution_path
    if not path.is_file():
        return False, f"attribution file missing: {path}"
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if "## French pack" not in text:
        problems.append("no '## French pack' section")
    authors = {
        row.get("tatoeba_author", "").strip() or "unknown" for row in inputs.rows()
    }
    missing = sorted(author for author in authors if author not in text)
    if missing:
        problems.append(f"{len(missing)} author(s) missing: {sample(missing)}")
    if problems:
        return False, sample(problems)
    return True, f"{len(authors)} author(s) named in the French section"


def gate_typography(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G17: no card field holds a no-break space and all text is NFC.

    ``text_original`` keeps the raw sentence, so only the card fields are read.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    glosses = {form.french: glossary(form) for form in table}
    bad: list[str] = []
    for row in inputs.rows():
        fields = {
            "sentence": row.get("sentence", ""),
            "english": row.get("english", ""),
            "target": row.get("target", ""),
            "gloss": glosses.get(row.get("target", ""), ""),
        }
        for name, value in fields.items():
            if any(char in value for char in SPACE_CHARS):
                bad.append(f"{name}={value}: no-break space")
            elif unicodedata.normalize("NFC", value) != value:
                bad.append(f"{name}={value}: not NFC")
    if bad:
        return False, f"{len(bad)} card field(s) fail the typography: {sample(bad)}"
    return True, f"{len(inputs.rows())} cards hold NFC text with normal spaces"


def gate_no_blocked_license(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G18: a public pack has no card with a blocked recording.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    manifest = inputs.manifest()
    pack_license = manifest.get("pack_license", "public")
    if pack_license != "public":
        return True, f"pack license is {pack_license}, so no recording is blocked"
    blocked = [
        row["sentence"]
        for row in inputs.rows()
        if row.get("audio_source") == "native"
        and row.get("audio_license", "").strip() in ("", "CC BY-NC 4.0")
    ]
    if blocked:
        return False, f"{len(blocked)} card(s) use a blocked license: {sample(blocked)}"
    return True, "no card uses a CC BY-NC 4.0 or empty-license recording"


def gate_discontinuous(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G19: a discontinuous target keeps both parts in the sentence.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    bad: list[str] = []
    checked = 0
    for row in inputs.rows():
        target = row["target"]
        if form_kind(target) != "discontinuous":
            continue
        checked += 1
        tokens = word_tokens(row["sentence"])
        if not _in_order(tokens, form_parts(target)):
            bad.append(f"{target} in {row['sentence']}")
    if bad:
        return False, f"{len(bad)} card(s) lose a part: {sample(bad)}"
    return True, f"{checked} discontinuous card(s) keep both parts"


def gate_noun_gender(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """French gate G20: report the noun rows with an empty gender.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text. A reported count is not a failure.
    """
    table = inputs.table()
    nouns = [
        form
        for form in table
        if form.swadesh_index in NOUN_INDICES
    ]
    empty = sorted(form.french for form in nouns if not form.gender)
    report = inputs.manifest()
    counted = report.get("noun_gender_missing")
    if empty:
        return True, (
            f"{len(empty)} of {len(nouns)} noun row(s) have no gender: "
            f"{sample(empty)}"
        )
    return True, f"{len(nouns)} noun row(s) have a gender (manifest {counted})"


def french_gates() -> tuple[tuple[str, Any], ...]:
    """Give the ordered gate list G1 to G21 of the French pack.

    :returns: Pairs of gate name and gate function.
    """
    from anki_tools.sentence_pack.audit import (
        gate_apkg,
        gate_coverage,
        gate_manifest_counts,
        gate_media_count,
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
        ("G17", gate_typography),
        ("G18", gate_no_blocked_license),
        ("G19", gate_discontinuous),
        ("G20", gate_noun_gender),
        ("G21", gate_no_sense_mismatch),
    )


def run_french_audit(
    *,
    apkg_path: Path | None = None,
    sentences_csv: Path = SENTENCES_PATH,
    word_csv: Path = WORD_CSV_PATH,
    review_csv: Path = WORD_REVIEW_PATH,
    audio_dir: Path = AUDIO_DIR,
    manifest_path: Path | None = None,
    attribution_path: Path = ATTRIBUTION_PATH,
) -> list[GateRow]:
    """Run the French quality gates G1 to G21.

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
        FRENCH,
        apkg_path=apkg_path or FRENCH.output_apkg,
        sentences_csv=sentences_csv,
        word_csv=word_csv,
        review_csv=review_csv,
        audio_dir=audio_dir,
        manifest_path=manifest_path or FRENCH.manifest_path,
        attribution_path=attribution_path,
        load_table=lambda csv_path: load_word_table(csv_path),
    )
    return run_gates(french_gates(), inputs, FRENCH)


__all__ = [
    "FRENCH",
    "Candidate",
    "NOUN_INDICES",
    "WordForm",
    "check_word_table",
    "clean_term",
    "form_kind",
    "form_keys",
    "form_parts",
    "glossary",
    "highlight",
    "hits",
    "load_word_table",
    "parse_module_terms",
    "prepare_text",
    "run_french_audit",
    "script_ok",
    "token_matches",
]
