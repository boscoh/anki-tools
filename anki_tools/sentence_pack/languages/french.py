"""French language profile of the shared sentence pipeline.

The French deck is not built in this task. The profile holds the French values
of the French spec so that the shared stages can run once the reviewed word CSV
and the review file exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

from anki_tools.rank import latin_text_dedup_key, word_similarity_fr
from anki_tools.sentence_pack.profile import LanguageProfile, register
from anki_tools.sentence_pack.tokens import has_finite_verb, tokenize, words

WORD_CSV_PATH = Path("vocab/english-french.csv")
SENTENCES_PATH = Path("vocab/french_sentences.csv")
SENTENCE_REVIEW_PATH = Path("vocab/french_sentences_review.csv")
POOL_REPORT_PATH = Path("data/french_pool_report.json")
AUDIO_DIR = Path("audio/french_sentences")
WORD_REVIEW_PATH = Path("vocab/french_word_review.csv")

MODEL_NAME = "French Swadesh Sentence"
DECK_NAME = "French Swadesh Sentences"
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
DECK_DESCRIPTION = "Word data CC BY-SA 4.0. Sentence text CC BY 2.0 FR."
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

FINAL_MARKS = ".?!"
INTERNAL_MARKS = ",;:"
QUOTE_MARKS = '«»"“”—–'
ORPHAN_CONJUNCTIONS = (
    "et",
    "mais",
    "ou",
    "car",
    "parce que",
    "donc",
    "pourtant",
)


def normalized(text: str) -> str:
    """Normalize French text for comparison.

    :param text: Text to normalize.
    :returns: NFC, casefolded text with collapsed spaces.
    """
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def token_words(sentence: str) -> list[str]:
    """Give the word tokens of a French sentence.

    :param sentence: Sentence text.
    :returns: Token texts with punctuation removed.
    """
    return words(TOKENIZER, sentence)


def form_keys(form: str) -> frozenset[str]:
    """Build the target keys of one French form.

    :param form: Word-table form.
    :returns: Normalized surfaces of the form.
    """
    return frozenset({normalized(form)})


def _key_index(forms: frozenset[str]) -> dict[str, frozenset[str]]:
    index: dict[str, set[str]] = {}
    for form in forms:
        for key in form_keys(form):
            index.setdefault(key, set()).add(form)
    return {key: frozenset(value) for key, value in index.items()}


def hits(sentence: str, forms: Collection[str]) -> dict[str, int]:
    """Count the forms that the sentence contains exactly one time.

    :param sentence: Sentence text.
    :param forms: Word-table forms to look for.
    :returns: Mapping from form to 1, for forms with one token match.
    """
    index = _key_index(frozenset(forms))
    counts: dict[str, int] = {}
    for token in tokenize(TOKENIZER, sentence):
        if token.punctuation:
            continue
        for key in {normalized(token.text), normalized(token.lemma)}:
            for form in index.get(key, ()):
                counts[form] = counts.get(form, 0) + 1
    return {form: count for form, count in counts.items() if count == 1}


def match_count(sentence: str, form: str) -> int:
    """Count the token matches of one form.

    :param sentence: Sentence text.
    :param form: Word-table form.
    :returns: Token match count.
    """
    found = hits(sentence, frozenset({form}))
    return found.get(form, 0)


def count_unknown(sentence: str, known: Collection[str]) -> int:
    """Count the tokens outside the known word set.

    :param sentence: Sentence text.
    :param known: Known word set.
    :returns: Unknown token count.
    """
    keys = {normalized(word) for word in known}
    return sum(
        1
        for token in tokenize(TOKENIZER, sentence)
        if not token.punctuation
        and normalized(token.text) not in keys
        and normalized(token.lemma) not in keys
    )


def has_orphan(sentence: str) -> bool:
    """Check filter F10: a sentence-initial conjunction with no comma.

    :param sentence: Sentence text.
    :returns: True when the leading conjunction has no main clause.
    """
    if "," in sentence:
        return False
    return normalized(sentence).startswith(ORPHAN_CONJUNCTIONS)


def _form_positions(table: Sequence[Any]) -> dict[str, tuple[int, int]]:
    return {
        form.french: (form.swadesh_index, 0 if form.variant == "primary" else 1)
        for form in table
    }


def _excluded_words(table: Sequence[Any], targets: Collection[str]) -> list[str]:
    wanted = set(targets)
    return [form.french for form in table if form.french not in wanted]


def _unbuilt(
    rows: Sequence[dict[str, str]],
    voice: str,
    authors: Any,
    path: Any = None,
) -> None:
    raise RuntimeError("the French pack build is not part of this task")


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
        core_concepts=207,
        zipf_min=3.0,
        similarity_limit=0.67,
        coverage_min=0.95,
        prepare_text=normalized,
        unit_count=lambda sentence: len(token_words(sentence)),
        script_ok=lambda sentence: not any(
            char.isascii() and char.isdigit() for char in sentence
        ),
        is_repeated=lambda sentence: False,
        has_predicate=lambda sentence: has_finite_verb(
            tokenize(TOKENIZER, sentence)
        ),
        has_orphan=has_orphan,
        hits=hits,
        match_count=match_count,
        candidate_length=lambda card: card.word_len,
        count_unknown=count_unknown,
        zipf_words=token_words,
        similarity=word_similarity_fr,
        dedupe_key=latin_text_dedup_key,
        similarity_keys=lambda sentence: set(token_words(sentence)),
        form_text=lambda word: word.french,
        form_positions=_form_positions,
        gloss=lambda word: word.french,
        card_fields=lambda row, word: {},
        card_order=lambda word, row: (
            word.swadesh_index,
            0 if word.variant == "primary" else 1,
            int(row["tatoeba_id"]),
        ),
        excluded_words=_excluded_words,
        extra_manifest=lambda rows: {},
        attribution=_unbuilt,
        finalize_candidate=lambda card, target: card,
    )
)
