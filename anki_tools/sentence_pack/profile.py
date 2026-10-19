"""Language profile of a Swadesh sentence pack.

A :class:`LanguageProfile` holds every language-specific value and function that
the shared stages need: file paths, word window, punctuation, tokenizer name,
`wordfreq` code, TTS voice, the native-audio switch, and the hook functions for
tokenization, target matching, similarity, and the card gloss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Protocol, Sequence


class WordRow(Protocol):
    """A row of a reviewed Swadesh word table."""

    swadesh_index: int
    english: str
    variant: str
    tier: str
    candidate_count: int
    zipf: float


@dataclass(frozen=True)
class LanguageProfile:
    """Every language-specific value of one Swadesh sentence pack.

    :param name: Profile name, for example ``greek``.
    :param lang_tag: Short language tag for the ``lang::`` card tag.
    :param tatoeba_lang: Tatoeba language code, for example ``ell``.
    :param english_lang: Tatoeba code of the translation language.
    :param word_csv: Reviewed word CSV, the S1 input.
    :param sentence_csv: Card CSV written by S3.
    :param sentence_review_csv: Review CSV written by S3.
    :param pool_report: JSON pool report written by S3.
    :param audio_dir: Directory of the MP3 files and ``index.csv``.
    :param word_review_csv: Word review CSV that must have no open row.
    :param deck_name: Anki deck name.
    :param model_name: Anki note model name.
    :param output_apkg: APKG file written by S5.
    :param manifest_path: Manifest JSON written by S5.
    :param fields: Note fields in display order.
    :param sort_field_index: Index of the sort field.
    :param front_template: Front card template.
    :param back_template: Back card template.
    :param deck_description: Deck description text.
    :param licenses: License entries of the manifest, as ``(data, license)`` pairs.
    :param extra_css: Card CSS classes of the pack.
    :param min_units: Minimum word count of a core card.
    :param max_units: Maximum word count of a core card.
    :param unit_name: Name of a sentence unit, ``han`` or ``word``.
    :param fallback_min_units: Minimum word count of a scaffold card.
    :param preferred_min_units: First word count of the preferred range.
    :param preferred_max_units: Last word count of the preferred range.
    :param final_marks: Accepted sentence-final marks.
    :param internal_marks: Internal punctuation marks of filter F8.
    :param quote_marks: Quotation marks and dialogue dashes of filter F7.
    :param comma_marks: Marks that rescue a sentence-initial conjunction.
    :param orphan_conjunctions: Sentence-initial conjunctions of filter F10.
    :param wordfreq_lang: `wordfreq` language code.
    :param tokenizer: spaCy model name, empty when the language has no model.
    :param native_audio: True when a native Tatoeba tier exists.
    :param prefer_native_audio: True when a recording outranks TTS in the rank key.
    :param voice: Default TTS voice.
    :param tts_lang: gTTS language code.
    :param gtts_voice: Voice value of a gTTS file.
    :param swadesh_url: Canonical Swadesh module page of the language.
    :param swadesh_license: License of the word data.
    :param tatoeba_license: License of the sentence text.
    :param core_concepts: Concept count of the Swadesh list.
    :param zipf_min: Zipf value below which a form is rare.
    :param similarity_limit: Jaccard value at which two cards are duplicates.
    :param coverage_min: Minimum core coverage of gate G10.
    :param prepare_text: Hook that normalizes a raw sentence.
    :param unit_count: Hook that counts the words of a sentence.
    :param script_ok: Hook for filter F3, the script test.
    :param is_repeated: Hook for filter F4, the repeated-pattern test.
    :param has_predicate: Hook for filter F9, the predicate test.
    :param has_orphan: Hook for filter F10, the orphan-conjunction test.
    :param hits: Hook that counts the one-time target forms of a sentence.
    :param match_count: Hook that counts the target matches of a sentence.
    :param candidate_length: Hook that gives the unit length of a card.
    :param count_unknown: Hook that counts the tokens outside the known set.
    :param zipf_words: Hook that gives the words of the mean Zipf value.
    :param similarity: Hook for the word-set Jaccard value.
    :param dedupe_key: Hook for the exact dedupe key of filter F12.
    :param similarity_keys: Hook that buckets sentences for the Jaccard guard.
    :param form_text: Hook that gives the card target string of a form.
    :param form_positions: Hook that gives each form its index and variant rank.
    :param gloss: Hook that builds the card gloss of a form.
    :param card_fields: Hook that builds the language-specific note fields.
    :param extra_manifest: Hook that adds language-specific manifest keys.
    :param attribution: Hook that writes the attribution file, given the
        rows, the voice, the audio authors, and the output path.
    :param card_order: Hook that orders the notes of the deck.
    :param excluded_words: Hook that lists the forms without a card, given the
        word table and the card target set.
    :param finalize_candidate: Hook that fills the language-specific card fields.
    :param highlight: Hook that marks the matched target tokens of a card
        sentence, None to wrap the first target substring.
    :param char_links: True when the back template links each Han character
        to the CC-Canto dictionary.
    """

    name: str
    lang_tag: str
    tatoeba_lang: str
    english_lang: str
    word_csv: Path
    sentence_csv: Path
    sentence_review_csv: Path
    pool_report: Path
    audio_dir: Path
    word_review_csv: Path
    deck_name: str
    model_name: str
    output_apkg: Path
    manifest_path: Path
    fields: tuple[str, ...]
    sort_field_index: int
    front_template: str
    back_template: str
    deck_description: str
    licenses: tuple[tuple[str, str], ...]
    extra_css: str
    min_units: int
    max_units: int
    unit_name: str
    fallback_min_units: int
    preferred_min_units: int
    preferred_max_units: int
    final_marks: str
    internal_marks: str
    quote_marks: str
    comma_marks: str
    orphan_conjunctions: tuple[str, ...]
    wordfreq_lang: str
    tokenizer: str
    native_audio: bool
    prefer_native_audio: bool
    voice: str
    tts_lang: str
    gtts_voice: str
    swadesh_url: str
    swadesh_license: str
    tatoeba_license: str
    core_concepts: int
    zipf_min: float
    similarity_limit: float
    coverage_min: float
    prepare_text: Callable[[str], str]
    unit_count: Callable[[str], int]
    script_ok: Callable[[str], bool]
    is_repeated: Callable[[str], bool]
    has_predicate: Callable[[str], bool]
    has_orphan: Callable[[str], bool]
    hits: Callable[[str, Collection[str]], dict[str, int]]
    match_count: Callable[[str, str], int]
    candidate_length: Callable[[Any], int]
    count_unknown: Callable[[str, Collection[str]], int]
    zipf_words: Callable[[str], list[str]]
    similarity: Callable[[str, str], float]
    dedupe_key: Callable[[str], str]
    similarity_keys: Callable[[str], set[str]]
    form_text: Callable[[Any], str]
    form_positions: Callable[[Sequence[Any]], dict[str, tuple[int, int]]]
    gloss: Callable[[Any], str]
    card_fields: Callable[[dict[str, str], Any], dict[str, str]]
    extra_manifest: Callable[[Sequence[dict[str, str]]], dict[str, Any]]
    attribution: Callable[..., None]
    card_order: Callable[[Any, dict[str, str]], tuple]
    excluded_words: Callable[[Sequence[Any], Collection[str]], list[str]]
    finalize_candidate: Callable[[Any, str], Any]
    highlight: Callable[[str, str], str] | None = None
    char_links: bool = False

    def target_length(self, candidate: Any) -> int:
        """Give the unit length of a candidate.

        :param candidate: Candidate sentence.
        :returns: Word or character count of the sentence.
        """
        return self.candidate_length(candidate)

    def english_bounds(self) -> tuple[int, int]:
        """Give the accepted English word-count window of filter F11.

        :returns: Minimum and maximum English word count.
        """
        return (2, 12)


def excluded_forms(
    profile: LanguageProfile, table: Sequence[Any], targets: Collection[str]
) -> list[str]:
    """List the word-table forms that get no card.

    :param profile: Language profile.
    :param table: Reviewed word table.
    :param targets: Card target strings of the pack.
    :returns: Forms that the pack leaves out.
    """
    return profile.excluded_words(table, targets)


def duplicate_limit(profile: LanguageProfile) -> float:
    """Give the Jaccard limit at which two cards are duplicates.

    :param profile: Language profile.
    :returns: Jaccard value of the duplicate guard.
    """
    return profile.similarity_limit


PROFILES: dict[str, LanguageProfile] = {}


def register(profile: LanguageProfile) -> LanguageProfile:
    """Add a profile to the registry.

    :param profile: Profile of one language.
    :returns: The same profile, for use as a module constant.
    """
    PROFILES[profile.name] = profile
    return profile


def get_profile(name: str) -> LanguageProfile:
    """Look up a registered profile.

    :param name: Language name, for example ``greek``.
    :raises KeyError: The name has no registered profile.
    :returns: The language profile.
    """
    if name not in PROFILES:
        raise KeyError(f"no sentence pack profile: {name}")
    return PROFILES[name]


def profile_names() -> list[str]:
    """List the registered profile names.

    :returns: Sorted names.
    """
    return sorted(PROFILES)
