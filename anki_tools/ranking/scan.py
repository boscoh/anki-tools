"""Shared scan: extract, tokenize, count, and blend the frequency ranks.

The scan is the first stage of the core flow of spec section 3. A language
package supplies the adapters of a :class:`CoreProfile`, and :func:`scan`
runs the four scan adapters in order. The rest of the profile feeds the shared
target-list builders of ``anki_tools/ranking/corelist.py`` and the shared core
flow of ``anki_tools/ranking/coreflow.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence


@dataclass(frozen=True)
class CoreProfile:
    """Language hooks of the shared core flow.

    The first four fields are the scan adapters, called by :func:`scan` in the
    order below. The remaining fields feed the rest of the core flow.

    :param tokenize_deck: Tokenize every card into ``(lemma, part of speech)``
        pairs.
    :param load_frequency: Read the frequency rows of the language.
    :param frequency_ranks: Rank the lemmas by the blended frequency.
    :param read_names: Read the name stoplist.
    :param word_table: Read the reviewed Swadesh rows as ``(index, form,
        english gloss)`` triples in table order.
    :param sense_path: TSV file of the sense rows.
    :param hits: Count the tier S forms that a sentence holds exactly once.
    :param highlight: Wrap the matched target tokens of a sentence in a bold
        tag, or None when the deck does not bold.
    :param lemma_key: Normalize a lemma before the tier F sense guard.
    :param input_model_name: Name of the input note model.
    :param input_sentence_field: Name of the sentence field of the input model.
    :param input_english_field: Name of the English field of the input model.
    :param input_audio_field: Name of the audio field of the input model.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :param templates: Card templates of the core deck.
    :param card_css: CSS of the core deck.
    :param min_units: Minimum word count of a card.
    :param max_units: Maximum word count of a card.
    :param final_marks: Accepted sentence-final marks of the language.
    :param quote_marks: Quotation marks and dialogue dashes of the language.
    :param internal_marks: Internal punctuation marks of the language.
    :param script_ok: Alphabet test of the ``script`` rule.
    :param is_repeated: Test of the ``repeat`` rule.
    :param register_ok: Test of the ``register`` rule, or None to skip it.
    :param dedupe_key: Key of the ``unique`` rule.
    :param unit_count: Unit count of a sentence, the same count as the sentence
        pack.
    :param default_options: Selection settings of the core cover.
    :param frequency_source: Label of the frequency source for the manifest.
    :param coverage_floor: Lowest share of core targets that hold at least one
        kept card. Gate K23 reads it. It is a ratchet: set it to the current
        share, rounded down to 0.05.
    :param recount_model: spaCy model that gate K24 lemmatizes the card text
        with, for a language whose tier S headwords inflect. Empty when the
        forms appear unchanged, and K24 matches them as substrings.
    :param lemma_overrides: Lemma repair map of the lemmatizer artefacts,
        applied by :func:`scan` to the tokens of every card.
    :param finalize: Post-processing step of the written deck, called with
        the run, or None when the language needs none.
    :param romanization_source: Input field of the romanization line, for
        example ``Jyutping`` or ``Sentence (Latin)``. The output field is
        ``Romanization``. None when the language has no romanization.
    """
    tokenize_deck: Callable[[Any], Any]
    load_frequency: Callable[[], Any]
    frequency_ranks: Callable[[Any, float], dict[str, int]]
    read_names: Callable[[], set[str]]

    word_table: Callable[[], Sequence[tuple[int, str, str]]]
    sense_path: Path
    hits: Callable[[str, Collection[str]], Mapping[str, int]]
    highlight: Callable[[str, str], str] | None
    lemma_key: Callable[[str], str]
    input_model_name: str
    input_sentence_field: str
    input_english_field: str
    input_audio_field: str
    clean_sentence: Callable[[str], str]
    templates: tuple[tuple[str, str], ...]
    card_css: Callable[[], str]
    min_units: int
    max_units: int
    final_marks: str
    quote_marks: str
    internal_marks: str
    script_ok: Callable[[str], bool]
    is_repeated: Callable[[str], bool]
    dedupe_key: Callable[[str], str]
    unit_count: Callable[[str], int]
    default_options: Any
    frequency_source: str
    coverage_floor: float
    recount_model: str = ""
    lemma_overrides: Mapping[str, str] = field(default_factory=dict)
    finalize: Callable[[Any], None] | None = None
    romanization_source: str | None = None
    register_ok: Callable[[str], bool] | None = None


@dataclass(frozen=True)
class ScanResult:
    """Output of one scan.

    :param tokens: Lemma and part of speech per card, as ``tokenize_deck``
        returns them.
    :param ranks: Lemma to blended frequency rank.
    :param names: Name stoplist.
    """

    tokens: Any
    ranks: dict[str, int]
    names: set[str]


def scan(
    profile: CoreProfile, cards: Sequence[Any], spoken_weight: float
) -> ScanResult:
    """Run the scan adapters over the input cards.

    The lemma overrides of the profile repair the tokens after
    ``tokenize_deck``, so the target list and the sense guard see the
    repaired lemma.

    :param profile: Language hooks of the scan.
    :param cards: Input cards that have an audio file.
    :param spoken_weight: Weight of the spoken term in the frequency blend.
    :returns: The tokens, the ranks, and the names.
    """
    tokens = profile.tokenize_deck(cards)
    overrides = profile.lemma_overrides
    if overrides:
        tokens = [
            [(overrides.get(lemma, lemma), pos) for lemma, pos in card]
            for card in tokens
        ]
    frequency = profile.load_frequency()
    ranks = profile.frequency_ranks(frequency, spoken_weight)
    names = profile.read_names()
    return ScanResult(tokens=tokens, ranks=ranks, names=names)
