"""Shared scan: extract, tokenize, count, and blend the frequency ranks.

The scan is the first stage of the core flow of spec section 3. A language
package supplies the adapters of a :class:`CoreProfile`, and :func:`scan`
runs the five scan adapters in order. The rest of the profile feeds the shared
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

    The first five fields are the scan adapters, called by :func:`scan` in the
    order below. The remaining fields feed the rest of the core flow.

    :param load_input_deck: Read the cards of the input deck from an APKG file.
    :param tokenize_deck: Tokenize every card into ``(lemma, part of speech)``
        pairs.
    :param load_frequency: Read the frequency rows of the language.
    :param frequency_ranks: Rank the lemmas by the blended frequency.
    :param read_names: Read the name stoplist.
    :param word_table: Read the reviewed Swadesh rows as ``(index, form,
        english gloss)`` triples in table order.
    :param sense_path: TSV file of the tier F sense rows.
    :param loan_path: Text file of the loan word list.
    :param has_orphan: Tell whether a sentence starts with an orphan
        conjunction.
    :param hits: Count the tier S forms that a sentence holds exactly once.
    :param sense_filter: Reject a form whose paired English holds a
        false-friend sense, or None when the language has no such filter.
    :param highlight: Wrap the matched target tokens of a sentence in a bold
        tag, or None when the deck does not bold.
    :param lemma_key: Normalize a lemma before the tier F sense guard.
    :param model_name: Name of the input note model.
    :param sentence_field: Name of the sentence field of the model.
    :param english_field: Name of the English field of the model.
    :param audio_field: Name of the audio field of the model.
    :param core_field: Name of the core word field.
    :param core_gloss_field: Name of the core gloss field.
    :param templates: Card templates of the core deck.
    :param card_css: CSS of the core deck.
    :param min_units: Minimum word count of a card.
    :param max_units: Maximum word count of a card.
    :param default_options: Selection settings of the core cover.
    :param deck_name: Name of the output deck.
    :param frequency_source: Label of the frequency source for the manifest.
    :param lemma_overrides: Lemma repair map of the lemmatizer artefacts,
        applied by :func:`scan` to the tokens of every card.
    :param outside_set_ratio: Largest accepted outside set, as a share of
        the targets.
    :param outside_rare_ratio: Largest accepted rare outside count, as a
        share of the outside set.
    :param spoken_weight: Blend weight of the spoken term, or None when the
        request supplies it.
    :param finalize: Post-processing step of the written deck, called with
        the run, or None when the language needs none.
    """

    load_input_deck: Callable[[Path], Any]
    tokenize_deck: Callable[[Any], Any]
    load_frequency: Callable[[], Any]
    frequency_ranks: Callable[[Any, float], dict[str, int]]
    read_names: Callable[[], set[str]]

    word_table: Callable[[], Sequence[tuple[int, str, str]]]
    sense_path: Path
    loan_path: Path
    has_orphan: Callable[[str], bool]
    hits: Callable[[str, Collection[str]], Mapping[str, int]]
    sense_filter: Callable[[str, str], bool] | None
    highlight: Callable[[str, str], str] | None
    lemma_key: Callable[[str], str]
    model_name: str
    sentence_field: str
    english_field: str
    audio_field: str
    core_field: str
    core_gloss_field: str
    templates: tuple[tuple[str, str], ...]
    card_css: Callable[[], str]
    min_units: int
    max_units: int
    default_options: Any
    deck_name: str
    frequency_source: str
    lemma_overrides: Mapping[str, str] = field(default_factory=dict)
    outside_set_ratio: float = 1.3
    outside_rare_ratio: float = 0.1
    spoken_weight: float | None = None
    finalize: Callable[[Any], None] | None = None


@dataclass(frozen=True)
class ScanResult:
    """Output of one scan.

    :param cards: Input cards, as ``load_input_deck`` returns them.
    :param tokens: Lemma and part of speech per card, as ``tokenize_deck``
        returns them.
    :param ranks: Lemma to blended frequency rank.
    :param names: Name stoplist.
    """

    cards: Any
    tokens: Any
    ranks: dict[str, int]
    names: set[str]


def scan(profile: CoreProfile, apkg_path: Path, spoken_weight: float) -> ScanResult:
    """Run the five scan adapters of the profile in order.

    The lemma overrides of the profile repair the tokens after
    ``tokenize_deck``, so the target list and the sense guard see the
    repaired lemma.

    :param profile: Language hooks of the scan.
    :param apkg_path: Input APKG file.
    :param spoken_weight: Weight of the spoken term in the frequency blend.
    :returns: The cards, the tokens, the ranks, and the names.
    """
    cards = profile.load_input_deck(apkg_path)
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
    return ScanResult(cards=cards, tokens=tokens, ranks=ranks, names=names)
