"""One language deck profile, with a core mode view and a cull mode view.

A language package states the language facts once. The core flow and the cull
flow read their own mode view, and a mode view adds the values that belong to
that mode alone. The two views agree on every language fact by construction, so
the input names, the romanization, the sense file, and the CSS land once per
language.

The seam sits between a language package and the two flows. The core flow and
the cull flow are the two adapters. ADR 0004 keeps its decision: each mode
supplies its own values, for example its templates and its gate limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from anki_tools.ranking.cullflow import CullProfile
from anki_tools.ranking.scan import CoreProfile


def _tokenize_deck(tokenize_texts: Callable[[Iterable[str]], Any]) -> Callable[[Any], Any]:
    """Build the core tokenizer from the one tokenizer of the language.

    :param tokenize_texts: Tokenizer of sentence texts.
    :returns: Tokenizer of the sentence in each card.
    """

    def tokenize_deck(cards: Iterable[Any]) -> list[list[tuple[str, str]]]:
        return tokenize_texts([card.sentence for card in cards])

    return tokenize_deck


@dataclass(frozen=True)
class LanguageDeckProfile:
    """The language facts of one deck family, and the two mode views over them.

    :param load_input_deck: Read the cards of the input deck from an APKG file.
    :param tokenize_texts: Tokenize sentence texts into ``(lemma, part of
        speech)`` pairs. The core view derives its card tokenizer from it.
    :param load_frequency: Read the frequency rows of the language.
    :param frequency_ranks: Rank the lemmas by the blended frequency.
    :param read_names: Read the name stoplist.
    :param input_model_name: Name of the input note type.
    :param input_sentence_field: Name of the sentence field of the input note
        type.
    :param input_english_field: Name of the English field of the input note type.
    :param input_audio_field: Name of the audio field of the input note type.
    :param clean_sentence: Cleaner of the raw sentence field value.
    :param card_css: CSS of the written deck.
    :param sense_path: Sense file of the target words.
    :param highlight: Bold hook, or None when the language does not bold.
    :param romanization_source: Input field of the romanization line, or None
        when the language has none.
    """

    load_input_deck: Callable[[Path], Any]
    tokenize_texts: Callable[[Iterable[str]], list[list[tuple[str, str]]]]
    load_frequency: Callable[[], Any]
    frequency_ranks: Callable[[Any, float], dict[str, int]]
    read_names: Callable[[], set[str]]
    input_model_name: str
    input_sentence_field: str
    input_english_field: str
    input_audio_field: str
    clean_sentence: Callable[[str], str]
    card_css: Callable[[], str]
    sense_path: Path
    highlight: Callable[[str, str], str] | None = None
    romanization_source: str | None = None

    def core(self, **mode: Any) -> CoreProfile:
        """Give the core mode view of the language.

        :param mode: Values that only the core mode states, for example the word
            table, the templates, and the selection settings.
        :returns: The core profile of the flow.
        """
        facts: dict[str, Any] = {
            "load_input_deck": self.load_input_deck,
            "tokenize_deck": _tokenize_deck(self.tokenize_texts),
            "load_frequency": self.load_frequency,
            "frequency_ranks": self.frequency_ranks,
            "read_names": self.read_names,
            "input_model_name": self.input_model_name,
            "input_sentence_field": self.input_sentence_field,
            "input_english_field": self.input_english_field,
            "input_audio_field": self.input_audio_field,
            "card_css": self.card_css,
            "sense_path": self.sense_path,
            "highlight": self.highlight,
            "romanization_source": self.romanization_source,
        }
        facts.update(mode)
        return CoreProfile(**facts)

    def cull(self, **mode: Any) -> CullProfile:
        """Give the cull mode view of the language.

        :param mode: Values that only the cull mode states, for example the
            reserve pack, the marks, and the gate limits.
        :returns: The cull profile of the flow.
        """
        facts: dict[str, Any] = {
            "tokenize_texts": self.tokenize_texts,
            "load_frequency": self.load_frequency,
            "frequency_ranks": self.frequency_ranks,
            "read_names": self.read_names,
            "input_model_name": self.input_model_name,
            "input_sentence_field": self.input_sentence_field,
            "input_english_field": self.input_english_field,
            "input_audio_field": self.input_audio_field,
            "clean_sentence": self.clean_sentence,
            "card_css": self.card_css,
            "sense_path": self.sense_path,
            "highlight": self.highlight,
            "romanization_source": self.romanization_source,
        }
        facts.update(mode)
        return CullProfile(**facts)
