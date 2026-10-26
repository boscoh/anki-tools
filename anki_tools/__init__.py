"""
Anki Tools - utilities for processing Anki packages.

Core classes:
    AnkiPackage - Read, edit, and create .apkg files

Modules:
    apkg.package    - Core AnkiPackage class
    apkg.templates  - Card template partials
    ranking.rank    - Shared sentence ranking helpers
    ranking.latin   - Shared Latin-script text helpers
    ranking.reorder - Deck reordering based on ranking
    languages       - One package per language: profile, rank, frequency, cull
    sentence_pack   - Shared Swadesh sentence pack stages
    cli             - Command-line interface

The language helpers resolve on first use. Importing this package loads no
language package, so the shared modules stay free of language code.
"""

from importlib import import_module
from typing import Any

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.rank import (
    FrequencyData,
    Sentence,
    extract_sentences,
    write_ranking_csv,
)
from anki_tools.ranking.reorder import load_ranking, reorder_deck

_LAZY: dict[str, tuple[str, str]] = {
    "fix_pinyin": ("anki_tools.languages.mandarin.pinyin", "fix_pinyin"),
    "char_similarity_zh": (
        "anki_tools.languages.mandarin.characters",
        "char_similarity_zh",
    ),
    "complexity_score_zh": (
        "anki_tools.languages.mandarin.rank",
        "complexity_score_zh",
    ),
    "frequency_score_zh": ("anki_tools.languages.mandarin.rank", "frequency_score_zh"),
    "rank_sentences_zh": ("anki_tools.languages.mandarin.rank", "rank_sentences_zh"),
    "rank_sentences_es": ("anki_tools.languages.spanish.rank", "rank_sentences_es"),
}


def __getattr__(name: str) -> Any:
    """Give a language helper, and load its language package on first use.

    :param name: Name of the public export.
    :returns: The attribute of the language module.
    :raises AttributeError: The name is not a public export.
    """
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'anki_tools' has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names, including the lazy language helpers."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "AnkiPackage",
    "Sentence",
    "FrequencyData",
    "extract_sentences",
    "rank_sentences_zh",
    "rank_sentences_es",
    "write_ranking_csv",
    "complexity_score_zh",
    "frequency_score_zh",
    "char_similarity_zh",
    "reorder_deck",
    "load_ranking",
    "fix_pinyin",
]
