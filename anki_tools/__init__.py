"""
Anki Tools - utilities for processing Anki packages.

Core classes:
    AnkiPackage - Read, edit, and create .apkg files

Modules:
    package  - Core AnkiPackage class
    rank     - Sentence ranking by frequency/complexity
    reorder  - Deck reordering based on ranking
    pinyin   - Pinyin correction and formatting
    cli      - Command-line interface
"""

from anki_tools.package import AnkiPackage
from anki_tools.rank import (
    Sentence,
    FrequencyData,
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
    complexity_score,
    frequency_score,
    char_similarity,
)
from anki_tools.reorder import reorder_deck, load_ranking
from anki_tools.pinyin import fix_pinyin, check_pinyin

__all__ = [
    "AnkiPackage",
    "Sentence",
    "FrequencyData",
    "extract_sentences",
    "load_frequency_data",
    "rank_sentences",
    "export_csv",
    "complexity_score",
    "frequency_score",
    "char_similarity",
    "reorder_deck",
    "load_ranking",
    "fix_pinyin",
    "check_pinyin",
]
