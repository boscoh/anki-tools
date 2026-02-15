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
from anki_tools.pinyin import fix_pinyin
from anki_tools.rank import (
    FrequencyData,
    Sentence,
    char_similarity_zh,
    complexity_score_zh,
    extract_sentences,
    frequency_score_zh,
    rank_sentences_es,
    rank_sentences_zh,
    reduce_sentences_stratified,
    write_ranking_csv,
)
from anki_tools.reorder import load_ranking, reorder_deck

__all__ = [
    "AnkiPackage",
    "Sentence",
    "FrequencyData",
    "extract_sentences",
    "rank_sentences_zh",
    "write_ranking_csv",
    "complexity_score_zh",
    "frequency_score_zh",
    "char_similarity_zh",
    "reorder_deck",
    "load_ranking",
    "fix_pinyin",
]
