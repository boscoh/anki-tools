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
"""

from anki_tools.apkg.package import AnkiPackage
from anki_tools.languages.mandarin.pinyin import fix_pinyin
from anki_tools.languages.mandarin.rank import (
    complexity_score_zh,
    frequency_score_zh,
    rank_sentences_zh,
)
from anki_tools.languages.spanish.rank import rank_sentences_es
from anki_tools.ranking.rank import (
    FrequencyData,
    Sentence,
    char_similarity_zh,
    extract_sentences,
    reduce_sentences_stratified,
    write_ranking_csv,
)
from anki_tools.ranking.reorder import load_ranking, reorder_deck

__all__ = [
    "AnkiPackage",
    "Sentence",
    "FrequencyData",
    "extract_sentences",
    "rank_sentences_zh",
    "rank_sentences_es",
    "reduce_sentences_stratified",
    "write_ranking_csv",
    "complexity_score_zh",
    "frequency_score_zh",
    "char_similarity_zh",
    "reorder_deck",
    "load_ranking",
    "fix_pinyin",
]
