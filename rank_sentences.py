#!/usr/bin/env python3
"""
Rank Chinese sentences by complexity, word frequency, and uniqueness.

See SPEC_RANK_SENTENCES.md for full specification.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import jieba

from anki_package import AnkiPackage


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Sentence:
    """A Chinese sentence extracted from Anki deck."""

    note_id: int
    chinese: str
    pinyin: str
    english: str
    original_order: int
    complexity_score: float = 0.0
    frequency_score: float = 0.0
    similarity_penalty: float = 0.0
    final_score: float = 0.0


@dataclass
class FrequencyData:
    """Word and character frequency lookup tables."""

    word_freq: dict[str, int] = field(default_factory=dict)
    char_freq: dict[str, int] = field(default_factory=dict)
    hsk_vocab: dict[str, int] = field(default_factory=dict)  # word -> HSK level


# =============================================================================
# Phase 1: Data Extraction
# =============================================================================


def extract_sentences(apkg_path: str, model_id: int | None = None) -> list[Sentence]:
    """
    Extract sentences from an Anki package.

    Args:
        apkg_path: Path to the .apkg file
        model_id: Optional model ID to filter by (extracts all if None)

    Returns:
        List of Sentence objects
    """
    sentences = []

    with AnkiPackage(apkg_path) as pkg:
        models = pkg.get_models()
        decks = pkg.get_decks()
        cards = pkg.get_cards()

        # Debug: show available models
        print(f"Available models in {apkg_path}:")
        for mid, model in models.items():
            print(f"  {mid}: {model['name']} ({len(model['flds'])} fields)")
            for i, fld in enumerate(model["flds"]):
                print(f"    [{i}] {fld['name']}")

        for card in cards:
            card_model_id = str(card["mid"])

            if model_id and card_model_id != str(model_id):
                continue

            model = models.get(card_model_id, {})
            field_names = [f["name"] for f in model.get("flds", [])]
            field_values = card["flds"].split("\x1f")
            fields = dict(zip(field_names, field_values))

            # Extract relevant fields based on spec
            chinese = fields.get("Sentence", fields.get("Front", ""))
            pinyin = fields.get("Sentence (Latin)", fields.get("Pinyin", ""))
            english = fields.get(
                "Sentence (Translation)", fields.get("Translation", "")
            )

            # Try to get original order from [counter 2] field
            counter_str = fields.get("[counter 2]", "0")
            try:
                original_order = int(counter_str) if counter_str else 0
            except ValueError:
                original_order = 0

            if chinese:
                sentences.append(
                    Sentence(
                        note_id=card["nid"],
                        chinese=chinese,
                        pinyin=pinyin,
                        english=english,
                        original_order=original_order,
                    )
                )

    # Deduplicate by note_id (cards can share notes)
    seen = set()
    unique = []
    for s in sentences:
        if s.note_id not in seen:
            seen.add(s.note_id)
            unique.append(s)

    print(f"Extracted {len(unique)} unique sentences")
    return unique


# =============================================================================
# Phase 2: Frequency Data Loading
# =============================================================================


def load_frequency_data(vocab_dir: str = "vocab") -> FrequencyData:
    """
    Load word frequency and HSK vocabulary data.

    Expected files in vocab_dir:
        - subtlex_ch.txt: Tab-separated word frequency list
        - hsk_vocab.json: JSON mapping words to HSK levels

    Returns:
        FrequencyData with loaded lookup tables
    """
    vocab_path = Path(vocab_dir)
    data = FrequencyData()

    # Load SUBTLEX-CH word frequencies
    subtlex_path = vocab_path / "subtlex_ch.txt"
    if subtlex_path.exists():
        with open(subtlex_path, encoding="utf-8") as f:
            for rank, line in enumerate(f, 1):
                parts = line.strip().split("\t")
                if parts:
                    word = parts[0]
                    data.word_freq[word] = rank
        print(f"Loaded {len(data.word_freq)} words from SUBTLEX-CH")
    else:
        print(f"Warning: {subtlex_path} not found, frequency scoring disabled")

    # Load HSK vocabulary
    hsk_path = vocab_path / "hsk_vocab.json"
    if hsk_path.exists():
        with open(hsk_path, encoding="utf-8") as f:
            data.hsk_vocab = json.load(f)
        print(f"Loaded {len(data.hsk_vocab)} HSK words")
    else:
        print(f"Warning: {hsk_path} not found, HSK scoring disabled")

    # Build character frequency from word frequency
    for word, rank in data.word_freq.items():
        for char in word:
            if is_chinese_char(char) and char not in data.char_freq:
                data.char_freq[char] = rank

    return data


def is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character."""
    return "\u4e00" <= char <= "\u9fff"


def get_chinese_chars(text: str) -> list[str]:
    """Extract Chinese characters from text."""
    return [c for c in text if is_chinese_char(c)]


# =============================================================================
# Phase 3: Scoring Functions
# =============================================================================


def complexity_score(sentence: str) -> float:
    """
    Calculate complexity score (0-100) based on structural features.

    Metrics (see SPEC_RANK_SENTENCES.md):
        - Character count (15%)
        - Unique character count (15%)
        - Word count (20%)
        - Average word length (10%)
        - Character stroke count (15%) - TODO
        - HSK level of characters (25%) - TODO

    For now, simplified version using available metrics.
    """
    chars = get_chinese_chars(sentence)
    if not chars:
        return 0.0

    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    char_count = len(chars)
    unique_chars = len(set(chars))
    word_count = len(chinese_words)
    avg_word_len = sum(len(w) for w in chinese_words) / max(len(chinese_words), 1)

    # Normalize to 0-1 range (using reasonable max values)
    char_norm = min(char_count / 30, 1.0)  # Max 30 chars
    unique_norm = min(unique_chars / 20, 1.0)  # Max 20 unique
    word_norm = min(word_count / 15, 1.0)  # Max 15 words
    avg_len_norm = min(avg_word_len / 4, 1.0)  # Max avg 4 chars/word

    # Weighted combination (simplified weights for available metrics)
    score = (
        char_norm * 0.25 + unique_norm * 0.25 + word_norm * 0.30 + avg_len_norm * 0.20
    )

    return score * 100


def get_word_rank(word: str, freq_data: FrequencyData, max_rank: int = 50000) -> int:
    """
    Get frequency rank for a word, falling back to character average if not found.

    If the word isn't in the frequency list (e.g., jieba combined common words),
    calculate average rank of its characters.
    """
    if word in freq_data.word_freq:
        return freq_data.word_freq[word]

    # Fall back to average character rank
    chars = get_chinese_chars(word)
    if not chars:
        return max_rank

    char_ranks = [freq_data.char_freq.get(c, max_rank) for c in chars]
    return int(sum(char_ranks) / len(char_ranks))


def frequency_score(sentence: str, freq_data: FrequencyData) -> float:
    """
    Calculate frequency score (0-100) based on word commonality.

    Higher score = more common vocabulary = easier to learn first.

    Metrics:
        - Average word frequency rank (40%)
        - % words in top 1000 (25%)
        - % words in top 5000 (15%)
        - HSK coverage (20%)
    """
    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    if not chinese_words:
        return 0.0

    if not freq_data.word_freq:
        return 50.0  # Neutral score if no frequency data

    max_rank = 50000
    ranks = [get_word_rank(w, freq_data, max_rank) for w in chinese_words]

    # Average rank (inverted: lower rank = higher score)
    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)

    # Coverage metrics
    top_1000 = sum(1 for r in ranks if r <= 1000) / len(ranks)
    top_5000 = sum(1 for r in ranks if r <= 5000) / len(ranks)

    # HSK coverage (words in HSK 1-4, also check characters)
    if freq_data.hsk_vocab:
        hsk_count = 0
        for w in chinese_words:
            if freq_data.hsk_vocab.get(w, 7) <= 4:
                hsk_count += 1
            else:
                # Check if all characters are in HSK 1-4
                chars = get_chinese_chars(w)
                if chars and all(freq_data.hsk_vocab.get(c, 7) <= 4 for c in chars):
                    hsk_count += 1
        hsk_coverage = hsk_count / len(chinese_words)
    else:
        hsk_coverage = 0.5  # Neutral

    # Weighted combination
    score = avg_score * 0.40 + top_1000 * 0.25 + top_5000 * 0.15 + hsk_coverage * 0.20

    return score * 100


# =============================================================================
# Phase 4: Similarity Analysis
# =============================================================================


def char_similarity(sent_a: str, sent_b: str) -> float:
    """
    Calculate Jaccard similarity between character sets.

    Returns value between 0 (no overlap) and 1 (identical).
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))

    if not chars_a or not chars_b:
        return 0.0

    intersection = len(chars_a & chars_b)
    union = len(chars_a | chars_b)

    return intersection / union


def compute_similarity_penalties(
    sentences: list[Sentence], top_n: int = 100
) -> list[float]:
    """
    Compute similarity penalties for each sentence.

    For each sentence, find maximum similarity to higher-ranked sentences
    (those with better preliminary scores) and apply penalty.

    Args:
        sentences: List of sentences sorted by preliminary score (best first)
        top_n: Number of higher-ranked sentences to check (for efficiency)

    Returns:
        List of penalties (0-50 scale) in same order as input
    """
    penalties = []

    for i, sent in enumerate(sentences):
        if i == 0:
            penalties.append(0.0)
            continue

        # Check similarity to top_n higher-ranked sentences
        higher_ranked = sentences[: min(i, top_n)]
        max_sim = max(char_similarity(sent.chinese, h.chinese) for h in higher_ranked)

        # Penalty scales from 0 to 50 based on similarity
        penalty = max_sim * 50
        penalties.append(penalty)

    return penalties


# =============================================================================
# Phase 5: Ranking and Output
# =============================================================================


def rank_sentences(
    sentences: list[Sentence],
    freq_data: FrequencyData,
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """
    Rank sentences by combined score.

    Default weights:
        - complexity: 0.3 (simpler first)
        - frequency: 0.5 (common vocab first)
        - similarity_penalty: 0.2 (variety)

    Note: Complexity is inverted (lower complexity = higher score for learning)
    """
    if weights is None:
        weights = {
            "complexity": 0.3,
            "frequency": 0.5,
            "similarity_penalty": 0.2,
        }

    # Calculate complexity and frequency scores
    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score(sent.chinese)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score(sent.chinese, freq_data)

    # Sort by preliminary score (frequency - complexity) for similarity calculation
    # Higher frequency + lower complexity = better for beginners
    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    # Calculate similarity penalties
    print("Calculating similarity penalties...")
    penalties = compute_similarity_penalties(sentences)
    for sent, penalty in zip(sentences, penalties):
        sent.similarity_penalty = penalty

    # Recalculate final score with penalties
    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    # Final sort
    sentences.sort(key=lambda s: s.final_score, reverse=True)

    return sentences


def export_csv(sentences: list[Sentence], output_path: str) -> None:
    """Export ranked sentences to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "sentence",
                "pinyin",
                "english",
                "original_order",
                "complexity",
                "frequency",
                "similarity_penalty",
                "final_score",
            ]
        )

        for rank, sent in enumerate(sentences, 1):
            writer.writerow(
                [
                    rank,
                    sent.chinese,
                    sent.pinyin,
                    sent.english,
                    sent.original_order,
                    f"{sent.complexity_score:.1f}",
                    f"{sent.frequency_score:.1f}",
                    f"{sent.similarity_penalty:.1f}",
                    f"{sent.final_score:.2f}",
                ]
            )

    print(f"Exported {len(sentences)} sentences to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank Chinese sentences by complexity and frequency"
    )
    parser.add_argument("apkg", help="Path to .apkg file")
    parser.add_argument(
        "-o", "--output", default="ranked_sentences.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--model-id", type=int, help="Filter by specific model ID"
    )
    parser.add_argument(
        "--complexity-weight", type=float, default=0.3, help="Complexity weight"
    )
    parser.add_argument(
        "--frequency-weight", type=float, default=0.5, help="Frequency weight"
    )
    parser.add_argument(
        "--similarity-weight", type=float, default=0.2, help="Similarity penalty weight"
    )
    parser.add_argument(
        "--top-n", type=int, help="Only output top N sentences"
    )
    parser.add_argument(
        "--vocab-dir", default="vocab", help="Directory containing frequency data"
    )

    args = parser.parse_args()

    # Phase 1: Extract sentences
    sentences = extract_sentences(args.apkg, args.model_id)
    if not sentences:
        print("No sentences found!")
        return

    # Phase 2: Load frequency data
    freq_data = load_frequency_data(args.vocab_dir)

    # Phases 3-5: Score and rank
    weights = {
        "complexity": args.complexity_weight,
        "frequency": args.frequency_weight,
        "similarity_penalty": args.similarity_weight,
    }
    ranked = rank_sentences(sentences, freq_data, weights)

    # Limit output if requested
    if args.top_n:
        ranked = ranked[: args.top_n]

    # Export
    export_csv(ranked, args.output)


if __name__ == "__main__":
    main()
